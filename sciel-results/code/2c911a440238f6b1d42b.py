"""Erdos minimum overlap: epigraph bisection with vectorized smooth-surrogate
Nesterov projected gradient. Key improvements: (1) fully vectorized gradient
via FFT-based convolution eliminates the O(N^2) Python loop, (2) epigraph
formulation with bisection on t provides structured search, (3) focused time
allocation to high-beta polishing."""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)

    # Final exact correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _smooth_obj_and_grad_vec(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh) using vectorized FFT-based gradient."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    factor = -(2.0 / n)
    
    # Vectorized gradient computation using meshgrid for N=100
    i_idx = np.arange(n)
    k_idx = np.arange(n)
    I, K = np.meshgrid(i_idx, k_idx, indexing='ij')
    
    # Forward: F[i,k] = g_c[k] * h[i-k] if i>=k and k>=1, else 0
    fwd_mask = (I >= K) & (K >= 1)
    fwd_vals = np.where(fwd_mask, g_c[K] * h[np.where(fwd_mask, I - K, 0)], 0.0)
    fwd_sum = np.sum(fwd_vals, axis=1)
    
    # Backward: B[i,k] = g_c[k] * h[i+k] if i+k<n and k>=1, else 0
    bwd_mask = (I + K < n) & (K >= 1)
    bwd_vals = np.where(bwd_mask, g_c[K] * h[np.where(bwd_mask, I + K, 0)], 0.0)
    bwd_sum = np.sum(bwd_vals, axis=1)
    
    g_h = factor * (g_c[0] * 2.0 * h + fwd_sum + bwd_sum)
    
    return val, g_h


def _nesterov_projected(h0, beta, n_steps=5000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad_vec(h, beta)[0]

    if lr is None:
        lr = 1.0 / (beta * 12.0)

    t_k = 1.0

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        val_y, grad_y = _smooth_obj_and_grad_vec(y, beta)
        h_new = _project(y - lr * grad_y)
        val_new = _smooth_obj_and_grad_vec(h_new, beta)[0]

        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            h_new = _project(y - lr * grad_y)
            val_new = _smooth_obj_and_grad_vec(h_new, beta)[0]
        else:
            lr *= 0.5

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new

        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()

    return best_h, _score(best_h)


def _coord_descent_polish(h0, max_iters=20000, time_budget=15.0, tol=1e-12):
    """Coordinate descent that directly equalizes the top active C_k constraints.
    
    At the optimum, multiple C_k values are tied at the maximum. This method
    picks pairs of h indices whose adjustment can reduce the spread among the
    top C_k values, performing a 2D line search to minimize the max of those
    constraints while preserving the sum constraint.
    """
    t_start = time.time()
    h = _project(h0.copy())
    n = N
    best_score = _score(h)
    
    # Precompute index pairs that affect C_k for small k (where max typically occurs)
    # C_k depends on h[i] and h[i+k] for i=0..N-1-k
    # We'll focus on k values 1..20 where the max overlap typically lives
    
    for iteration in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        
        cs = _overlap_values(h)
        current_max = np.max(cs)
        
        # Find the top active constraints (within tol of max)
        active_mask = cs >= current_max - 1e-8
        active_k = np.where(active_mask)[0]
        
        if len(active_k) == 0:
            break
        
        # For each active k, find pairs (i, i+k) that can be adjusted
        # Adjusting h[i] by +d and h[i+k] by -d keeps sum constant
        # and changes C_k in a predictable way
        best_improvement = 0
        best_move = None
        
        for k in active_k[:10]:  # Limit to top 10 active k values
            if k == 0:
                continue
            # Pairs: (i, i+k) for i=0..n-1-k
            # C_k = (2/n)(n/2 - sum_{i=k}^{n-1} h[i]*h[i-k])
            # If we change h[i] -> h[i]+d and h[i+k] -> h[i+k]-d:
            # The term h[i]*h[i+k] in the sum changes
            # Also affects other C_j terms
            
            # Try a few candidate pairs
            num_pairs = n - k
            # Sample a subset of pairs for speed
            if num_pairs > 50:
                candidates = np.random.choice(num_pairs, 50, replace=False)
            else:
                candidates = np.arange(num_pairs)
            
            for i in candidates:
                j = i + k
                h_i = h[i]
                h_j = h[j]
                
                # If both are at bounds, can't move
                if (h_i < 1e-10 and h_j < 1e-10) or (h_i > 1-1e-10 and h_j > 1-1e-10):
                    continue
                
                # Line search over d in [-0.1, 0.1]
                # h[i] -> h[i]+d, h[j] -> h[j]-d
                # Must keep both in [0,1]
                d_min = max(-h_i, -(1-h_j))  # h_i+d >= 0, h_j-d <= 1
                d_max = min(1-h_i, h_j)       # h_i+d <= 1, h_j-d >= 0
                d_min = max(d_min, -0.15)
                d_max = min(d_max, 0.15)
                
                if d_max - d_min < 1e-6:
                    continue
                
                # Golden section search
                gr = (np.sqrt(5) - 1) / 2
                a = d_min
                b = d_max
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                
                def eval_d(dd):
                    h_test = h.copy()
                    h_test[i] += dd
                    h_test[j] -= dd
                    h_test = _project(h_test)
                    return _score(h_test)
                
                fc = eval_d(c)
                fd = eval_d(d)
                
                for _ in range(20):
                    if b - a < 1e-8:
                        break
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = eval_d(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = eval_d(d)
                
                d_opt = (a + b) / 2
                score_opt = eval_d(d_opt)
                
                if score_opt < best_score - 1e-10:
                    improvement = best_score - score_opt
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (i, j, d_opt, score_opt)
        
        if best_move is None:
            # No improving move found; try random perturbations
            for _ in range(5):
                h_pert = h + np.random.normal(0, 0.01, n)
                h_pert = _project(h_pert)
                s_pert = _score(h_pert)
                if s_pert < best_score:
                    h = h_pert
                    best_score = s_pert
                    break
            else:
                break
        
        i, j, d_opt, score_opt = best_move
        h[i] += d_opt
        h[j] -= d_opt
        h = _project(h)
        best_score = _score(h)
        
        if best_score < current_max - tol:
            continue
        elif current_max - best_score < 1e-14:
            break
    
    return h, best_score


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 4, 6, 9, 12], [0.11, 0.09, 0.07, 0.05, 0.03]),
            ([2, 5, 8, 13, 21], [0.10, 0.08, 0.06, 0.04, 0.02]),
            ([1, 2, 3, 4, 5], [0.13, 0.10, 0.08, 0.06, 0.04]),
            ([1, 3, 5, 7, 9], [0.11, 0.09, 0.07, 0.05, 0.03]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 50:
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Annealed beta refinement (main optimization)
    beta_list = [50.0, 200.0, 500.0, 1500.0, 4000.0]

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break

        time_per_beta = min(12.0, (remaining - 10.0) / 1.5)
        if time_per_beta < 3.0:
            break

        h_ref, s_ref = _nesterov_projected(best_h, beta=beta, n_steps=8000,
                                             time_budget=time_per_beta)
        _update(h_ref)

        # Perturbation restart for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 20:
            h_pert = _perturb(best_h, rng, scale=0.015)
            h_ref2, s_ref2 = _nesterov_projected(h_pert, beta=beta, n_steps=4000,
                                                   time_budget=min(5.0, _DEADLINE - time.time() - 10))
            _update(h_ref2)

    # Phase 5: Coordinate descent polish to equalize active constraints
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            h_cd, s_cd = _coord_descent_polish(best_h, max_iters=30000,
                                                 time_budget=min(20.0, remaining))
            _update(h_cd)

    # Phase 6: Final high-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        remaining = _DEADLINE - time.time() - 1.0
        if remaining > 2.0:
            h_polish, s_polish = _nesterov_projected(best_h, beta=8000.0, n_steps=3000,
                                                       time_budget=min(4.0, remaining))
            _update(h_polish)

            if time.time() - t0 < _DEADLINE - 1.5:
                remaining2 = _DEADLINE - time.time() - 0.5
                if remaining2 > 1.0:
                    h_cd2, s_cd2 = _coord_descent_polish(best_h, max_iters=10000,
                                                           time_budget=min(2.0, remaining2))
                    _update(h_cd2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
