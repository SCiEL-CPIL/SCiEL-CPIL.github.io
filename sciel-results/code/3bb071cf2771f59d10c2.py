"""Erdos minimum overlap: smooth-surrogate optimization with bisection on t.
Combines the epigraph formulation with a bisection-based feasibility check
using projected gradient descent (with Nesterov acceleration) on the smooth
log-sum-exp surrogate. This avoids SLSQP's tendency to get stuck in local
minima by systematically narrowing the feasible t range. Multiple diverse
starts and annealed beta are used.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N


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
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2} via bisection on shift."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET
    s = h.sum()
    lo = min(h.min() - target, 0.0) - 1.0
    hi = max(h.max() - target, 0.0) + 1.0
    # f(tau) = sum(clip(h - tau, 0, 1)) is non-increasing in tau
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = np.sum(np.clip(h - mid, 0.0, 1.0))
        if val > target:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    h = np.clip(h - tau, 0.0, 1.0)
    # Tiny residual fix on the element closest to the boundary we need to adjust
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s  # gradient w.r.t. x
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh). Vectorized via FFT."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    factor = -(2.0 / n)
    # Build the convolution kernel: K[i] = sum_k g_c[k] * (1 if |i-k| in proper window)
    # C_k = (2/n)(n/2 - r_k), so dC_k/dh_j = -(2/n) * (r_k's derivative)
    # r_k = sum_i h[i]h[i-k]; d r_k/d h_j = h[j-k] (if 0<=j-k<n) + h[j+k] (if 0<=j+k<n)
    # Using FFT: gradient = -(2/n) * (g_c * h) circular conv, but we need LINEAR.
    # Build kernel k of length N: k[m] = g_c[m] + g_c[N-m] for m=1..N-1, k[0]=g_c[0]*2
    kern = np.zeros(_NFFT)
    kern[0] = 2.0 * g_c[0]
    for k in range(1, n):
        kern[k] = g_c[k]
        kern[_NFFT - k] = g_c[k]
    H = np.fft.rfft(h, _NFFT)
    K = np.fft.rfft(kern, _NFFT)
    g_h = np.fft.irfft(H * K, _NFFT)[:n]
    g_h *= factor
    return val, g_h


def _projected_gradient_step(h, grad, lr):
    """One projected gradient step: h_new = proj(h - lr * grad)."""
    h_new = h - lr * grad
    return _project(h_new)


def _nesterov_accelerated_gradient(h0, beta, n_steps=2000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate.
    Returns best h found and its score."""
    from scipy.optimize import minimize
    
    t_start = time.time()
    h = _project(h0.copy())
    
    # Define objective and gradient for SLSQP
    def _obj(x):
        val, _ = _smooth_obj_and_grad(x, beta)
        return val
    
    def _grad(x):
        _, g = _smooth_obj_and_grad(x, beta)
        return g
    
    # Constraints: sum(h) = 1
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - _TARGET, 'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Try SLSQP with the smooth surrogate
    # SLSQP can be slow, so we limit maxiter and use the Nesterov result as a good start
    # Actually, let's stick to Nesterov for speed, but add a final SLSQP polish if time permits
    # The current Nesterov is fast. Let's just improve the projection and step size.
    
    # Re-implementing Nesterov with better line search or just fixed step?
    # The previous implementation was okay. Let's try to integrate SLSQP as a final polish step
    # instead of replacing the whole thing, to keep the fast exploration.
    
    # For now, return the Nesterov result, and we will add a _polish_slsqp function
    h = h0.copy()
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]
    
    if lr is None:
        lr = 1.0 / (beta * 2.0)  # Lipschitz constant heuristic
    
    momentum = 0.9
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        # Nesterov lookahead
        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        
        # Gradient step
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        # Update momentum
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        y = h_new + (momentum / t_k) * (h_new - h)
        y = _project(y)
        h = h_new
        
        val_h = _smooth_obj_and_grad(h, beta)[0]
        if val_h < best_val:
            best_val = val_h
            best_h = h.copy()
        
        # Adaptive learning rate: reduce if objective increases
        if step > 10 and val_h > best_val + 1e-10:
            lr *= 0.95
    
    return best_h, _score(best_h)


def _bisection_check(t_target, h0, beta, n_steps=1500, time_budget=8.0):
    """Check if there exists h with max(C(h)) <= t_target using smooth surrogate.
    Minimize S_beta(C(h) - t_target) and check if result is close to 0."""
    t_start = time.time()
    h = h0.copy()
    best_h = h.copy()
    best_obj = float('inf')
    
    def _obj_grad_t(h):
        cs = _overlap_values(h)
        shifted = cs - t_target
        val, g_c = _smooth_max_and_grad(shifted, beta)
        n = N
        factor = -(2.0 / n)
        kern = np.zeros(_NFFT)
        kern[0] = 2.0 * g_c[0]
        for k in range(1, n):
            kern[k] = g_c[k]
            kern[_NFFT - k] = g_c[k]
        H = np.fft.rfft(h, _NFFT)
        K = np.fft.rfft(kern, _NFFT)
        g_h = np.fft.irfft(H * K, _NFFT)[:n]
        g_h *= factor
        return val, g_h
    
    # Nesterov accelerated projected gradient
    lr = 1.0 / (beta * 2.0)
    momentum = 0.9
    y = h.copy()
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break
        
        val_y, grad_y = _obj_grad_t(y)
        h_new = y - lr * grad_y
        h_new = _project(h_new)
        
        t_k = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        y = h_new + (momentum / t_k) * (h_new - h)
        y = _project(y)
        h = h_new
        
        val_h, _ = _obj_grad_t(h)
        if val_h < best_obj:
            best_obj = val_h
            best_h = h.copy()
        
        if step > 10 and val_h > best_obj + 1e-10:
            lr *= 0.95
    
    # Check if we found a feasible solution
    cs = _overlap_values(best_h)
    max_c = np.max(cs)
    feasible = max_c <= t_target + 1e-8
    return feasible, best_h, max_c


def _coordinate_descent_polish(h0, time_budget=15.0):
    """Coordinate descent polish: directly minimize max(C(h)) by adjusting one coordinate at a time.
    For each coordinate j, find the best h[j] in [0,1] (keeping others fixed, adjusting sum) that
    minimizes max_k C_k. Uses ternary/bisection search per coordinate."""
    t_start = time.time()
    h = _project(h0.copy())
    best_score = _score(h)
    
    def _score_with_j(h, j, val_j):
        """Score if h[j] is set to val_j, with sum constraint handled by distributing residual."""
        h_try = h.copy()
        h_try[j] = val_j
        h_try = _project(h_try)
        return _score(h_try)
    
    n_passes = 0
    while time.time() - t_start < time_budget:
        improved = False
        for j in range(N):
            if time.time() - t_start > time_budget:
                break
            # Ternary search for best h[j] in [0,1]
            lo, hi = 0.0, 1.0
            for _ in range(12):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                s1 = _score_with_j(h, j, m1)
                s2 = _score_with_j(h, j, m2)
                if s1 < s2:
                    hi = m2
                else:
                    lo = m1
            best_j = 0.5 * (lo + hi)
            s_new = _score_with_j(h, j, best_j)
            if s_new < best_score - 1e-12:
                h = _project(h.copy())
                h[j] = best_j
                h = _project(h)
                best_score = _score(h)
                improved = True
        n_passes += 1
        if not improved and n_passes > 1:
            break
    
    return h, best_score


def _refine_smooth(h0, beta, n_max_iter=3000, time_budget=25.0):
    """Refine h using Nesterov accelerated projected gradient on the smooth surrogate."""
    h0 = _project(h0.copy())
    h_opt, s_opt = _nesterov_accelerated_gradient(h0, beta, n_steps=n_max_iter, time_budget=time_budget)
    return h_opt, s_opt


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


def _bisection_optimize(h0, beta, t_lo, t_hi, n_bisect=15, time_budget=30.0):
    """Bisection on t to find the minimum feasible max overlap.
    Returns best h and its score."""
    t_start = time.time()
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Ensure t_hi is feasible
    for _ in range(5):
        if time.time() - t_start > time_budget:
            break
        feasible, h_try, max_c = _bisection_check(t_hi, h0, beta, n_steps=800, 
                                                   time_budget=min(5.0, time_budget - (time.time() - t_start)))
        if feasible:
            break
        t_hi += 0.05
        if t_hi > 1.0:
            t_hi = 1.0
    
    # Bisection
    for bisect_iter in range(n_bisect):
        if time.time() - t_start > time_budget:
            break
        
        t_mid = (t_lo + t_hi) / 2.0
        remaining = time_budget - (time.time() - t_start)
        if remaining < 3.0:
            break
        
        feasible, h_try, max_c = _bisection_check(t_mid, h0, beta, 
                                                   n_steps=1200, time_budget=min(8.0, remaining - 1.0))
        
        if feasible:
            t_hi = t_mid
            if max_c < best_score:
                best_score = max_c
                best_h = h_try.copy()
        else:
            t_lo = t_mid
        
        # Update h0 to the best found so far for next iteration
        h0 = best_h.copy()
    
    return best_h, best_score


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
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with Nesterov accelerated gradient
    # Use annealed beta for sharper max approximation
    beta_list = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break
        
        time_per_beta = min(20.0, (remaining - 10.0) / 2.0)
        if time_per_beta < 5.0:
            break
        
        h_ref, s_ref = _refine_smooth(best_h, beta=beta, n_max_iter=3000, time_budget=time_per_beta)
        _update(h_ref)
        
        # Add perturbation of best for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 30:
            h_pert = _perturb(best_h, rng, scale=0.02)
            h_ref2, s_ref2 = _refine_smooth(h_pert, beta=beta, n_max_iter=2000, time_budget=min(10.0, _DEADLINE - time.time() - 10))
            _update(h_ref2)

    # Phase 5: Bisection on t with increasing beta for high-precision optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 10.0:
            # Start bisection from current best score
            t_lo = max(0.0, best_score - 0.05)
            t_hi = best_score + 0.02
            
            # Try bisection with increasing beta
            for beta_bisect in [50.0, 100.0, 200.0]:
                if time.time() - t0 > _DEADLINE - 10:
                    break
                remaining_b = _DEADLINE - time.time() - 5.0
                if remaining_b < 5.0:
                    break
                
                h_bis, s_bis = _bisection_optimize(best_h, beta=beta_bisect, 
                                                    t_lo=t_lo, t_hi=t_hi,
                                                    n_bisect=12, time_budget=min(15.0, remaining_b))
                _update(h_bis)
                
                # Update bounds
                if s_bis < best_score:
                    t_hi = s_bis
                    t_lo = max(0.0, s_bis - 0.03)
        
        # Phase 6: Final high-precision polish
        if best_h is not None and time.time() - t0 < _DEADLINE - 8:
            remaining = _DEADLINE - time.time() - 3.0
            if remaining > 3.0:
                h_polish, s_polish = _refine_smooth(best_h, beta=800.0, n_max_iter=5000,
                                                    time_budget=min(10.0, remaining))
                _update(h_polish)
                
                # Second polish with even higher beta if time allows
                if time.time() - t0 < _DEADLINE - 5:
                    remaining2 = _DEADLINE - time.time() - 2.0
                    if remaining2 > 2.0:
                        h_polish2, s_polish2 = _refine_smooth(best_h, beta=1500.0, n_max_iter=2000,
                                                              time_budget=min(5.0, remaining2))
                        _update(h_polish2)

    # Phase 7: Coordinate descent polish (directly minimizes max overlap)
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            h_cd, s_cd = _coordinate_descent_polish(best_h, time_budget=min(20.0, remaining))
            _update(h_cd)
    
    # Phase 8: Local perturbation + smooth refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        for scale in [0.02, 0.01, 0.005]:
            if time.time() - t0 > _DEADLINE - 10:
                break
            for _ in range(3):
                if time.time() - t0 > _DEADLINE - 10:
                    break
                h_pert = _perturb(best_h, rng, scale=scale)
                _update(h_pert)
            if best_h is not None:
                remaining = _DEADLINE - time.time() - 8.0
                if remaining > 5.0:
                    h_ref, s_ref = _refine_smooth(best_h, beta=400.0, n_max_iter=1000,
                                                  time_budget=min(8.0, remaining))
                    _update(h_ref)
                    # Coordinate polish after refinement
                    if time.time() - t0 < _DEADLINE - 8:
                        remaining_cd = _DEADLINE - time.time() - 5.0
                        if remaining_cd > 3.0:
                            h_cd2, s_cd2 = _coordinate_descent_polish(best_h, time_budget=min(8.0, remaining_cd))
                            _update(h_cd2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
