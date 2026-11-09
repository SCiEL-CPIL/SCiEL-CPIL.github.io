import numpy as np
import time
from scipy.optimize import minimize

N = 400
N_FFT = 2 * N - 1

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _soft_max_abs_ac(f, beta=50.0):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns (value, gradient) using vectorized computation.
    """
    F = np.fft.rfft(f, n=N_FFT)
    ac = np.fft.irfft(F * F, n=N_FFT)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros_like(f)
    
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted)) / beta + max_ac
    
    weights = exp_shifted / np.sum(exp_shifted)
    signs = np.sign(ac)
    
    g = weights * signs
    grad = np.array([2.0 * np.dot(g[i:i+N], f) for i in range(N)])
    
    return log_sum, grad

def _objective_and_grad(f):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if s == 0 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _score(f):
    """Compute the actual C3' score."""
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _greedy_peak_shave(f, max_iters=2000, deadline=None):
    """
    Greedy peak shaving: iteratively flip the sign of the element most
    correlated with the current maximum autoconvolution peak.
    """
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(best)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = ac[peak_idx]
        
        # For peak at lag k, the contribution to ac[k] from element i is f[i]*f[k-i]
        # Flipping f[i] changes ac[k] by -2*f[i]*f[k-i] (if k-i is valid)
        # We want to flip the element that most reduces |ac[k]|
        # delta = -2 * f[i] * f[k-i] for valid k-i
        # We want the flip that makes ac[k] closest to 0
        
        # For each valid i (0 <= i < N, 0 <= k-i < N):
        # new_ac_k = ac[k] - 2*f[i]*f[k-i]
        # We want |new_ac_k| minimized, i.e., f[i]*f[k-i] has same sign as ac[k]
        
        valid_i = []
        if peak_idx < N:
            # k - i >= 0 => i <= k, and i >= 0, and i < N
            i_start = max(0, peak_idx - (N - 1))
            i_end = min(N - 1, peak_idx)
            for i in range(i_start, i_end + 1):
                valid_i.append(i)
        else:
            # k >= N, k - i < N => i > k - N, and i >= 0, i < N
            i_start = max(0, peak_idx - N + 1)
            i_end = min(N - 1, peak_idx)  # k-i >= 0 => i <= k, always true for i < N when k >= N
            for i in range(i_start, i_end + 1):
                valid_i.append(i)
        
        if not valid_i:
            break
        
        valid_i = np.array(valid_i)
        k_vals = peak_idx - valid_i
        
        # f[i] * f[k-i] for each valid i
        products = best[valid_i] * best[k_vals]
        
        # We want to flip i such that products[i] has the same sign as peak_val
        # (this reduces |ac[peak_idx]|)
        # Among those, pick the one with largest |product| (biggest reduction)
        same_sign_mask = np.sign(products) == np.sign(peak_val)
        
        if not np.any(same_sign_mask):
            # No flip helps at this peak; try a small perturbation instead
            candidate = best.copy()
            idx = np.random.randint(N)
            candidate[idx] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
            continue
        
        candidates_idx = valid_i[same_sign_mask]
        candidates_prod = products[same_sign_mask]
        
        # Pick the one with largest absolute product
        best_local = np.argmax(np.abs(candidates_prod))
        flip_idx = candidates_idx[best_local]
        
        candidate = best.copy()
        candidate[flip_idx] *= -1
        sc = _score(candidate)
        
        if sc < best_score:
            best = candidate
            best_score = sc
        else:
            # Try second best
            if len(candidates_idx) > 1:
                order = np.argsort(np.abs(candidates_prod))[::-1]
                for trial in range(min(5, len(order))):
                    fi = candidates_idx[order[trial]]
                    candidate = best.copy()
                    candidate[fi] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
                        break
    
    return best, best_score

def _spectral_barker_init(seed=0):
    """
    Barker-like initialization: inverse FFT of phase-only mask with flat magnitude.
    Then apply greedy peak shaving.
    """
    rng = np.random.RandomState(seed)
    n_freqs = N_FFT // 2 + 1
    
    # Phase-only mask
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    mags = np.ones(n_freqs)
    mags[0] = 0  # zero DC
    
    F = mags * np.exp(1j * phases)
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    
    # Normalize so sum is positive
    s = np.sum(f)
    if s < 0:
        f = -f
    elif abs(s) < 1e-12:
        f = rng.randn(N)
    
    # Apply greedy peak shaving
    f, _ = _greedy_peak_shave(f, max_iters=100)
    
    return f

def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips
        n_flips = np.random.randint(1, 5)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        sc = _score(candidate)
        if sc < best_score:
            best = candidate
            best_score = sc
        
        # Try single flip of the most correlated element with peak
        if it % 5 == 0:
            ac = _autoconv(best)
            abs_ac = np.abs(ac)
            peak_idx = np.argmax(abs_ac)
            peak_val = ac[peak_idx]
            
            if peak_idx < N:
                i_start = max(0, peak_idx - (N - 1))
                i_end = min(N - 1, peak_idx)
            else:
                i_start = max(0, peak_idx - N + 1)
                i_end = min(N - 1, peak_idx)
            
            if i_start <= i_end:
                valid_i = np.arange(i_start, i_end + 1)
                k_vals = peak_idx - valid_i
                products = best[valid_i] * best[k_vals]
                same_sign_mask = np.sign(products) == np.sign(peak_val)
                
                if np.any(same_sign_mask):
                    candidates_idx = valid_i[same_sign_mask]
                    candidates_prod = products[same_sign_mask]
                    best_local = np.argmax(np.abs(candidates_prod))
                    flip_idx = candidates_idx[best_local]
                    
                    candidate = best.copy()
                    candidate[flip_idx] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (5s buffer)
    
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Phase 1: Spectral Barker initializations with greedy shaving
    n_spectral = 8
    for r in range(n_spectral):
        if time.time() > deadline - 30.0:
            break
        f0 = _spectral_barker_init(seed=r * 137 + 1)
        sc = _score(f0)
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Phase 2: Gradient optimization from best and other starts
    starts = []
    if best_f is not None:
        starts.append(best_f.copy())
    
    # Add diverse starts
    starts.append(np.ones(N))
    starts.append(np.random.randn(N))
    starts.append(np.random.choice([-1.0, 1.0], size=N))
    s = np.sum(starts[2])
    if s < 0:
        starts[2] = -starts[2]
    
    t = np.linspace(0, 10, N)
    starts.append(np.cos(t**2))
    
    starts.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    for f0 in starts:
        if time.time() > deadline - 20.0:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                f0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            
            candidate = res.x
            sc = _score(candidate)
            if sc < best_score:
                best_f = candidate.copy()
                best_score = sc
                
        except Exception:
            pass
    
    # Phase 3: Greedy peak shaving polish
    if best_f is not None and time.time() < deadline - 10.0:
        polished, polished_score = _greedy_peak_shave(best_f, max_iters=5000, deadline=deadline - 10.0)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Phase 4: Local search polish
    if best_f is not None and time.time() < deadline - 3.0:
        polished, polished_score = _local_search(best_f, deadline - 3.0, max_iters=30000)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Final: try a few more spectral inits with remaining time
    if time.time() < deadline - 5.0:
        for r in range(5):
            if time.time() > deadline - 5.0:
                break
            f0 = _spectral_barker_init(seed=1000 + r * 73)
            sc = _score(f0)
            if sc < best_score:
                best_f = f0.copy()
                best_score = sc
                # Quick polish
                if time.time() < deadline - 3.0:
                    polished, polished_score = _greedy_peak_shave(best_f, max_iters=1000, deadline=deadline - 3.0)
                    if polished_score < best_score:
                        best_f = polished
                        best_score = polished_score
    
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
