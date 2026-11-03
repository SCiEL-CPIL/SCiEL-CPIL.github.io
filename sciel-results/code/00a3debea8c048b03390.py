"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses epigraph formulation with SLSQP, refined with higher precision polishes.
Combines cosine initialization with random restarts and adaptive beta tuning."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill


def _autocorr(h):
    """
    Compute linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1.
    Uses FFT for O(N log N) computation.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """
    Compute overlap values C_k = (2/N) * (N/2 - r_k) for k=0..N-1.
    """
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """
    Smooth maximum of overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _coordinate_descent_polish(h, beta, max_iter=500, tol=1e-12):
    """
    Coordinate descent polish: iterate over coordinates, taking small steps
    along the negative gradient while maintaining sum constraint and bounds.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    n = len(h)
    target_sum = n / 2.0
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 2:
            break
        
        val, grad = _smooth_max_grad(h, beta)
        
        # Find the coordinate with largest |gradient| component
        # that can be adjusted while maintaining sum constraint
        # We do a simple line search on a pair of coordinates
        
        # Pick the two coordinates with largest opposite gradient signs
        # and largest magnitude
        abs_grad = np.abs(grad)
        
        # Try adjusting the most "improving" coordinate
        # If grad[j] > 0, decreasing h[j] helps; if grad[j] < 0, increasing helps
        # But we need to maintain sum, so adjust a pair
        
        # Simple approach: find j with max |grad[j]|, adjust it and compensate
        # on the coordinate with smallest |grad| (or opposite sign)
        j = np.argmax(abs_grad)
        
        # Find compensation coordinate: prefer opposite sign gradient, then small |grad|
        # Exclude j
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        
        # Score: prefer opposite sign, then smaller |grad|
        if grad[j] > 0:
            # We want to decrease h[j], so increase h[k]
            # Prefer k with grad[k] < 0 (helps objective)
            candidates = np.where(mask & (grad < 0))[0]
            if len(candidates) == 0:
                candidates = np.where(mask)[0]
            k = candidates[np.argmin(abs_grad[candidates])]
            direction_j = -1.0  # decrease h[j]
            direction_k = 1.0   # increase h[k]
        else:
            # We want to increase h[j], so decrease h[k]
            candidates = np.where(mask & (grad > 0))[0]
            if len(candidates) == 0:
                candidates = np.where(mask)[0]
            k = candidates[np.argmin(abs_grad[candidates])]
            direction_j = 1.0   # increase h[j]
            direction_k = -1.0  # decrease h[k]
        
        # Line search: try step sizes
        best_step = 0.0
        best_val = val
        
        for step in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
            h_trial = h.copy()
            h_trial[j] += direction_j * step
            h_trial[k] += direction_k * step
            
            # Check bounds
            if h_trial[j] < 0 or h_trial[j] > 1 or h_trial[k] < 0 or h_trial[k] > 1:
                continue
            
            v_trial, _ = _smooth_max_grad(h_trial, beta)
            if v_trial < best_val - 1e-15:
                best_val = v_trial
                best_step = step
        
        if best_step == 0.0:
            # Try smaller steps
            for step in [5e-7, 1e-7, 5e-8]:
                h_trial = h.copy()
                h_trial[j] += direction_j * step
                h_trial[k] += direction_k * step
                
                if h_trial[j] < 0 or h_trial[j] > 1 or h_trial[k] < 0 or h_trial[k] > 1:
                    continue
                
                v_trial, _ = _smooth_max_grad(h_trial, beta)
                if v_trial < best_val - 1e-15:
                    best_val = v_trial
                    best_step = step
        
        if best_step > 0:
            h[j] += direction_j * best_step
            h[k] += direction_k * best_step
        else:
            # No improvement found; check convergence
            if abs_grad.max() < tol:
                break
    
    h = _fix_sum(h)
    return h


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to smooth_max(h) <= t,
    with x = [h_0, ..., h_{N-1}, t].
    """
    h = x[:N]
    t = x[N]
    val, _ = _smooth_max_grad(h, beta)
    return t + 0.01 * max(0, val - t)  # Penalize constraint violation


def _epigraph_grad(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    val, grad_h = _smooth_max_grad(h, beta)
    
    grad = np.zeros(len(x))
    if val > t:
        grad[:N] = grad_h * 0.01
        grad[N] = 1.0 + 0.01
    else:
        grad[:N] = grad_h * 0.0
        grad[N] = 1.0
    
    return grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # Second pass if clipping changed sum
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _run_slsqp(h0, beta, maxiter, ftol, gtol):
    """
    Run SLSQP optimization on the smooth max objective.
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val
        
    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad
        
    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _polish(h, beta, maxiter, ftol, gtol):
    """
    High-precision polish pass with tighter tolerances.
    """
    h0 = h.copy()
    h_new = _run_slsqp(h0, beta, maxiter, ftol, gtol)
    h_new = _fix_sum(h_new)
    
    # Verify improvement
    if _score(h_new) < _score(h0):
        return h_new
    return h0


def _spectral_h_from_coeffs(coeffs):
    """
    Reconstruct h from real DFT coefficients (modes 1..K).
    coeffs = [a1, b1, a2, b2, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2*pi*k*i/N) + b_k sin(2*pi*k*i/N))
    """
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2*(k-1)]
        b = coeffs[2*(k-1)+1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """
    Extract the first K real DFT coefficients from h.
    Returns [a1, b1, ..., aK, bK] where h ~ 0.5 + sum (a_k cos + b_k sin)
    """
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter):
    """
    Optimize the spectral coefficients of h to minimize the smooth max objective.
    """
    coeffs0 = _h_to_spectral_coeffs(h_init, K)
    
    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        
        # Penalty for out-of-bounds to keep solution valid
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2)
        
        val, _ = _smooth_max_grad(h, beta)
        return val + 1.0 * penalty

    try:
        res = minimize(obj, coeffs0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass
    return h_init


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Multiple diverse initializations
    i = np.arange(N)
    
    # Cosine start
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Square wave start
    h_sq = np.where(np.arange(N) % 2 == 0, 1.0, 0.0)
    h_sq = _fix_sum(h_sq)
    _update_best(h_sq)
    
    # High-frequency cosine
    h_hf = 0.5 + 0.5 * np.cos(2.0 * np.pi * 4.0 * i / N)
    h_hf = _fix_sum(h_hf)
    _update_best(h_hf)
    
    # 2. Spectral Optimization from best start
    for K in [2, 4, 8, 16]:
        if time.time() - t0 > _DEADLINE - 45:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=200)
        _update_best(h_spec)
        
    # 3. Beta-annealing SLSQP optimization
    if time.time() - t0 < _DEADLINE - 35:
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - t0 > _DEADLINE - 30:
                break
            h_opt = _run_slsqp(best_h, beta=beta, maxiter=300, ftol=1e-10, gtol=1e-8)
            _update_best(h_opt)
        
    # 4. Random restarts with Spectral Polish
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 18:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_spec_rand = _optimize_spectral(h_rand, K=8, beta=50.0, maxiter=100)
            _update_best(h_spec_rand)
            # Quick SLSQP polish
            if time.time() - t0 < _DEADLINE - 15:
                h_pol = _run_slsqp(h_spec_rand, beta=100.0, maxiter=200, ftol=1e-10, gtol=1e-8)
                _update_best(h_pol)
            
    # 5. Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_polished = _polish(best_h, beta=100.0, maxiter=500, ftol=1e-12, gtol=1e-10)
        _update_best(h_polished)
        
    # 6. Final ultra-precise polish with coordinate descent
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_final = _polish(best_h, beta=200.0, maxiter=1000, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
        
        # Coordinate descent polish at very high beta
        if time.time() - t0 < _DEADLINE - 2:
            h_cd = _coordinate_descent_polish(best_h, beta=500.0, max_iter=200)
            _update_best(h_cd)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
