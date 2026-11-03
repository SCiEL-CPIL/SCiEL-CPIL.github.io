import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _true_score(h):
    """Compute the true objective: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _smooth_score(h, beta=30.0):
    """
    Smooth surrogate for the objective.
    Uses soft-max for the maximum absolute value.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    ac = ac[:size]
    
    abs_ac = np.abs(ac)
    # Soft max approximation
    max_val = np.max(abs_ac)
    if max_val < 1e-12:
        smax = 0.0
    else:
        shifted = abs_ac - max_val
        smax = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        
    return smax / (s * s)

def _peak_suppression(h, n_iters=10, deadline=None):
    """
    Directly target and suppress the peak of the autoconvolution.
    At each iteration, find the argmax of |ac| and apply a corrective perturbation
    to h that reduces the value at that peak location.
    """
    h = h.copy()
    best_val = _true_score(h)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)
        ac = ac[:size]
        abs_ac = np.abs(ac)
        
        peak_idx = np.argmax(abs_ac)
        peak_sign = np.sign(ac[peak_idx])
        
        # The autoconvolution at lag k is sum_i h[i]*h[i+k]
        # To reduce ac[peak_idx], we want to reduce the correlation at that lag.
        # A simple approach: perturb h in a way that anti-correlates at that lag.
        # For lag k, ac[k] = sum_i h[i]*h[i+k]
        # Gradient of ac[k] w.r.t. h[i] is h[i+k] (for i < k) and h[i-k] (for i >= k)
        # We want to decrease |ac[k]|, so move h opposite to the gradient direction
        
        k = peak_idx
        # Compute gradient of ac[k] w.r.t. h
        grad = np.zeros(n)
        if k < n:
            # ac[k] = sum_{i=0}^{n-1-k} h[i] * h[i+k]
            # d(ac[k])/d(h[i]) = h[i+k] for i in [0, n-1-k]
            # d(ac[k])/d(h[i]) = h[i-k] for i in [k, n-1]
            for i in range(n - k):
                grad[i] = h[i + k]
            for i in range(k, n):
                grad[i] = h[i - k]
        else:
            # For k >= n, ac[k] = sum_{i=0}^{n-1-(k-n+1)} h[i+k-n+1] * h[i]
            # This is getting complex; let's use FFT-based gradient instead
            pass
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Move opposite to the sign-weighted gradient to reduce |ac[k]|
        step = 0.02 / grad_norm
        h_new = h - peak_sign * step * grad
        
        # Ensure we don't make things worse
        new_val = _true_score(h_new)
        if new_val < best_val:
            h = h_new
            best_val = new_val
        else:
            # Try smaller step
            for factor in [0.5, 0.25, 0.1]:
                h_try = h - peak_sign * step * factor * grad
                v_try = _true_score(h_try)
                if v_try < best_val:
                    h = h_try
                    best_val = v_try
                    break
    
    return h, best_val

def _local_search_true(h, max_passes=5, deadline=None, step_size=0.05):
    """
    Local search on the true (non-smooth) objective using coordinate descent.
    """
    h = h.copy()
    best_val = _true_score(h)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            best_local = old_val
            best_local_val = best_val
            
            # Try small perturbations
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size, 
                         2*step_size, -2*step_size]:
                h[i] = old_val + delta
                new_val = _true_score(h)
                if new_val < best_local_val:
                    best_local_val = new_val
                    best_local = h[i]
            
            if best_local != old_val:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _spectral_init(n):
    """
    Generate initial candidates using spectral properties.
    The all-ones vector gives ac_k = n for all k, so max|ac|/S^2 = n/n^2 = 1/n.
    We want to minimize this ratio.
    
    For a sequence with sum S, the 'average' autoconvolution is S^2/n (by Parseval).
    So the minimum possible max|ac| is at least S^2/n, giving ratio >= 1/n.
    
    To achieve this, we need |ac_k| = S^2/n for all k, which means |FFT(h)_k| is constant.
    This is achieved by h being a scaled version of a chirp or similar.
    """
    candidates = []
    
    # Chirp: h_i = exp(2*pi*i*i/n) has flat magnitude spectrum
    i = np.arange(n)
    chirp = np.cos(2 * np.pi * i**2 / n)
    candidates.append(chirp)
    
    # Real chirp variant
    chirp2 = np.cos(np.pi * i**2 / n)
    candidates.append(chirp2)
    
    # Alternating with Gaussian
    gauss = np.exp(-0.5 * ((i - n/2) / (n*0.2))**2)
    alt = np.where(i % 2 == 0, 1.0, -1.0)
    candidates.append(gauss * alt)
    
    # Random with zero mean component
    rng = np.random.default_rng(123)
    for seed in [42, 123, 456]:
        rng_s = np.random.default_rng(seed)
        v = rng_s.standard_normal(n)
        v -= np.mean(v)  # zero mean
        candidates.append(v)
    
    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    frac = (i * phi) % 1.0
    candidates.append(2 * frac - 1)
    
    # Triangle window
    mu = n / 2
    triangle = np.maximum(0, 1 - np.abs(i - mu) / mu)
    candidates.append(triangle)
    
    return candidates

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N) / N
    
    # Evaluate baseline
    base_score = _true_score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
    
    # Generate initializations - combine current with spectral inits
    inits = []
    inits.append(np.ones(N) / N)
    
    rng = np.random.default_rng(123)
    for _ in range(5):
        inits.append(rng.standard_normal(N))
    
    # Chirp
    x = np.arange(N)
    inits.append(np.cos(2 * np.pi * x**2 / N))
    
    # Binary-like
    inits.append(np.sign(rng.standard_normal(N)))
    
    # Add spectral inits
    spectral_candidates = _spectral_init(N)
    inits.extend(spectral_candidates)
    
    # Optimization loop with adaptive beta schedule
    beta_schedule = [60.0, 120.0, 250.0]
    
    for h_init in inits:
        if time.time() > deadline - 30:
            break
            
        # Normalize initial guess
        s_init = np.sum(h_init)
        if abs(s_init) > 1e-9:
            h_init = h_init / s_init
        else:
            h_init = np.ones(N) / N
            
        try:
            # Multi-stage optimization with increasing beta
            h_current = h_init.copy()
            for beta in beta_schedule:
                if time.time() > deadline - 20:
                    break
                
                # Create a closure with the current beta
                def obj_func(x, b=beta):
                    return _smooth_score(x, beta=b)
                
                res = minimize(
                    obj_func, 
                    h_current, 
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
                )
                
                h_opt = res.x
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-9:
                    h_current = h_opt / s_opt
                else:
                    break
                    
                # Evaluate true score
                true_val = _true_score(h_current)
                
                if true_val < best_score:
                    best_score = true_val
                    best_h = h_current.copy()
                    
        except Exception:
            pass
        
    # Peak suppression phase on the best result
    if time.time() < deadline - 15:
        h_ps, val_ps = _peak_suppression(best_h, n_iters=20, deadline=deadline-10)
        if val_ps < best_score:
            best_score = val_ps
            best_h = h_ps.copy()
            
    # Local search refinement on the best result found
    if time.time() < deadline - 5:
        h_local, val_local = _local_search_true(best_h, max_passes=5, deadline=deadline-2, step_size=0.05)
        if val_local < best_score:
            best_score = val_local
            best_h = h_local.copy()
            
    # Final check
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
        
    return best_h
