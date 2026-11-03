import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = _fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_gaussian(n):
    """Generate a Gaussian window."""
    x = np.arange(n)
    mu = n / 2
    sigma = n / 4
    return np.exp(-(x - mu)**2 / (2*sigma**2))

def _gen_cosine(n):
    """Generate a cosine window."""
    x = np.arange(n)
    return np.cos(np.pi * x / n)

def _grad_ac_at_lag(h, k):
    """
    Compute gradient of ac[k] w.r.t h vectorized.
    ac[k] = sum_i h[i]*h[i+k] for k in [0, n-1]
    ac[k] = sum_i h[i+k-n+1]*h[i] for k in [n, 2n-2]
    """
    n = len(h)
    g = np.zeros(n)
    if k < n:
        # ac[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k]
        # d/dh[i] = h[i+k], d/dh[i+k] = h[i]
        m = n - k
        g[:m] += h[k:k+m]
        g[k:k+m] += h[:m]
    else:
        # k >= n, lag = k-(n-1) is negative
        lag = k - (n - 1)  # negative
        abs_lag = -lag
        # ac[k] = sum_{i=abs_lag}^{n-1} h[i]*h[i-abs_lag]
        # d/dh[i] = h[i-abs_lag], d/dh[i-abs_lag] = h[i]
        g[abs_lag:] += h[:-abs_lag]
        g[:-abs_lag] += h[abs_lag:]
    return g

def _local_search_flip(h, deadline, max_passes=200):
    """
    Local search: try flipping individual elements to reduce max|ac|.
    Uses incremental autoconvolution update for O(N) per flip.
    """
    best_h = h.copy()
    best_val = _score(best_h)
    n = len(h)
    ac = _autoconv_fft(best_h)
    abs_ac = np.abs(ac)
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        # Find top peaks to target
        top_k = 5
        peak_indices = np.argsort(abs_ac)[-top_k:]
        
        improved = False
        for k in peak_indices:
            if time.time() > deadline:
                break
            
            # Try flipping each element that contributes to ac[k]
            # For efficiency, only try a subset
            g = _grad_ac_at_lag(best_h, k)
            # Elements with largest |g[j]| are most impactful
            top_j = np.argsort(np.abs(g))[-10:]
            
            for j in top_j:
                # Compute new ac[k] after flipping h[j]
                # ac[k] changes by: -2*h[j]*g[j] (since we flip h[j] to -h[j])
                # Actually: new_ac[k] = ac[k] - 2*h[j]*g[j]
                # But this is approximate; let's do exact update
                h_old = best_h[j]
                best_h[j] = -h_old
                
                # Update ac incrementally
                # ac changes by: -2*h_old * (shifted h)
                # For lag k: delta_ac[i] for i near k
                # Simpler: just recompute ac for this one flip (O(N) with vectorized)
                ac_new = ac.copy()
                # The change in autoconvolution from flipping h[j]:
                # ac_new[m] = ac[m] - 2*h_old*(h[m-j] + h[j+m] - 2*h_old*h[m-j]*h[j+m]...) 
                # This gets complicated. Let's just recompute if N is small enough.
                # For N=400, recomputing ac is ~10us, so 200*5*10 = 10000 recomputes = 100ms. OK.
                ac_new = _autoconv_fft(best_h)
                val_new = np.max(np.abs(ac_new)) / (np.sum(best_h)**2)
                
                if val_new < best_val - 1e-12:
                    best_val = val_new
                    ac = ac_new
                    abs_ac = np.abs(ac)
                    improved = True
                else:
                    best_h[j] = h_old  # revert
        
        if not improved:
            break
    
    return best_h, best_val

def _peak_targeting_polish(h, deadline, max_iters=50):
    """
    Polish by targeting the specific autoconvolution peaks.
    Moves h in the direction that reduces the largest |ac[k]|.
    """
    best_h = h.copy()
    best_val = _score(best_h)
    n = len(h)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        ac = _autoconv_fft(best_h)
        abs_ac = np.abs(ac)
        max_idx = np.argmax(abs_ac)
        max_val = abs_ac[max_idx]
        
        if max_val < 1e-10:
            break
            
        # Vectorized gradient computation
        g = _grad_ac_at_lag(best_h, max_idx)
        
        # We want to reduce |ac[k]|.
        sign = np.sign(ac[max_idx])
        direction = -sign * g
        
        # Normalize direction
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            break
        direction = direction / norm_dir
        
        # Line search for alpha
        best_alpha = 0.0
        best_local_val = best_val
        
        for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            h_try = best_h + alpha * direction
            val_try = _score(h_try)
            if val_try < best_local_val:
                best_local_val = val_try
                best_alpha = alpha
                
        if best_alpha == 0.0:
            for alpha in [0.001, 0.005, 0.02]:
                h_try = best_h + alpha * direction
                val_try = _score(h_try)
                if val_try < best_local_val:
                    best_local_val = val_try
                    best_alpha = alpha
                    
        if best_alpha > 0:
            best_h = best_h + best_alpha * direction
            best_val = best_local_val
        else:
            break
            
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    
    # 1. Gaussian
    inits.append(_gen_gaussian(N))
    
    # 2. Cosine
    inits.append(_gen_cosine(N))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Random smoothed
    rng = np.random.default_rng(42)
    for i in range(4):
        noise = rng.standard_normal(N)
        kernel_size = 5 + i * 2
        kernel = np.ones(kernel_size)/kernel_size
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # 5. Random raw
    inits.append(rng.standard_normal(N))

    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            # Stage 1: Smooth optimization with moderate beta
            beta = 50.0
            def obj(x):
                return _objective_smooth(x, beta=beta)
            
            res = minimize(obj, h_norm, method='L-BFGS-B', 
                           options={'maxiter': 200, 'disp': False, 'gtol': 1e-6})
            h_opt = res.x
            
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
        except Exception:
            continue

    # Final Polish 1: High Beta Smooth
    if time.time() < deadline - 5:
        try:
            beta_high = 200.0
            def obj_high(x):
                return _objective_smooth(x, beta=beta_high)
            
            res = minimize(obj_high, best_h, method='L-BFGS-B', 
                           options={'maxiter': 100, 'disp': False, 'gtol': 1e-8})
            h_opt = res.x
            s_opt = np.sum(h_opt)
            if abs(s_opt) > 1e-6:
                h_opt = h_opt / s_opt
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Final Polish 2: Peak Targeting
    if time.time() < deadline - 5:
        best_h, best_score = _peak_targeting_polish(best_h, deadline - 5, max_iters=50)
    
    # Final Polish 3: Local Search with Flip
    if time.time() < deadline - 1:
        best_h, best_score = _local_search_flip(best_h, deadline - 0.5, max_passes=100)

    return best_h
