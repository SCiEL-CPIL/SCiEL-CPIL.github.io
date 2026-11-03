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
            
        # Gradient of ac[max_idx] w.r.t h
        # ac[k] = sum_{i} h[i] h[i+k] (linear conv, zero padded)
        # d(ac[k])/d(h[j]) = h[j-k] + h[j+k] (with bounds)
        # We can compute this efficiently by shifting h.
        
        # Let's use the property that ac = ifft(fft(h)^2)
        # Gradient of ac[k] is circular convolution of h with delta(k) ? No, linear.
        # Simpler: Just compute the gradient vector g where g[j] = d(ac[max_idx])/d(h[j])
        
        # ac[max_idx] = sum_i h[i] h[i+max_idx] (where valid)
        # If max_idx >= 0:
        #   terms are h[i] * h[i+max_idx] for i=0..N-1-max_idx
        #   grad[i] += h[i+max_idx]
        #   grad[i+max_idx] += h[i]
        # If max_idx < 0 (not possible in linear ac index 0..2N-2, but let's assume standard)
        # In our ac array, index k corresponds to lag k.
        # ac[k] = sum_{i=0}^{N-1-k} h[i] h[i+k] for k >= 0
        # ac[k] = sum_{i=0}^{N-1+(-k)} h[i-k] h[i] for k < 0 (if we extended, but we have 0..2N-2)
        
        # Our ac is length 2N-1. Index 0 is lag -N+1? No.
        # np.convolve(h, h, 'full') has index 0 corresponding to h[0]*h[-N+1] (0) ... 
        # Actually, standard linear conv:
        # ac[k] for k=0..N-1 is sum h[i]h[i+k]
        # ac[k] for k=N..2N-2 is sum h[i]h[i-(k-N+1)]?
        
        # Let's just use FFT gradient if we can, or simple loop for N=400 (fast enough for polish).
        # N=400, loop is 400 ops. 50 iters is 20k ops. Very fast.
        
        g = np.zeros(n)
        k = max_idx
        if k < n:
            # ac[k] = sum_{i=0}^{n-1-k} h[i] h[i+k]
            # d/dh[i] = h[i+k]
            # d/dh[i+k] = h[i]
            for i in range(n - k):
                g[i] += h[i+k]
                g[i+k] += h[i]
        else:
            # k >= n
            # Let l = k - (n - 1)  # effective negative lag magnitude?
            # In full conv of length 2n-1:
            # Index k corresponds to correlation with lag k - (n-1).
            # If k > n-1, lag is negative.
            # Let lag = k - (n - 1). lag is negative.
            # ac[k] = sum_{i} h[i] h[i + lag] where i+lag >= 0.
            # This is equivalent to sum_{j} h[j] h[j - lag]
            lag = k - (n - 1) # negative
            abs_lag = -lag
            for i in range(abs_lag, n):
                # h[i] * h[i - abs_lag]
                g[i] += h[i - abs_lag]
                g[i - abs_lag] += h[i]
                
        # We want to reduce |ac[k]|.
        # If ac[k] > 0, we want to decrease it -> move in -g direction.
        # If ac[k] < 0, we want to increase it (make it less negative) -> move in +g direction.
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
        
        # Try a few alphas
        # Scale of h is ~1/N or 1. Let's estimate step size.
        # Sum(h) is normalized? No, we don't normalize in this function.
        # But _score divides by (sum h)^2.
        # If we change h, sum changes.
        
        # Heuristic step sizes
        for alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            h_try = best_h + alpha * direction
            val_try = _score(h_try)
            if val_try < best_local_val:
                best_local_val = val_try
                best_alpha = alpha
                
        if best_alpha == 0.0:
            # Try smaller steps
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
            # If no improvement, try opposite direction? Or break?
            # The subgradient direction should decrease the specific peak, but might increase others.
            # If line search fails, the peak is likely "stuck" or we are at a local min of the max.
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
    if time.time() < deadline - 1:
        best_h, best_score = _peak_targeting_polish(best_h, deadline - 0.5, max_iters=100)

    return best_h
