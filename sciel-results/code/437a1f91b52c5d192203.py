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

def _walsh_hadamard_basis(n, num_basis):
    """
    Generate orthogonal Walsh-Hadamard basis functions of length n.
    Returns an (num_basis, n) matrix where each row is a normalized basis vector.
    """
    # Find smallest power of 2 >= n
    m = 1
    while m < n:
        m *= 2
    
    # Generate full Hadamard matrix of size m x m
    H = np.ones((1, 1))
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    
    # Take first num_basis rows, truncate to length n
    basis = H[:num_basis, :n].astype(float)
    
    # Normalize each row to unit norm
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    basis = basis / norms
    
    return basis

def _walsh_hadamard_optimize(basis, deadline, max_iters=200):
    """
    Optimize coefficients of Walsh-Hadamard basis functions to minimize
    the smooth objective. The signal is h = sum_j c_j * basis[j].
    """
    n = basis.shape[1]
    num_basis = basis.shape[0]
    
    # Initial coefficients: small random values
    rng = np.random.default_rng(123)
    c = rng.standard_normal(num_basis) * 0.1
    
    def obj_wh(c):
        h = basis @ c
        return _objective_smooth(h, beta=50.0)
    
    def grad_wh(c):
        h = basis @ c
        s = np.sum(h)
        if abs(s) < 1e-9:
            return np.zeros(num_basis)
        ac = _autoconv_fft(h)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        if m < 1e-15:
            return np.zeros(num_basis)
        
        # Gradient of soft_max_abs w.r.t. ac
        beta = 50.0
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        weights = np.exp(scaled) / np.sum(np.exp(scaled))
        grad_ac = weights * np.sign(ac)
        
        # Gradient of ac w.r.t. h: ac[k] = sum_i h[i]h[i+k] (for k>=0)
        # This is the autocorrelation, so grad_h[j] = sum_k grad_ac[k] * (h[j-k] + h[j+k])
        # More efficiently: grad_h = ifft(fft(grad_ac_padded) * fft(h_padded))
        # But we need linear convolution, not circular.
        
        # For N=400 and num_basis small, we can compute gradient numerically or analytically.
        # Let's use a simple approach: grad of objective w.r.t. c_j
        # d/dc_j [soft_max_abs(ac) / s^2]
        # = (1/s^2) * d(soft_max_abs)/dc_j - (2*s_j/s^3) * soft_max_abs
        # where s_j = basis[j].sum()
        
        # d(soft_max_abs)/dc_j = sum_k grad_ac[k] * d(ac[k])/dc_j
        # d(ac[k])/dc_j = d/dh_j [sum_i h[i]h[i+k]] summed over i, then projected to basis[j]
        # This gets complex. Let's use finite differences for the gradient.
        
        eps = 1e-6
        grad = np.zeros(num_basis)
        for j in range(num_basis):
            c_plus = c.copy()
            c_plus[j] += eps
            c_minus = c.copy()
            c_minus[j] -= eps
            f_plus = obj_wh(c_plus)
            f_minus = obj_wh(c_minus)
            grad[j] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    # Use L-BFGS-B with analytic-like gradient (finite differences)
    best_c = c.copy()
    best_val = obj_wh(c)
    
    try:
        res = minimize(obj_wh, c, method='L-BFGS-B', 
                       jac=grad_wh,
                       options={'maxiter': max_iters, 'disp': False, 'gtol': 1e-5, 'maxls': 20})
        c_opt = res.x
        val_opt = obj_wh(c_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_c = c_opt.copy()
    except Exception:
        pass
    
    # Convert back to h
    h = basis @ best_c
    s = np.sum(h)
    if abs(s) > 1e-9:
        h = h / s
    
    return h, best_val

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
            
        # Compute gradient of ac[max_idx] w.r.t. h
        g = np.zeros(n)
        k = max_idx
        if k < n:
            # ac[k] = sum_{i=0}^{n-1-k} h[i] h[i+k]
            for i in range(n - k):
                g[i] += best_h[i+k]
                g[i+k] += best_h[i]
        else:
            lag = k - (n - 1)
            abs_lag = -lag
            for i in range(abs_lag, n):
                g[i] += best_h[i - abs_lag]
                g[i - abs_lag] += best_h[i]
                
        sign = np.sign(ac[max_idx])
        direction = -sign * g
        
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-10:
            break
        direction = direction / norm_dir
        
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
    deadline = t_start + 145.0

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    inits = []
    inits.append(_gen_gaussian(N))
    inits.append(_gen_cosine(N))
    inits.append(_gen_chirp(N))
    
    rng = np.random.default_rng(42)
    for i in range(4):
        noise = rng.standard_normal(N)
        kernel_size = 5 + i * 2
        kernel = np.ones(kernel_size)/kernel_size
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
    inits.append(rng.standard_normal(N))

    # NEW: Walsh-Hadamard basis optimization
    # Try different numbers of basis functions
    for num_basis in [8, 16, 32]:
        if time.time() > deadline - 30:
            break
        
        try:
            basis = _walsh_hadamard_basis(N, num_basis)
            h_wh, val_wh = _walsh_hadamard_optimize(basis, deadline - 5)
            
            s_wh = np.sum(h_wh)
            if abs(s_wh) > 1e-9:
                h_wh = h_wh / s_wh
                val_wh = _score(h_wh)
                if val_wh < best_score:
                    best_score = val_wh
                    best_h = h_wh.copy()
        except Exception:
            continue

    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
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
