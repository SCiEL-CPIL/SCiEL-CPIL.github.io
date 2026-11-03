import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """
    Compute linear autoconvolution using FFT.
    c[k] = sum_i x[i] x[i-k]
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_raw(x):
    """
    Metric: max |c_k| for all k / S^2
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _objective(x, beta=50.0):
    """
    Smoothed objective for optimization.
    Uses log-sum-exp for soft max.
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    # Soft max
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    candidates = []
    
    # 1. Hanning
    candidates.append(np.hanning(N))
    
    # 2. Kaiser windows
    for beta in [5.0, 10.0, 15.0, 20.0, 30.0]:
        candidates.append(np.kaiser(N, beta))
        
    # 3. Sinc-like
    t = np.linspace(-2, 2, N)
    candidates.append(np.sinc(t))
    
    # 4. Gaussian
    t = np.linspace(-3, 3, N)
    candidates.append(np.exp(-t**2))
    
    # 5. Zadoff-Chu (Real part)
    n_idx = np.arange(N)
    # k=1 is standard
    phase = np.pi * n_idx * (n_idx + 1) / N
    candidates.append(np.cos(phase))
    
    # 6. Random continuous
    candidates.append(rng.normal(0, 1, N))
    
    # 7. Random binary
    candidates.append(rng.choice([-1, 1], N))
    
    # 8. Constant
    candidates.append(np.ones(N))

    for x_init in candidates:
        if time.time() > deadline:
            break
            
        x_start = x_init.copy()
        # Add small noise to help escape symmetry/saddle points
        x_start += rng.normal(0, 0.05, N)
        
        try:
            res = scipy.optimize.minimize(
                _objective, 
                x_start, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            x_opt = res.x
            val = _metric_raw(x_opt)
            
            if val < best_val:
                best_val = val
                best_x = x_opt
                
        except Exception:
            continue
            
    if best_x is None:
        for x_init in candidates:
            val = _metric_raw(x_init)
            if val < best_val:
                best_val = val
                best_x = x_init
                
    return best_x
