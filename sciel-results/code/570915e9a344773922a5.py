import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # 5s safety margin before 150s kill

def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    H = np.fft.rfft(h, size)
    return np.fft.irfft(H * H, size)

def _soft_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _evaluate_exact(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective(h, beta=100.0):
    """
    Objective function for optimization using soft-max smoothing.
    Minimizes the smoothed version of the max autoconvolution term.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _autoconv_fft(h)
    # Use soft max to make the function differentiable
    smax = _soft_max(conv, beta)
    # We want to minimize |2*N*smax/s^2|
    # Since we can't easily handle abs in optimization, we try to make smax negative
    # or minimize the magnitude
    val = 2.0 * N * smax / (s * s)
    return val * val  # Minimize squared to handle sign

def _objective_grad(h, beta=100.0):
    """
    Analytical gradient of the objective function.
    """
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(n)
    
    conv = _autoconv_fft(h)
    
    # Soft max gradient
    x_max = np.max(conv)
    weights = np.exp(beta * (conv - x_max))
    weights /= np.sum(weights)
    
    # d(smax)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_i) = sum_j h_j where k = i + j (with appropriate indexing)
    # For full convolution: conv[k] = sum_i h[i] * h[k-i] for valid indices
    # d(conv_k)/d(h_i) = h[k-i] if 0 <= k-i < n, else 0
    
    # Gradient of soft max wrt h
    grad_smax = np.zeros(n)
    for k in range(2 * n - 1):
        if weights[k] > 1e-10:
            # d(conv_k)/d(h_i) = h[k-i] if 0 <= k-i < n
            i_min = max(0, k - n + 1)
            i_max = min(n - 1, k)
            for i in range(i_min, i_max + 1):
                j = k - i
                if 0 <= j < n:
                    grad_smax[i] += weights[k] * h[j]
    
    # Objective: f = (2*N*smax/s^2)^2
    # df/d(h_i) = 2 * (2*N*smax/s^2) * d(2*N*smax/s^2)/d(h_i)
    # d(2*N*smax/s^2)/d(h_i) = 2*N * (grad_smax[i] * s^2 - smax * 2*s) / s^4
    #                          = 2*N * (grad_smax[i] * s - 2*smax) / s^3
    
    val = 2.0 * N * _soft_max(conv, beta) / (s * s)
    if abs(val) < 1e-15:
        return np.zeros(n)
    
    grad_inner = 2.0 * N * (grad_smax * s - 2.0 * _soft_max(conv, beta)) / (s * s * s)
    grad = 2.0 * val * grad_inner
    
    return grad

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_val = _evaluate_exact(best_h)  # Should be 2.0
    
    # Try various structured initializations
    initializations = []
    t = np.linspace(0, 1, N)
    
    # 1. Constant (baseline)
    initializations.append(np.ones(N))
    
    # 2. Linear ramp
    initializations.append(np.linspace(0.5, 1.5, N))
    
    # 3. Cosine bump
    initializations.append(1.0 + 0.5 * np.cos(2 * np.pi * t))
    
    # 4. Sinc-like
    initializations.append(np.sinc((t - 0.5) * 2) + 0.5)
    
    # 5. Alternating with decay
    initializations.append(np.ones(N) * (1.0 + 0.3 * np.cos(4 * np.pi * t)))
    
    # 6. Gaussian bump
    initializations.append(np.exp(-20 * (t - 0.5)**2) + 0.5)
    
    # 7. Square wave
    initializations.append(np.sign(np.sin(2 * np.pi * t)) * 0.5 + 1.0)
    
    # 8. DCT-based: emphasize low frequencies
    for freq_scale in [1, 2, 5, 10]:
        spectrum = np.exp(-np.arange(N) / freq_scale)
        h_dct = np.real(np.fft.ifft(spectrum * 2))[:N]
        h_dct = h_dct - np.mean(h_dct) + 1.0
        initializations.append(h_dct)
    
    # Evaluate all initializations
    for h_init in initializations:
        h_init = h_init.astype(float)
        if not np.all(np.isfinite(h_init)) or np.sum(h_init) == 0:
            continue
        val = _evaluate_exact(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
        
        if time.time() > DEADLINE:
            break
    
    # Use L-BFGS with soft-max smoothed objective
    if time.time() < DEADLINE - 20:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > DEADLINE - 10:
                break
            
            h_current = best_h.copy()
            
            def obj_wrapper(x):
                return _objective(x, beta)
            
            def grad_wrapper(x):
                return _objective_grad(x, beta)
            
            try:
                result = minimize(
                    obj_wrapper,
                    h_current,
                    jac=grad_wrapper,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                
                if result.success and np.all(np.isfinite(result.x)):
                    val = _evaluate_exact(result.x)
                    if val < best_val:
                        best_val = val
                        best_h = result.x.copy()
            except Exception:
                pass
    
    # Final local search with random perturbations
    if time.time() < DEADLINE - 5:
        h = best_h.copy()
        np.random.seed(42)
        for _ in range(200):
            if time.time() > DEADLINE - 2:
                break
            h_trial = h + np.random.randn(N) * 0.05
            val_trial = _evaluate_exact(h_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_h = h_trial.copy()
                h = best_h.copy()
    
    return best_h
