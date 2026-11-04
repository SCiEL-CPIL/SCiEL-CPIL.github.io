import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0

# EVOLVE-BLOCK-START
def _compute_ratio_fast(h):
    """
    Compute max(autoconv) / l2_sq for step function h.
    Uses FFT on a fine grid for accurate peak detection.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    if np.sum(h) < 1e-15:
        return float('inf')
    
    n_grid = 2048
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    max_conv = np.max(conv)
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return float('inf')
    return max_conv / l2_sq

def _fourier_to_heights(params):
    """
    Convert Fourier parameters to nonnegative step heights.
    params = [a1, b1, a2, b2, ..., aK, bK, c]
    h_i = max(0, c + sum_j a_j * cos(2*pi*(j+1)*t_i + b_j))
    """
    K = len(params) // 2 - 1
    c = params[-1]
    t = np.arange(N) / N
    
    h = np.full(N, c)
    for j in range(K):
        a = params[2*j]
        b = params[2*j+1]
        h += a * np.cos(2 * np.pi * (j+1) * t + b)
    h = np.clip(h, 0.0, None)
    return h

def _smooth_objective(params, temperature=0.01):
    """
    Smooth log-sum-exp surrogate of max(autoconv)/l2_sq.
    This makes the objective differentiable for gradient-based optimization.
    """
    h = _fourier_to_heights(params)
    if np.sum(h) < 1e-15:
        return 1e10
    
    n_grid = 1024
    x_fine = np.linspace(-0.25, 0.25, n_grid, endpoint=False)
    cell_idx = np.clip(((x_fine + 0.25) / W).astype(np.int32), 0, N - 1)
    f_fine = h[cell_idx]
    
    n_fft = 2 * n_grid
    F = np.fft.rfft(f_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    
    # Log-sum-exp smooth max
    conv_shifted = conv / temperature
    max_val = np.max(conv_shifted)
    logsumexp = max_val + np.log(np.sum(np.exp(conv_shifted - max_val)))
    smooth_max = temperature * logsumexp
    
    l2_sq = np.sum(f_fine ** 2) * dx
    if l2_sq < 1e-15:
        return 1e10
    return smooth_max / l2_sq

def _smooth_gradient(params, temperature=0.01):
    """
    Numerical gradient of the smooth objective for L-BFGS-B.
    """
    grad = np.zeros_like(params)
    eps = 1e-7
    f0 = _smooth_objective(params, temperature)
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _smooth_objective(params_plus, temperature)
        grad[i] = (f_plus - f0) / eps
        params[i] = params_plus[i]
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Number of Fourier modes to use
    K = 10  # 10 modes + 1 bias = 21 parameters total
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform (c=1, all a=0, b=0)
    params_uniform = np.zeros(2*K + 1)
    params_uniform[-1] = 1.0
    candidates.append(params_uniform)
    
    # 2. Cosine bell: h = 0.5*(1+cos(2*pi*t))
    params_cosbell = np.zeros(2*K + 1)
    params_cosbell[0] = 0.5  # a1
    params_cosbell[1] = 0.0  # b1
    params_cosbell[-1] = 0.5  # c
    candidates.append(params_cosbell)
    
    # 3. Half cosine: h = max(0, cos(pi*(t-0.5)))
    # This is approximated by cos(pi*t) shifted
    params_halfcos = np.zeros(2*K + 1)
    params_halfcos[0] = 1.0
    params_halfcos[1] = np.pi  # phase shift
    params_halfcos[-1] = 0.0
    candidates.append(params_halfcos)
    
    # 4. Gaussian-like: use multiple low-frequency modes
    for sigma in [0.1, 0.2, 0.3]:
        params_gauss = np.zeros(2*K + 1)
        params_gauss[-1] = 1.0
        # Approximate Gaussian with low-frequency cosines
        for j in range(min(3, K)):
            params_gauss[2*j] = np.exp(-0.5 * (j+1)**2 * (2*np.pi*sigma)**2) * 0.5
            params_gauss[2*j+1] = 0.0
        candidates.append(params_gauss)
    
    # 5. Disjoint pulses pattern (approximate with high-frequency modes)
    params_pulses = np.zeros(2*K + 1)
    params_pulses[-1] = 0.5
    # Use high-frequency mode to create pulse-like pattern
    params_pulses[2*(K-1)] = 0.5
    params_pulses[2*(K-1)+1] = 0.0
    candidates.append(params_pulses)
    
    # 6. Random smooth candidates
    np.random.seed(42)
    for _ in range(15):
        params_rand = np.zeros(2*K + 1)
        params_rand[-1] = np.random.uniform(0.3, 1.0)
        for j in range(K):
            params_rand[2*j] = np.random.uniform(-0.3, 0.3) * np.exp(-0.1*j)
            params_rand[2*j+1] = np.random.uniform(0, 2*np.pi)
        candidates.append(params_rand)
        
    # Evaluate all candidates with the exact ratio
    best_params = candidates[0].copy()
    best_ratio = _compute_ratio_fast(_fourier_to_heights(best_params))
    
    for params in candidates:
        h = _fourier_to_heights(params)
        if np.sum(h) < 1e-15:
            continue
        r = _compute_ratio_fast(h)
        if r < best_ratio:
            best_ratio = r
            best_params = params.copy()
    
    # Local refinement using L-BFGS-B on the smooth surrogate
    temperature = 0.01
    result = None
    
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb the best params for diversity
        if restart == 0:
            x0 = best_params.copy()
        else:
            x0 = best_params + np.random.normal(0, 0.1, size=len(best_params))
        
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(temperature,),
                method='L-BFGS-B',
                jac=_smooth_gradient,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_ratio:
                # Verify with exact ratio
                h_test = _fourier_to_heights(result.x)
                r_test = _compute_ratio_fast(h_test)
                if r_test < best_ratio:
                    best_ratio = r_test
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final local polish with coordinate descent on the full 600-dim space
    h = _fourier_to_heights(best_params)
    step_size = max(0.01, np.max(h) * 0.01)
    
    for iteration in range(300):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        n_trials = 80
        idxs = np.random.choice(N, size=n_trials, replace=False)
        
        for idx in idxs:
            h_old = h[idx]
            
            h[idx] = h_old + step_size
            r_plus = _compute_ratio_fast(h)
            
            h[idx] = h_old - step_size
            r_minus = _compute_ratio_fast(h)
            
            h[idx] = h_old
            
            if r_plus < best_ratio:
                h[idx] = h_old + step_size
                best_ratio = r_plus
                improved = True
            elif r_minus < best_ratio and h[idx] > 0:
                h[idx] = h_old - step_size
                best_ratio = r_minus
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return h.tolist()
# EVOLVE-BLOCK-END
