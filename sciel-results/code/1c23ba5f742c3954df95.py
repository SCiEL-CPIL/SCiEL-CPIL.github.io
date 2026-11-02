import numpy as np
from scipy.optimize import minimize
import time

N = 600

def autoconvolve(f):
    n = len(f)
    # Next power of 2 >= 2n
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F, size)
    return C[:2*n-1]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    i = np.arange(N)
    center = (N - 1) / 2.0
    
    # Initializations
    # 1. Uniform (log-softmax of zeros)
    x_uniform = np.zeros(N)
    
    # 2. Slightly perturbed uniform
    x_perturbed = np.random.normal(0, 0.1, N)
    
    # 3. Gaussian
    x_gauss = -0.05 * (i - center)**2
    
    # 4. U-shaped
    x_u = 0.05 * (i - center)**2
    
    inits = [x_uniform, x_perturbed, x_gauss, x_u]
    
    best_f = None
    best_score = float('inf')
    
    for x0 in inits:
        if time.time() > deadline - 10:
            break
            
        def objective(x):
            # Softmax for stability
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            f = exp_x / np.sum(exp_x)
            
            # Autoconvolution
            c = autoconvolve(f)
            
            # Max
            return np.max(c)
            
        try:
            # L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                x_opt = res.x
                x_max = np.max(x_opt)
                exp_x = np.exp(x_opt - x_max)
                f_opt = exp_x / np.sum(exp_x)
                
                # Calculate score to compare
                score = np.max(autoconvolve(f_opt))
                
                if score < best_score:
                    best_score = score
                    best_f = f_opt
                    
        except Exception:
            pass
            
    if best_f is not None:
        return best_f.tolist()
    else:
        # Fallback to uniform
        return [1.0/N] * N
