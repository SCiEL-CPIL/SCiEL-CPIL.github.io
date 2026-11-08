import numpy as np
import time
from scipy.optimize import minimize

N = 600
dx = 0.5 / N  # interval width

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def compute_autoconv(x):
        """Compute the autoconvolution of x (normalized)."""
        x = np.maximum(x, 0)
        s = np.sum(x)
        if s < 1e-15:
            return None
        x_norm = x / s
        n_fft = 2 * N
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x_norm
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real
        return conv
    
    def autoconvolution_max(x):
        """Compute the normalized maximum of the autoconvolution."""
        conv = compute_autoconv(x)
        if conv is None:
            return 1e15
        return np.max(conv)
    
    def smooth_max_logsumexp(conv, beta=100.0):
        """Smooth approximation to max using log-sum-exp."""
        conv_clipped = np.clip(conv, 1e-300, None)
        m = np.max(conv_clipped)
        return m + np.log(np.sum(np.exp(beta * (conv_clipped - m)))) / beta
    
    def objective_smooth(x, beta=100.0):
        """Smooth objective using log-sum-exp surrogate for max."""
        conv = compute_autoconv(x)
        if conv is None:
            return 1e15
        return smooth_max_logsumexp(conv, beta)
    
    def objective(x):
        return autoconvolution_max(x)
    
    # Start with a smooth raised-cosine-like distribution
    t = np.linspace(0, 1, N)
    # Use a broad bump: cosine raised to a power to make it flatter in the middle
    x0 = np.cos(np.pi * (t - 0.5))**2  # This is a smooth bump, zero at edges
    x0 = np.maximum(x0, 0)
    
    # Ensure it's non-negative and has reasonable magnitude
    x0 = x0 / np.sum(x0) * N  # Scale so integral is ~1
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try different initial shapes and optimize
    candidates = [
        x0.copy(),
        np.ones(N),  # Uniform
        np.cos(np.pi * (np.linspace(0, 1, N) - 0.5))**1.5,  # Sharper cosine
        np.exp(-((np.linspace(0, 1, N) - 0.5)**2) / 0.1),  # Gaussian-like
        np.exp(-((np.linspace(0, 1, N) - 0.5)**2) / 0.2),  # Broader Gaussian
    ]
    
    # First pass: use smooth objective for better gradient behavior
    for x_init in candidates:
        x_init = np.maximum(x_init, 0)
        if np.sum(x_init) < 1e-10:
            continue
        x_init = x_init / np.sum(x_init) * N
        
        if time.time() > deadline - 30:
            break
        
        try:
            # Use smooth objective with moderate beta
            result = minimize(lambda v: objective_smooth(v, beta=50.0), x_init.copy(), 
                            method='L-BFGS-B',
                            bounds=[(0, None)] * N,
                            options={'maxiter': 2000, 'maxfun': 20000, 'gtol': 1e-8})
            if result.fun < best_obj:
                best_x = result.x
                best_obj = objective(result.x)
        except Exception:
            pass
    
    # Second pass: refine with exact max objective
    for x_init in [best_x.copy()]:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(objective, x_init.copy(), method='L-BFGS-B',
                            bounds=[(0, None)] * N,
                            options={'maxiter': 5000, 'maxfun': 50000, 'gtol': 1e-12})
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Local refinement with full coordinate descent
    if time.time() < deadline - 15:
        # Adaptive step size based on current scale
        base_step = np.mean(best_x) * 0.1
        step_sizes = [base_step * 0.1, base_step * 0.5, base_step]
        
        for step_size in step_sizes:
            if time.time() > deadline - 5:
                break
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    
                    # Try increasing
                    x_trial = best_x.copy()
                    x_trial[i] = max(0, x_trial[i] + step_size)
                    obj_trial = objective(x_trial)
                    if obj_trial < best_obj - 1e-14:
                        best_x = x_trial
                        best_obj = obj_trial
                        improved = True
                    
                    # Try decreasing
                    if best_x[i] > step_size:
                        x_trial = best_x.copy()
                        x_trial[i] = best_x[i] - step_size
                        obj_trial = objective(x_trial)
                        if obj_trial < best_obj - 1e-14:
                            best_x = x_trial
                            best_obj = obj_trial
                            improved = True
                
                if not improved:
                    break
    
    # Final high-precision polish: very small steps
    if time.time() < deadline - 5:
        final_step = np.mean(best_x) * 0.01
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                
                # Try small increase
                x_trial = best_x.copy()
                x_trial[i] = max(0, x_trial[i] + final_step)
                obj_trial = objective(x_trial)
                if obj_trial < best_obj - 1e-16:
                    best_x = x_trial
                    best_obj = obj_trial
                    improved = True
                
                # Try small decrease
                if best_x[i] > final_step:
                    x_trial = best_x.copy()
                    x_trial[i] = best_x[i] - final_step
                    obj_trial = objective(x_trial)
                    if obj_trial < best_obj - 1e-16:
                        best_x = x_trial
                        best_obj = obj_trial
                        improved = True
            
            if not improved:
                break
    
    # Final cleanup: ensure nonnegativity
    best_x = np.maximum(best_x, 0)
    
    return list(best_x)
