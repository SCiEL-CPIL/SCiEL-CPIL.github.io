import numpy as np
from scipy import optimize
import time

N = 600

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    if np.all(h == 0):
        return 0.0
    conv = np.convolve(h, h, 'full')
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Initial guess: uniform
    h = np.ones(N) / N
    best_h = h.copy()
    best_score = compute_autoconv_max(best_h)
    
    # Objective function with normalization to sum=1 to prevent overflow and improve conditioning
    def soft_objective(x, T=0.02):
        h = np.abs(x)
        total = np.sum(h)
        if total < 1e-10:
            return 1e10
        h = h / total # Normalize to sum=1
        
        conv = np.convolve(h, h, 'full')
        conv_max = np.max(conv)
        # Use log-sum-exp for stability
        soft_max = conv_max + T * np.log(np.sum(np.exp((conv - conv_max) / T)))
        # Since sum(h)=1, the score is just soft_max
        return soft_max

    # Multiple restarts
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline - 20:
            break
            
        if restart == 0:
            h0 = np.ones(N)
        elif restart == 1:
            h0 = 1.0 - np.abs(np.linspace(-1, 1, N))
            h0 = np.maximum(h0, 0)
        elif restart == 2:
            h0 = np.abs(np.linspace(-1, 1, N))
            h0 = np.maximum(h0, 0.1)
        elif restart == 3:
            h0 = np.random.rand(N) + 0.1
        elif restart == 4:
            h0 = 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 1, N)))
            h0 = np.maximum(h0, 0)
        elif restart == 5:
            # Two peaks
            h0 = np.exp(-((np.linspace(-1, 1, N) - 0.5)**2) / 0.1) + np.exp(-((np.linspace(-1, 1, N) + 0.5)**2) / 0.1)
        elif restart == 6:
            # Three peaks
            h0 = np.exp(-((np.linspace(-1, 1, N) - 0.7)**2) / 0.1) + np.exp(-((np.linspace(-1, 1, N))**2) / 0.1) + np.exp(-((np.linspace(-1, 1, N) + 0.7)**2) / 0.1)
        elif restart == 7:
            # Random with smoothing
            h0 = np.random.rand(N)
            h0 = np.convolve(h0, np.ones(10)/10, 'same')
        elif restart == 8:
            # Inverse of previous best
            h0 = 1.0 - best_h / np.max(best_h)
            h0 = np.maximum(h0, 0.01)
        else:
            h0 = np.random.rand(N)
            
        h0 = np.abs(h0)
        
        try:
            result = optimize.minimize(
                soft_objective, h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                h_new = np.abs(result.x)
                actual_score = compute_autoconv_max(h_new)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = h_new.copy()
                    
        except Exception as e:
            pass

    return list(best_h)
