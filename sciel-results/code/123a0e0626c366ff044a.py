import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Best score found so far
    best_score = 0.0
    best_h = np.ones(N) / N  # Uniform baseline
    
    def compute_ratio(h):
        # h is normalized to sum to 1
        # L2 norm squared
        l2_sq = np.sum(h**2)
        # L1 norm is 1.0
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        # Supremum norm
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        return l2_sq / max_conv

    # Initial score for uniform
    best_score = compute_ratio(best_h)
    
    # Use L-BFGS-B to optimize
    # We optimize over z where h = softmax(z)
    # This ensures h > 0 and sum(h) = 1
    
    def objective(z):
        h = np.exp(z - np.max(z))  # Stability
        h /= np.sum(h)
        r = compute_ratio(h)
        return -r  # Minimize negative ratio
    
    # Gradient of objective with respect to z is complex, 
    # but L-BFGS-B can approximate it or we can use finite differences.
    # Given N=50, finite differences are feasible if we are careful.
    # However, analytic gradient is better.
    # Let's stick to BFGS with numerical gradients first for simplicity and robustness.
    
    num_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and num_restarts < max_restarts:
        # Random initialization
        # Mix of Gaussian-like and Uniform
        if num_restarts % 2 == 0:
            # Centered Gaussian
            x = np.linspace(-1, 1, N)
            sigma = 0.2 + 0.3 * np.random.rand()
            z0 = -x**2 / (2 * sigma**2)
        else:
            # Multi-peak
            z0 = np.random.rand(N) * 0.5
        
        # Optimize
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            h_opt = np.exp(res.x - np.max(res.x))
            h_opt /= np.sum(h_opt)
            
            score = compute_ratio(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = h_opt
                
        except Exception:
            pass
            
        num_restarts += 1
        
        # If we are doing well, we might spend more time, but we have a hard deadline.
        
    return best_h.tolist()

# EVOLVE-BLOCK-START
# The function construct is defined above. 
# To adhere to the "Keep function names" rule, we just ensure construct is the entry point.
# The code above defines construct.
# EVOLVE-BLOCK-END
