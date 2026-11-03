import numpy as np
from scipy.optimize import differential_evolution

def construct():
    N = 50
    DELTA = 0.5 / N  # 0.01

    def objective(x):
        # x is the parameter vector
        # Softmax to ensure positive and sum to 1
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        h = exp_x / np.sum(exp_x)
        
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-15:
            return 1e15
            
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        
        if max_conv < 1e-15:
            return 1e15
            
        return max_conv / l2_sq

    # Bounds for differential evolution
    # We can use a wide range for x
    bounds = [(-10, 10) for _ in range(N)]
    
    # Run Differential Evolution
    # maxiter and popsize control the effort. 
    # 120s limit. DE can be slow. 
    # Let's try a moderate setting.
    res = differential_evolution(objective, bounds, maxiter=100, popsize=20, seed=42)
    
    x_best = res.x
    x_max = np.max(x_best)
    exp_x = np.exp(x_best - x_max)
    h = exp_x / np.sum(exp_x)
    
    return list(h)
