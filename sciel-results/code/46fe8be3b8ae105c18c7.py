import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    N = 50
    DELTA = 0.5 / N
    
    # Symmetric parameterization
    # h is symmetric: h[i] == h[N-1-i]
    # We define first N//2 elements, then mirror.
    M = N // 2
    
    def make_h(x):
        # x is length M
        # Softmax on the symmetric part
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        h_half = exp_x / np.sum(exp_x)
        
        h = np.zeros(N)
        h[:M] = h_half
        h[M:] = h_half[::-1]
        
        # Normalize to sum 1
        h = h / np.sum(h)
        return h

    def objective(x):
        h = make_h(x)
        
        l2_sq = np.sum(h**2)
        
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        
        if l2_sq < 1e-15 or max_conv < 1e-15:
            return 1e15
            
        return max_conv / l2_sq

    # Bounds for x
    bounds = [(-10, 10) for _ in range(M)]
    
    # Run Differential Evolution
    res = differential_evolution(objective, bounds, maxiter=100, popsize=20, tol=1e-8, seed=42)
    
    best_x = res.x
    best_h = make_h(best_x)
    
    # Refine with L-BFGS-B on the symmetric parameters
    res_refine = minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 500})
    best_x = res_refine.x
    best_h = make_h(best_x)
    
    l2_sq = np.sum(best_h**2)
    h_conv = np.convolve(best_h, best_h, mode='full')
    max_conv = np.max(h_conv)
    
    score = (l2_sq / (np.sum(best_h) * DELTA * max_conv))
    
    return list(best_h)
