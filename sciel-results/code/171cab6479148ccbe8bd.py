import numpy as np
from scipy.optimize import differential_evolution

def construct():
    N = 50
    DELTA = 0.5 / N
    
    def softmax(x):
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def objective(x):
        h = softmax(x)
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-15:
            return 1e15
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 1e15
        return max_conv / l2_sq

    # Bounds for the log-softmax parameters
    # Large enough to allow very peaked distributions
    bounds = [(-10, 10) for _ in range(N)]
    
    # Run Differential Evolution
    # popsize=20, tol=1e-8, maxiter=200
    # This might take a while, let's estimate.
    # 50 dims.
    # Let's try to keep it fast.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            popsize=20, 
            tol=1e-8, 
            maxiter=100, 
            seed=42,
            polish=True # Use L-BFGS-B on the best result found by DE
        )
        best_x = res.x
    except Exception:
        # Fallback to random if DE fails or times out (though we can't easily catch timeout in standard lib without threads)
        best_x = np.zeros(N)

    # Convert back to h
    best_h = softmax(best_x)
    
    # Calculate score
    l2_sq = np.sum(best_h**2)
    h_conv = np.convolve(best_h, h_conv if False else best_h, mode='full') # Wait, h_conv variable name collision in thought, use best_h
    h_conv = np.convolve(best_h, best_h, mode='full')
    max_conv = np.max(h_conv)
    
    score = (l2_sq / (np.sum(best_h) * DELTA * max_conv))
    
    return list(best_h)
