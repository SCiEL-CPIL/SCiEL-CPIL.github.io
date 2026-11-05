"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _objective(x):
    h = np.exp(x)
    S = np.sum(h)
    if S == 0:
        return 1e10
    c = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(c) / (S ** 2))


def construct():
    import time
    t0 = time.time()
    deadline = 140  # 10s safety margin before 150s kill
    
    # Start from all-ones (in log space, log(1) = 0)
    x0 = np.zeros(N)
    
    # Try a few random restarts with small perturbations
    best_val = _objective(x0)
    best_x = x0.copy()
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        # Perturb the starting point
        x = x0 + rng.normal(0, 0.1, size=N)
        res = minimize(_objective, x, method='L-BFGS-B', 
                       options={'maxiter': 100, 'gtol': 1e-8})
        if time.time() > deadline:
            break
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    
    h = np.exp(best_x)
    # Normalize so sum is 1 for stability (doesn't change the objective)
    h = h / np.sum(h)
    
    return h
# EVOLVE-BLOCK-END
