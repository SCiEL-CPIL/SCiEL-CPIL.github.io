import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

def _score_normalized(f):
    f = np.asarray(f, dtype=float)
    f = np.maximum(f, 0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0
    f = f / l1
    l2_sq = np.sum(f**2)
    conv = np.convolve(f, f, mode='full')
    amax = np.max(conv)
    if amax < 1e-12:
        return 0.0
    return l2_sq / amax

def _objective(x):
    # x is the raw vector from optimizer
    # We want to maximize score, so minimize negative score
    return -_score_normalized(x)

def construct():
    # Bounds: [0, 1] for each element. 
    # Since the score is scale invariant (for the normalized part), 
    # the optimizer will likely push values to boundaries or distribute them.
    # Using [0, 1] is safe.
    bounds = [(0, 1)] * N
    
    # Differential Evolution
    # maxiter and popsize control runtime.
    # N=50. popsize=15 -> 750 evaluations per generation.
    # maxiter=100 -> 75000 evaluations.
    # Each eval is O(N^2) for conv (actually N^2 is 2500, very fast).
    # 75000 * 2500 ops ~ 10^8 ops. Should be fast in NumPy.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            polish=True,  # Uses L-BFGS to refine the best solution
            tol=1e-6
        )
        f_opt = np.maximum(res.x, 0)
        if np.sum(f_opt) > 1e-12:
            f_opt = f_opt / np.sum(f_opt)
        else:
            # Fallback
            f_opt = np.ones(N) / N
    except Exception:
        # Fallback to Gaussian if DE fails
        x = np.arange(N)
        center = N / 2.0
        sigma = N / 4.0
        f_opt = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        f_opt = f_opt / np.sum(f_opt)

    return list(f_opt)
