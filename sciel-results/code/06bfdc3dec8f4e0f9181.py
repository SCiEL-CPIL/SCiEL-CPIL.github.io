import numpy as np
from scipy.optimize import differential_evolution

N = 50

def _objective(x):
    f = np.asarray(x, dtype=float)
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
    score = l2_sq / amax
    return -score

def construct():
    # Bounds for each height. Since we normalize inside, the scale doesn't matter for the objective.
    # We can bound them to [0, 1] to make DE work efficiently.
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Differential Evolution parameters
    # maxiter and popsize control the search depth and breadth.
    # 120s limit allows for a decent number of iterations.
    # N=50 is the dimension.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8, 
            seed=42,
            polish=True
        )
        f_opt = np.maximum(res.x, 0)
        if np.sum(f_opt) > 1e-12:
            f_opt = f_opt / np.sum(f_opt)
        else:
            # Fallback to Gaussian if DE fails completely
            x = np.arange(N)
            center = N / 2.0
            sigma = N / 4.0
            f_init = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            f_opt = f_init / np.sum(f_init)
    except Exception:
        x = np.arange(N)
        center = N / 2.0
        sigma = N / 4.0
        f_init = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        f_opt = f_init / np.sum(f_init)
        
    return list(f_opt)
