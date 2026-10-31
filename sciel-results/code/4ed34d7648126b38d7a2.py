import numpy as np
from scipy.optimize import minimize

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
    # Initialize with a Gaussian bump
    x = np.arange(N)
    center = N / 2.0
    sigma = N / 4.0
    f_init = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    # Local refinement using BFGS
    try:
        res = minimize(_objective, f_init, method='L-BFGS-B', bounds=[(0, None)] * N)
        f_opt = np.maximum(res.x, 0)
        if np.sum(f_opt) > 1e-12:
            f_opt = f_opt / np.sum(f_opt)
        else:
            f_opt = f_init / np.sum(f_init)
    except Exception:
        f_opt = f_init / np.sum(f_init)
        
    return list(f_opt)
