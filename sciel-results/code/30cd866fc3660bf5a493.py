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
    best_f = None
    best_score = -np.inf
    
    # Define multiple diverse initializations to escape local minima
    inits = []
    
    # 1. Gaussian (original)
    x = np.arange(N)
    center = N / 2.0
    sigma = N / 4.0
    inits.append(np.exp(-0.5 * ((x - center) / sigma) ** 2))
    
    # 2. Uniform
    inits.append(np.ones(N))
    
    # 3. Triangular
    tri = np.zeros(N)
    mid = N // 2
    for i in range(N):
        tri[i] = 1 - abs(i - mid) / (mid + 1)
    inits.append(tri)
    
    # 4. Random smooth (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    r = rng.random(N)
    kernel = np.ones(5) / 5
    r_smooth = np.convolve(r, kernel, mode='same')
    inits.append(r_smooth)
    
    # 5. Spiky (Delta-like at center)
    spike = np.zeros(N)
    spike[mid] = 1.0
    inits.append(spike)
    
    # 6. Two spikes
    two_spike = np.zeros(N)
    two_spike[max(0, mid - 5)] = 0.5
    two_spike[min(N-1, mid + 5)] = 0.5
    inits.append(two_spike)
    
    # 7. Narrow Gaussian
    sigma_narrow = N / 10.0
    inits.append(np.exp(-0.5 * ((x - center) / sigma_narrow) ** 2))
    
    # 8. Wide Gaussian
    sigma_wide = N / 2.0
    inits.append(np.exp(-0.5 * ((x - center) / sigma_wide) ** 2))

    for f_init in inits:
        # Normalize init
        f_init = np.maximum(f_init, 0)
        s = np.sum(f_init)
        if s < 1e-12:
            continue
        f_init = f_init / s
        
        try:
            # Use L-BFGS-B with bounds
            res = minimize(_objective, f_init, method='L-BFGS-B', bounds=[(0, None)] * N, options={'maxiter': 200, 'ftol': 1e-10})
            f_opt = np.maximum(res.x, 0)
            s_opt = np.sum(f_opt)
            if s_opt > 1e-12:
                f_opt = f_opt / s_opt
                score = -_objective(f_opt)
                if score > best_score:
                    best_score = score
                    best_f = f_opt
        except Exception:
            pass

    if best_f is None:
        # Fallback to Gaussian
        x = np.arange(N)
        center = N / 2.0
        sigma = N / 4.0
        best_f = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        best_f = best_f / np.sum(best_f)

    return list(best_f)
