import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = (np.arange(n) + 1) / n
    
    # Coarse grid for fast exploration
    M_coarse = 2000
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    w_coarse = (1 - y_coarse) / n
    
    # Full grid for accurate evaluation
    M_full = 20000
    y_full = (np.arange(M_full) + 0.5) / M_full
    w_full = (1 - y_full) / n

    def calc_area(x, y_vals, w_vals):
        L = x[None, :] + a[None, :] * y_vals[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)
        min_val = np.minimum(diffs, w_vals[:, None])
        union_lengths = w_vals + np.sum(min_val, axis=1)
        return np.mean(union_lengths)

    def area_coarse(x):
        return calc_area(x, y_coarse, w_coarse)

    def area_full(x):
        return calc_area(x, y_full, w_full)

    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated Annealing on coarse grid
    # Start with fan shape
    x = -a.copy()
    current_area = area_coarse(x)
    best_x_coarse = x.copy()
    best_area_coarse = current_area
    
    T = 0.05
    cooling = 0.9995
    steps = 20000
    noise_scale = 0.05
    
    for i in range(steps):
        noise = rng.normal(0, noise_scale, n)
        x_new = x + noise
        area_new = area_coarse(x_new)
        
        delta = area_new - current_area
        if delta < 0 or rng.random() < np.exp(-delta / T):
            x = x_new
            current_area = area_new
            if current_area < best_area_coarse:
                best_area_coarse = current_area
                best_x_coarse = x.copy()
        
        T *= cooling
        if T < 1e-4:
            break

    # Phase 2: Refine top candidates with Nelder-Mead on full grid
    # We only have one best candidate from SA. Let's also keep the initial guess.
    candidates = [best_x_coarse, -a.copy()]
    
    # Add a few random perturbations of the best coarse solution
    for _ in range(4):
        candidates.append(best_x_coarse + rng.normal(0, 0.02, n))

    best_x_final = None
    best_area_final = np.inf

    for cand in candidates:
        # Nelder-Mead is a derivative-free local optimizer
        res = minimize(
            area_full,
            cand,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'maxfev': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if res.fun < best_area_final:
            best_area_final = res.fun
            best_x_final = res.x

    return best_x_final.astype(np.float64)
