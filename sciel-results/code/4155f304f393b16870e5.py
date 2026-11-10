# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _keich_positions(n):
    """Generate baseline Keich positions."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        bin_str = bin(i)[2:]
        if len(bin_str) < k:
            bin_str = '0' * (k - len(bin_str)) + bin_str
        eps_arr = np.array(list(bin_str), dtype=int)
        val = np.sum((1 - j) * eps_arr * (2.0 ** -j) / k)
        positions.append(val)
    return np.array(positions, dtype=np.float64)

def _compute_area(x, n=64, num_points=20000):
    """
    Compute the area of the union of the triangles defined by positions x.
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(num_points) + 0.5) / num_points
    
    # L_i(y) = x_i + a_i * y
    L = x[:, None] + a[:, None] * y[None, :]
    
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    
    w = (1 - y) / n
    
    min_diffs = np.minimum(diffs, w[None, :])
    union_lengths = w + np.sum(min_diffs, axis=0)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x0 = _keich_positions(n)
    best_x = x0.copy()
    best_area = _compute_area(best_x, n)
    
    start_time = time.time()
    time_limit = 620.0 # 600s limit, keep 20s margin for safety and return

    def objective(x):
        return _compute_area(x, n)

    # 1. Initial Differential Evolution to explore global landscape
    # DE is robust to local minima and can find better basins than L-BFGS-B starting from Keich
    # We use a relatively tight maxiter to ensure we don't blow the time budget
    # Bounds: Keich positions are roughly in [-0.5, 0.5]. Let's allow a bit more room.
    # Actually, for n=64, the optimal x_i should be close to each other to minimize overlap.
    # Let's set bounds based on the spread of the initial solution plus some margin.
    min_val = np.min(x0) - 1.0
    max_val = np.max(x0) + 1.0
    bounds = [(min_val, max_val)] * n

    # Use a subset of points for DE to speed up evaluation?
    # _compute_area is fast (~1ms for 20k points). DE with popsize=10, maxiter=100 is ~1000 evals.
    # 1000 * 1ms = 1s. This is very fast. We can afford full precision.
    
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100, 
            popsize=15, 
            tol=1e-8, 
            disp=False,
            polish=False # We will polish manually later
        )
        
        if res_de.fun < best_area:
            best_x = res_de.x
            best_area = res_de.fun

    except Exception:
        pass

    # 2. Local Refinement with L-BFGS-B
    # Start from the best of (Keich, DE result)
    start_x = best_x.copy()
    
    try:
        res_lbfgs = minimize(
            objective, 
            start_x, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'maxfun': 500, 'disp': False, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < time_limit:
            if res_lbfgs.fun < best_area:
                best_x = res_lbfgs.x
                best_area = res_lbfgs.fun

    except Exception:
        pass

    # 3. Fine-tuning with Nelder-Mead
    # NM is derivative-free and good for final polishing, especially if the landscape is non-smooth
    # due to the min() in the union calculation (though it's piecewise linear, so gradients exist almost everywhere)
    
    try:
        res_nm = minimize(
            objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'maxfev': 1000, 'disp': False, 'xatol': 1e-6, 'fatol': 1e-10}
        )
        
        if time.time() - start_time < time_limit:
            if res_nm.fun < best_area:
                best_x = res_nm.x
                best_area = res_nm.fun

    except Exception:
        pass

    # 4. Final check
    if not np.all(np.isfinite(best_x)):
        return x0

    return best_x

# EVOLVE-BLOCK-END
