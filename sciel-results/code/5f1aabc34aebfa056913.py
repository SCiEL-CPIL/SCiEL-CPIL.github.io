# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _calc_area(x):
    """
    Compute the area of the union of 64 triangles.
    x: shape (64,)
    Returns: scalar area
    """
    n = 64
    # Quadrature points
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    # L_i(y) = x_i + a_i * y
    # Shape: (64, 20000)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n  # Shape: (20000,)
    
    # Sort L along axis 0 for each y
    L_sorted = np.sort(L, axis=0)
    
    # Gaps between consecutive sorted L's
    gaps = np.diff(L_sorted, axis=0)  # Shape: (63, 20000)
    
    # Union length = w + sum(min(gaps, w))
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    
    # Area is the mean of the union lengths
    area = union_len.mean()
    return area

def _calc_area_and_grad(x):
    """
    Compute area and gradient for L-BFGS-B polishing.
    """
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n
    
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    active = gaps < w[None, :]
    order = np.argsort(L, axis=0)
    
    grad = np.zeros(n)
    
    if np.any(active):
        rows, cols = np.where(active)
        idx_down = order[rows, cols]
        idx_up = order[rows + 1, cols]
        
        np.add.at(grad, idx_down, -1.0)
        np.add.at(grad, idx_up, 1.0)
        
    grad /= 20000.0
    
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    area = union_len.mean()
    
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # 1. Generate Keich construction as a strong initial guess
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_keich = np.array(positions, dtype=np.float64)
    
    # Evaluate Keich
    area_keich = _calc_area(x_keich)
    best_x = x_keich.copy()
    best_area = area_keich
    
    # 2. Try Differential Evolution (Global Search)
    # We restrict the search space to a reasonable box around the Keich solution
    # The x values in Keich are in [0, 1).
    # Let's search in [-0.5, 1.5] to be safe.
    bounds = [(-0.5, 1.5)] * n
    
    # Check time
    if time.time() < deadline - 100:
        try:
            # Use a small population and maxiter to keep it fast
            # DE can be slow for 64 dims. We use a limited budget.
            res_de = differential_evolution(
                _calc_area, 
                bounds, 
                maxiter=20, 
                popsize=10, 
                tol=1e-4, 
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.success or res_de.fun < best_area:
                # Polish the DE result with L-BFGS-B
                x_de = res_de.x
                res_lf = minimize(
                    _calc_area_and_grad, 
                    x_de, 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res_lf.fun < best_area:
                    best_area = res_lf.fun
                    best_x = res_lf.x
        except Exception:
            pass # If DE fails or times out, keep Keich

    # 3. Polish the Keich solution with L-BFGS-B (often finds a slightly better local min)
    if time.time() < deadline - 20:
        try:
            res_lf_keich = minimize(
                _calc_area_and_grad, 
                x_keich, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res_lf_keich.fun < best_area:
                best_area = res_lf_keich.fun
                best_x = res_lf_keich.x
        except Exception:
            pass

    # 4. Try a few random perturbations of the best solution and polish
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            x_pert = best_x + rng.normal(0, 0.01, n)
            try:
                res_pert = minimize(
                    _calc_area_and_grad, 
                    x_pert, 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': 20, 'ftol': 1e-10}
                )
                if res_pert.fun < best_area:
                    best_area = res_pert.fun
                    best_x = res_pert.x
            except Exception:
                pass

    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        return x_keich
        
    return best_x

# EVOLVE-BLOCK-END
