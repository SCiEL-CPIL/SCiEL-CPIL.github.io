# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _exact_1d_union_length(intervals):
    """
    Compute the exact measure (total length) of the union of 1D intervals.
    """
    n = intervals.shape[0]
    if n == 0:
        return 0.0
    
    lo = intervals[:, 0]
    hi = intervals[:, 1]
    
    order = np.argsort(lo, kind='stable')
    lo_s = lo[order]
    hi_s = hi[order]
    
    total = 0.0
    current_lo = lo_s[0]
    current_hi = hi_s[0]
    
    for k in range(1, n):
        if lo_s[k] > current_hi:
            total += current_hi - current_lo
            current_lo = lo_s[k]
            current_hi = hi_s[k]
        else:
            if hi_s[k] > current_hi:
                current_hi = hi_s[k]
    
    total += current_hi - current_lo
    return total


def _cross_section_area_2d(base_x, base_y, i_flat, j_flat, c, N=8):
    """
    Compute the 2D cross-section area of the union at height c.
    The union of axis-aligned rectangles has area = product of 1D union lengths.
    """
    N = float(N)
    shift_x = c * i_flat / N
    shift_y = c * j_flat / N
    
    x_lo = base_x + shift_x
    x_hi = x_lo + 1.0 / N
    y_lo = base_y + shift_y
    y_hi = y_lo + 1.0 / N
    
    x_intervals = np.stack([x_lo, x_hi], axis=1)
    y_intervals = np.stack([y_lo, y_hi], axis=1)
    
    x_len = _exact_1d_union_length(x_intervals)
    y_len = _exact_1d_union_length(y_intervals)
    
    return x_len * y_len


def _union_volume(base_x, base_y, i_flat, j_flat, N=8, n_quad=48):
    """
    Estimate the union volume via Gauss-Legendre quadrature over c in [0, 1].
    """
    c_nodes, c_weights = np.polynomial.legendre.leggauss(n_quad)
    c_nodes = 0.5 * (c_nodes + 1.0)
    c_weights = 0.5 * c_weights
    
    areas = np.array([_cross_section_area_2d(base_x, base_y, i_flat, j_flat, c, N) for c in c_nodes])
    return np.sum(c_weights * areas)


def _objective(params, i_flat, j_flat, N=8, n_quad=40):
    """
    Objective function for optimization: union volume.
    """
    base_x = params[:64]
    base_y = params[64:]
    return _union_volume(base_x, base_y, i_flat, j_flat, N, n_quad)


def _generate_diverse_initializations(rng, n_init=10, base_scale=1.0):
    """
    Generate diverse initializations for multi-start optimization.
    Uses different strategies to explore the parameter space.
    """
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    inits = []
    
    # Strategy 1: Keich-like construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    base_x0 = keich[i_flat]
    base_y0 = keich[j_flat]
    inits.append(np.concatenate([base_x0, base_y0]))
    
    # Strategy 2: Random with small scale
    for _ in range(3):
        scale = rng.uniform(0.1, 0.5)
        x0 = scale * rng.standard_normal(128)
        inits.append(x0)
    
    # Strategy 3: Structured random - group by i and j
    for _ in range(3):
        x_vals = np.zeros(64)
        y_vals = np.zeros(64)
        for i in range(N):
            for j in range(N):
                k = i * N + j
                # Correlate x with i, y with j, but add cross terms
                x_vals[k] = rng.uniform(-0.5, 0.5) + 0.3 * i / N
                y_vals[k] = rng.uniform(-0.5, 0.5) + 0.3 * j / N
        inits.append(np.concatenate([x_vals, y_vals]))
    
    # Strategy 4: Large random perturbations
    for _ in range(1):
        x0 = 1.0 * rng.standard_normal(128)
        inits.append(x0)
    
    return inits


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Minimizes the union volume of 64 sheared tubes using multi-start gradient-based
    optimization with diverse initializations and a time budget.
    """
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    start_time = time.time()
    deadline = start_time + 570  # 570 second deadline (30s safety margin)
    
    # Generate diverse initializations
    rng = np.random.default_rng(42)
    inits = _generate_diverse_initializations(rng, n_init=10)
    
    best_params = inits[0].copy()
    best_vol = _union_volume(best_params[:64], best_params[64:], i_flat, j_flat, N, n_quad=64)
    
    # Multi-start L-BFGS-B optimization
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Use higher precision for the first few starts, lower for later ones
        n_quad = 48 if init_idx < 3 else 32
        maxiter = 300 if init_idx < 3 else 150
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(i_flat, j_flat, N, n_quad),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_vol:
                best_params = result.x.copy()
                best_vol = result.fun
                
                # If we found a good solution, do a refinement pass
                if time.time() < deadline - 30:
                    result2 = minimize(
                        _objective,
                        best_params,
                        args=(i_flat, j_flat, N, 48),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if result2.fun < best_vol:
                        best_params = result2.x.copy()
                        best_vol = result2.fun
                        
        except Exception:
            continue
    
    # Final refinement with best parameters
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective,
                best_params,
                args=(i_flat, j_flat, N, 56),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result_final.fun < best_vol:
                best_params = result_final.x.copy()
                best_vol = result_final.fun
        except Exception:
            pass
    
    # Extract best positions
    base_x = best_params[:64]
    base_y = best_params[64:]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
