import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Uses multiple initializations, fast coordinate descent with cached sort order,
    and Nelder-Mead refinement.
    """
    n = 64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    w_col = w[:, None]  # shape (M, 1)

    def compute_area(x):
        """
        Computes the area of the union of the triangles.
        x: shape (n,)
        """
        # L_i(y) = x_i + a_i * y
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        min_gaps = np.minimum(gaps, w_col)
        union_len = w + np.sum(min_gaps, axis=1)
        return np.mean(union_len)

    def compute_area_fast(x, sort_order=None):
        """
        Computes area using a cached sort order. If sort_order is provided and
        still valid, skip the sort. Returns (area, new_sort_order).
        """
        L = x[None, :] + a[None, :] * y[:, None]
        
        if sort_order is not None:
            # Check if the sort order is still valid by verifying L values are non-decreasing
            L_permuted = L[:, sort_order]
            diffs = np.diff(L_permuted, axis=1)
            if np.all(diffs >= -1e-15):
                L_sorted = L_permuted
            else:
                L_sorted = np.sort(L, axis=1)
                sort_order = np.argsort(L, axis=1)
        else:
            L_sorted = np.sort(L, axis=1)
            sort_order = np.argsort(L, axis=1)
        
        gaps = np.diff(L_sorted, axis=1)
        min_gaps = np.minimum(gaps, w_col)
        union_len = w + np.sum(min_gaps, axis=1)
        area = np.mean(union_len)
        return area, sort_order

    def coordinate_descent(x, max_rounds=15, tol=1e-11, deadline=None, safety_margin=30):
        """Perform coordinate descent on the area function with time guard and cached sort."""
        x = x.copy()
        prev_val, sort_order = compute_area_fast(x)
        
        for round_num in range(max_rounds):
            if deadline is not None and time.time() > deadline - safety_margin:
                break
            
            improved = False
            for i in range(n):
                if deadline is not None and time.time() > deadline - safety_margin:
                    break
                
                x_old = x[i]
                
                # Adaptive range based on previous step size
                lo = x_old - 0.1
                hi = x_old + 0.1
                
                # Coarse grid search
                grid = np.linspace(lo, hi, 16)
                best_local_val = prev_val
                best_local_x = x_old
                best_local_so = sort_order
                
                for gx in grid:
                    x[i] = gx
                    val, so = compute_area_fast(x, sort_order)
                    if val < best_local_val:
                        best_local_val = val
                        best_local_x = gx
                        best_local_so = so
                
                # Refine around best_local_x
                if best_local_x != x_old:
                    step = (hi - lo) / 16
                    lo2 = best_local_x - step
                    hi2 = best_local_x + step
                    grid2 = np.linspace(lo2, hi2, 12)
                    for gx in grid2:
                        x[i] = gx
                        val, so = compute_area_fast(x, best_local_so)
                        if val < best_local_val:
                            best_local_val = val
                            best_local_x = gx
                            best_local_so = so
                    
                    x[i] = best_local_x
                    sort_order = best_local_so
                    if abs(best_local_val - prev_val) > tol:
                        improved = True
                        prev_val = best_local_val
                else:
                    x[i] = x_old
            
            if not improved:
                break
        
        return x, prev_val

    # Initializations
    inits = []
    
    # Aligned: all converge to (0.5, 1)
    x_aligned = 0.5 - a
    inits.append(x_aligned.copy())
    
    # Converging to 0
    x_conv0 = -a.copy()
    inits.append(x_conv0)
    
    # Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    inits.append(np.array(positions, dtype=np.float64))
    
    # Random perturbations around aligned
    rng = np.random.default_rng(42)
    for trial in range(5):
        x_pert = x_aligned + rng.normal(0, 0.03, size=n)
        inits.append(x_pert)
    
    # More diverse: scaled versions
    for scale in [0.3, 0.7, 1.3]:
        x_scaled = scale * x_aligned
        inits.append(x_scaled.copy())
    
    # Evaluate all inits
    best_x = inits[0].copy()
    best_val = compute_area(best_x)
    for x0 in inits[1:]:
        val = compute_area(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    start_time = time.time()
    deadline = start_time + 630
    safety_margin = 45

    # Coordinate descent from best init
    x_cd, val_cd = coordinate_descent(best_x, max_rounds=15, deadline=deadline, safety_margin=safety_margin)
    if val_cd < best_val:
        best_x = x_cd
        best_val = val_cd

    # Nelder-Mead refinement with time guard
    if time.time() < deadline - 120:
        nm_options = {
            'maxiter': 800,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
        try:
            res = minimize(
                compute_area,
                best_x,
                method='Nelder-Mead',
                options=nm_options
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    # Final coordinate descent pass
    if time.time() < deadline - 60:
        x_cd2, val_cd2 = coordinate_descent(best_x, max_rounds=8, deadline=deadline, safety_margin=safety_margin)
        if val_cd2 < best_val:
            best_x = x_cd2
            best_val = val_cd2

    if not np.all(np.isfinite(best_x)):
        return inits[0].copy()  # Fallback to aligned

    return best_x
