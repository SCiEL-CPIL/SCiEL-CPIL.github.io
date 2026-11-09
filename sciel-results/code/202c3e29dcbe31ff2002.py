# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _objective_fast(x, n=64, n_heights=5000):
    """Fast objective for optimization using fewer heights."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0

def _coord_descent(x, n=64, n_heights=5000, max_rounds=30, tol=1e-12):
    """Coordinate descent with line search for each coordinate."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    
    best_f = _objective_fast(x, n, n_heights)
    
    for rnd in range(max_rounds):
        improved = False
        for i in range(n):
            # Current value
            xi = x[i]
            # Try to find better xi via ternary search or grid
            # The objective as function of xi is piecewise linear (sum of min terms)
            # so it's convex-ish; try a few candidates
            
            # Estimate range for xi
            # L_i(y) = xi + a_i * y, y in [0,1], so L_i ranges over [xi, xi + a_i]
            # We want xi such that the intervals overlap well
            # Try values around current best and some offsets
            
            candidates = [xi]
            # Add perturbations
            for delta in [-0.05, -0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02, 0.05]:
                candidates.append(xi + delta)
            
            # Also try based on a_i: shift to align with neighbors
            # The ideal is for L_i(y) to be close to other L_j(y)
            # At y=0, L_i = x_i; at y=1, L_i = x_i + a_i
            # Try x_i = -a_i * y_center for some y_center
            for yc in [0.25, 0.5, 0.75]:
                candidates.append(-a[i] * yc)
            
            best_candidate = xi
            best_candidate_f = best_f
            
            # Evaluate current
            L = x[None, :] + a[None, :] * y_vals[:, None]
            L_sorted = np.sort(L, axis=1)
            gaps = np.diff(L_sorted, axis=1)
            union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
            current_f = np.mean(union_lens)
            
            for cand in candidates:
                x[i] = cand
                L = x[None, :] + a[None, :] * y_vals[:, None]
                L_sorted = np.sort(L, axis=1)
                gaps = np.diff(L_sorted, axis=1)
                union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
                f_cand = np.mean(union_lens)
                if f_cand < best_candidate_f:
                    best_candidate_f = f_cand
                    best_candidate = cand
            
            x[i] = best_candidate
            if best_candidate_f < best_f - tol:
                best_f = best_candidate_f
                improved = True
        
        if not improved:
            break
    
    return x, best_f

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 610  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area(x0)
    
    # Strategy 1: Coordinate descent with moderate heights
    x_cd, f_cd = _coord_descent(best_x.copy(), n, n_heights=5000, max_rounds=50)
    if f_cd < best_area:
        best_x = x_cd
        best_area = f_cd
    
    if time.time() > deadline - 120:
        # Final polish
        final_area = _compute_area(best_x)
        if final_area <= 0 or not np.all(np.isfinite(best_x)):
            best_x = x0
        return best_x.astype(np.float64)
    
    # Strategy 2: More rounds of coordinate descent with finer grid
    x_cd2, f_cd2 = _coord_descent(best_x.copy(), n, n_heights=8000, max_rounds=40, tol=1e-13)
    if f_cd2 < best_area:
        best_x = x_cd2
        best_area = f_cd2
    
    if time.time() > deadline - 60:
        final_area = _compute_area(best_x)
        if final_area <= 0 or not np.all(np.isfinite(best_x)):
            best_x = x0
        return best_x.astype(np.float64)
    
    # Strategy 3: Final high-precision coordinate descent
    x_cd3, f_cd3 = _coord_descent(best_x.copy(), n, n_heights=20000, max_rounds=15, tol=1e-14)
    if f_cd3 < best_area:
        best_x = x_cd3
        best_area = f_cd3
    
    # Verify with full precision
    final_area = _compute_area(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        # Fallback to Keich
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
