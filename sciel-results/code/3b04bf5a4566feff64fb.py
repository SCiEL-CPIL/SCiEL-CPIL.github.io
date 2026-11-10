import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Strategy:
    1. Multiple high-quality initializations (Aligned, Keich, Random).
    2. Cyclic Coordinate Descent with Ternary Search (exploits convexity in each coordinate).
    3. Nelder-Mead polish for any remaining non-convexity artifacts or numerical noise.
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

    def coordinate_descent(x, max_rounds=10, tol=1e-9, deadline=None, safety_margin=30):
        """
        Perform cyclic coordinate descent.
        Since the objective is convex in each coordinate x_i (for fixed others),
        ternary search will find the local (and thus global for that coordinate) minimum.
        """
        x = x.copy()
        current_val = compute_area(x)
        
        for round_num in range(max_rounds):
            if deadline is not None and time.time() > deadline - safety_margin:
                break
            
            improved = False
            for i in range(n):
                if deadline is not None and time.time() > deadline - safety_margin:
                    break
                
                # Define range for x_i. 
                # Typically x_i are in [-1, 1]. We can tighten this.
                # Let's use a wide range initially, then narrow.
                lo = x[i] - 0.5
                hi = x[i] + 0.5
                
                # Ternary search for the minimum in [lo, hi]
                # f is convex/unimodal
                a_t = lo
                b_t = hi
                c_t = a_t + (b_t - a_t) / 3.0
                d_t = b_t - (b_t - a_t) / 3.0
                
                # Save original x_i
                x_old = x[i]
                
                # Evaluate at c and d
                x[i] = c_t
                f_c = compute_area(x)
                x[i] = d_t
                f_d = compute_area(x)
                
                # Iterate to converge
                for _ in range(50): # Max iterations for ternary search
                    if deadline is not None and time.time() > deadline - safety_margin:
                        break
                    if f_c < f_d:
                        b_t = d_t
                        d_t = c_t
                        f_d = f_c
                        c_t = a_t + (b_t - a_t) / 3.0
                        x[i] = c_t
                        f_c = compute_area(x)
                    else:
                        a_t = c_t
                        c_t = d_t
                        f_c = f_d
                        d_t = b_t - (b_t - a_t) / 3.0
                        x[i] = d_t
                        f_d = compute_area(x)
                    
                    if abs(b_t - a_t) < 1e-9:
                        break
                
                # The minimum is roughly in the middle of the final interval
                x_best = (a_t + b_t) / 2.0
                x[i] = x_best
                new_val = compute_area(x)
                
                if new_val < current_val - tol:
                    current_val = new_val
                    improved = True
                else:
                    # Revert if no improvement
                    x[i] = x_old
                    # If we reverted, the value is current_val (which was computed before this coordinate step)
                    # Actually, if we reverted, x is back to previous state, so current_val is still valid.
            
            if not improved:
                break
        
        return x, current_val

    # Initializations
    inits = []
    
    # 1. Aligned: all converge to (0.5, 1)
    x_aligned = 0.5 - a
    inits.append(x_aligned.copy())
    
    # 2. Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    inits.append(np.array(positions, dtype=np.float64))
    
    # 3. Random perturbations around aligned
    rng = np.random.default_rng(42)
    for trial in range(5):
        x_pert = x_aligned + rng.normal(0, 0.1, size=n)
        inits.append(x_pert)

    # Evaluate all inits and pick the best
    best_x = inits[0].copy()
    best_val = compute_area(best_x)
    for x0 in inits[1:]:
        val = compute_area(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    start_time = time.time()
    deadline = start_time + 600
    safety_margin = 30

    # Coordinate Descent
    x_cd, val_cd = coordinate_descent(best_x, max_rounds=15, deadline=deadline, safety_margin=safety_margin)
    if val_cd < best_val:
        best_x = x_cd
        best_val = val_cd

    # Nelder-Mead Polish
    if time.time() < deadline - 60:
        nm_options = {
            'maxiter': 200,
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

    # Final Coordinate Descent Pass
    if time.time() < deadline - 30:
        x_cd2, val_cd2 = coordinate_descent(best_x, max_rounds=5, deadline=deadline, safety_margin=safety_margin)
        if val_cd2 < best_val:
            best_x = x_cd2
            best_val = val_cd2

    if not np.all(np.isfinite(best_x)):
        return inits[0].copy()

    return best_x
