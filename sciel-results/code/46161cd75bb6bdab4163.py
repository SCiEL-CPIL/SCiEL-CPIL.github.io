import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570.0

    N = 8
    w = 1.0 / N

    def get_breakpoints(X):
        """Calculate all c in (0, 1) where the order of endpoints changes."""
        breakpoints = {0.0, 1.0}
        for i in range(N):
            for j in range(i + 1, N):
                denom = float(i - j)
                # L_i = L_j: X[i] + i*c/N = X[j] + j*c/N => c = N*(X[j]-X[i])/(i-j)
                c1 = N * (X[j] - X[i]) / denom
                if 0.0 < c1 < 1.0:
                    breakpoints.add(c1)
                # L_i = R_j: X[i] + i*c/N = X[j] + j*c/N + 1/N
                c2 = N * (X[j] + w - X[i]) / denom
                if 0.0 < c2 < 1.0:
                    breakpoints.add(c2)
                # R_i = L_j: X[i] + i*c/N + 1/N = X[j] + j*c/N
                c3 = N * (X[j] - X[i] - w) / denom
                if 0.0 < c3 < 1.0:
                    breakpoints.add(c3)
        return sorted(breakpoints)

    def compute_cost(X):
        """
        Compute integral of (Union Length)^2 over c in [0, 1].
        X is an array of length N.
        """
        if not np.all(np.isfinite(X)):
            return 1e10

        breakpoints = get_breakpoints(X)
        total_integral = 0.0

        for k in range(len(breakpoints) - 1):
            c_start = breakpoints[k]
            c_end = breakpoints[k + 1]
            if c_end - c_start < 1e-15:
                continue

            c_mid = 0.5 * (c_start + c_end)

            # Endpoints at c_mid
            Ls = X + (np.arange(N) / N) * c_mid
            Rs = Ls + w

            # Sort intervals by start
            order = np.argsort(Ls)
            Ls_sorted = Ls[order]
            Rs_sorted = Rs[order]

            # Merge intervals
            merged_len = 0.0
            cur_l = Ls_sorted[0]
            cur_r = Rs_sorted[0]
            
            for m in range(1, N):
                l = Ls_sorted[m]
                r = Rs_sorted[m]
                if l > cur_r + 1e-15:
                    merged_len += cur_r - cur_l
                    cur_l, cur_r = l, r
                else:
                    if r > cur_r:
                        cur_r = r
            merged_len += cur_r - cur_l
            
            # Slope of union length
            # We need to track which intervals form the union to calculate the slope exactly.
            # However, for Nelder-Mead, a finite difference approximation of the slope might be sufficient 
            # or we can just evaluate the integral at midpoints with Simpson's rule if we want to be faster, 
            # but the current exact method is good. Let's stick to the exact slope calculation from the original code.
            
            # Re-merging with indices for slope
            idx_sorted = order
            merged = []
            cur_l = Ls_sorted[0]
            cur_r = Rs_sorted[0]
            cur_li = idx_sorted[0]
            cur_ri = idx_sorted[0]

            for m in range(1, N):
                l = Ls_sorted[m]
                r = Rs_sorted[m]
                li = idx_sorted[m]
                ri = idx_sorted[m]
                if l > cur_r + 1e-15:
                    merged.append((cur_l, cur_r, cur_li, cur_ri))
                    cur_l, cur_r, cur_li, cur_ri = l, r, li, ri
                else:
                    if r > cur_r:
                        cur_r = r
                        cur_ri = ri
            merged.append((cur_l, cur_r, cur_li, cur_ri))

            slope = 0.0
            for _, _, li, ri in merged:
                slope += (ri - li) / N

            L_mid = sum(r - l for l, r, _, _ in merged)
            
            A = slope
            B = L_mid - A * c_mid

            def F(c):
                return (A * A * c * c * c) / 3.0 + (A * B * c * c) + (B * B * c)

            total_integral += F(c_end) - F(c_start)

        return total_integral

    # Keich initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = list(str(bin(idx))[2:])
        eps = np.array(bits, dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Center the Keich positions
    keich -= np.mean(keich)

    best_X = keich.copy()
    best_cost = compute_cost(keich)

    # Use Differential Evolution for global search
    # Bounds for x_i. The spread should be small. 
    # Typical optimal spread is around 1.
    bounds = [(-1.0, 1.0) for _ in range(N)]
    
    # We fix x_0 = 0 to remove translation invariance? 
    # DE doesn't handle equality constraints well. 
    # Instead, we can just let it search and the cost is invariant.
    # However, DE might wander. 
    # Let's run DE on the full N variables.
    
    if time.time() < deadline - 120:
        try:
            res = differential_evolution(compute_cost, bounds, seed=42, maxiter=1000, tol=1e-8, popsize=20)
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            res = minimize(compute_cost, best_X, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Powell refinement
    if time.time() < deadline - 30:
        try:
            res = minimize(compute_cost, best_X, method='Powell',
                           options={'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-16})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 10:
        try:
            res = minimize(compute_cost, best_X, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_X[i_flat]
    base_y = best_X[j_flat]

    positions = np.stack([base_x, base_y], axis=1)

    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions
