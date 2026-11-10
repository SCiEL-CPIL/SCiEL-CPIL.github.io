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
                c1 = N * (X[j] - X[i]) / denom
                if 0.0 < c1 < 1.0:
                    breakpoints.add(c1)
                c2 = N * (X[j] + w - X[i]) / denom
                if 0.0 < c2 < 1.0:
                    breakpoints.add(c2)
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

            Ls = X + (np.arange(N) / N) * c_mid
            Rs = Ls + w

            order = np.argsort(Ls)
            Ls_sorted = Ls[order]
            Rs_sorted = Rs[order]

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

    def compute_cost_reduced(X_reduced):
        """
        Compute cost with x_0 fixed to 0.
        X_reduced has length N-1 (x_1 through x_{N-1}).
        """
        X = np.empty(N, dtype=np.float64)
        X[0] = 0.0
        X[1:] = X_reduced
        return compute_cost(X)

    # Keich initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = list(str(bin(idx))[2:])
        eps = np.array(bits, dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Center the Keich positions and fix x_0 = 0
    keich -= keich[0]

    best_X = keich.copy()
    best_cost = compute_cost(keich)

    # Use Differential Evolution for global search on reduced variables (x_1..x_7)
    bounds = [(-1.0, 1.0) for _ in range(N - 1)]
    
    if time.time() < deadline - 150:
        try:
            res = differential_evolution(compute_cost_reduced, bounds, seed=42, maxiter=2000, tol=1e-10, popsize=30)
            X_reduced = res.x
            X_full = np.empty(N, dtype=np.float64)
            X_full[0] = 0.0
            X_full[1:] = X_reduced
            cost = compute_cost(X_full)
            if cost < best_cost:
                best_cost = cost
                best_X = X_full.copy()
        except Exception:
            pass

    # Local refinement with Nelder-Mead on reduced variables
    if time.time() < deadline - 80:
        try:
            X_reduced_init = best_X[1:].copy()
            res = minimize(compute_cost_reduced, X_reduced_init, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            X_reduced = res.x
            X_full = np.empty(N, dtype=np.float64)
            X_full[0] = 0.0
            X_full[1:] = X_reduced
            cost = compute_cost(X_full)
            if cost < best_cost:
                best_cost = cost
                best_X = X_full.copy()
        except Exception:
            pass

    # Powell refinement on reduced variables
    if time.time() < deadline - 40:
        try:
            X_reduced_init = best_X[1:].copy()
            res = minimize(compute_cost_reduced, X_reduced_init, method='Powell',
                           options={'maxiter': 10000, 'xtol': 1e-16, 'ftol': 1e-18})
            X_reduced = res.x
            X_full = np.empty(N, dtype=np.float64)
            X_full[0] = 0.0
            X_full[1:] = X_reduced
            cost = compute_cost(X_full)
            if cost < best_cost:
                best_cost = cost
                best_X = X_full.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 15:
        try:
            X_reduced_init = best_X[1:].copy()
            res = minimize(compute_cost_reduced, X_reduced_init, method='Nelder-Mead',
                           options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True})
            X_reduced = res.x
            X_full = np.empty(N, dtype=np.float64)
            X_full[0] = 0.0
            X_full[1:] = X_reduced
            cost = compute_cost(X_full)
            if cost < best_cost:
                best_cost = cost
                best_X = X_full.copy()
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
