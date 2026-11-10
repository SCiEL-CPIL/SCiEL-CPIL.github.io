import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin before 600s limit

    N = 8
    w = 1.0 / N

    def _union_length_at(X, c):
        """Compute the 1D union length at height c for base positions X."""
        Ls = X + (np.arange(N) / N) * c
        Rs = Ls + w
        order = np.argsort(Ls)
        Ls_sorted = Ls[order]
        Rs_sorted = Rs[order]
        
        total = 0.0
        cur_l = Ls_sorted[0]
        cur_r = Rs_sorted[0]
        for m in range(1, N):
            if Ls_sorted[m] > cur_r + 1e-15:
                total += cur_r - cur_l
                cur_l = Ls_sorted[m]
                cur_r = Rs_sorted[m]
            else:
                if Rs_sorted[m] > cur_r:
                    cur_r = Rs_sorted[m]
        total += cur_r - cur_l
        return total

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
        This equals the 3D union volume for the sheared tube construction.
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
            idx_sorted = order

            # Merge intervals, tracking which interval provides left and right endpoints
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

            # Calculate exact slope of union length
            # Each merged component [l_m, r_m] has:
            #   l_m = X[i_m] + (i_m/N)*c
            #   r_m = X[j_m] + (j_m/N)*c + 1/N
            # Component length = (X[j_m] - X[i_m] + 1/N) + ((j_m - i_m)/N)*c
            # Slope of component length = (j_m - i_m)/N
            slope = 0.0
            for _, _, li, ri in merged:
                slope += (ri - li) / N

            # Union length at c_mid
            L_mid = sum(r - l for l, r, _, _ in merged)

            # L(c) = slope * (c - c_mid) + L_mid  (linear within this interval)
            # Let A = slope, B = L_mid - A * c_mid
            # L(c) = A*c + B
            A = slope
            B = L_mid - A * c_mid

            # Integrate (A*c + B)^2 from c_start to c_end
            # = [A^2*c^3/3 + A*B*c^2 + B^2*c] evaluated at endpoints
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

    # Generate diverse initializations
    rng_init = np.random.default_rng(42)
    inits = [keich.copy()]

    # Add structured and random inits
    inits.append(np.zeros(N))
    inits.append(np.arange(N) * w)
    inits.append(-np.arange(N) * w)
    inits.append(np.linspace(0, 0.5, N))
    inits.append(-np.linspace(0, 0.5, N))
    inits.append(np.array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]) * w)
    inits.append(np.array([0, 0.25, 0.5, 0.75, 0, 0.25, 0.5, 0.75]) * w)
    inits.append(np.array([0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]) * w)

    # Random inits
    for _ in range(10):
        inits.append(rng_init.uniform(-0.5, 0.5, N))

    best_X = keich.copy()
    best_cost = compute_cost(keich)

    # Phase 1: L-BFGS-B from multiple starts (gradient-based for smooth regions)
    for init in inits:
        if time.time() > deadline - 200:
            break
        try:
            res = minimize(compute_cost, init, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 100000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement (handles kinks better)
    if time.time() < deadline - 150:
        try:
            res = minimize(compute_cost, best_X, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B again from improved point
    if time.time() < deadline - 100:
        try:
            res = minimize(compute_cost, best_X, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 200000})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead polish with very tight tolerance
    if time.time() < deadline - 50:
        try:
            res = minimize(compute_cost, best_X, method='Nelder-Mead',
                           options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_X = res.x.copy()
        except Exception:
            pass

    # Phase 5: Local perturbation + L-BFGS-B refinement
    if time.time() < deadline - 20:
        rng_local = np.random.default_rng(99)
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            for scale in [0.001, 0.005, 0.01, 0.02]:
                if time.time() > deadline - 10:
                    break
                z0 = best_X + rng_local.normal(0, scale, N)
                try:
                    res = minimize(compute_cost, z0, method='L-BFGS-B',
                                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 50000})
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
