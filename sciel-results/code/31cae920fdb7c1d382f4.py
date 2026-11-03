"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    np.random.seed(42)
    random.seed(42)

    best_value = -np.inf
    best_result = None

    # The key insight: for maximizing sum of radii with perimeter constraint,
    # we want to pack circles as tightly as possible.
    # Strategy: use a hexagonal lattice as starting point, then use SLSQP with
    # variable radii to optimize.
    #
    # The rectangle has perimeter 4, so w + h = 2.
    # We'll try several aspect ratios.

    def _make_hex_lattice(w, h, n):
        """Create a hexagonal lattice initialization for n circles."""
        xs = np.zeros(n)
        ys = np.zeros(n)

        # Try different configurations
        # For hexagonal packing, rows alternate offset
        for rows in range(2, 8):
            if rows >= n:
                break
            cols_needed = (n + rows - 1) // rows
            if cols_needed < 2:
                cols_needed = 2

            xs_tmp = np.zeros(n)
            ys_tmp = np.zeros(n)

            # Hexagonal spacing
            # In a hexagonal lattice, vertical spacing is r * sqrt(3)
            # horizontal spacing is 2r, with offset r for odd rows
            # But we're working in normalized coordinates for now

            row_spacing = h / (rows + 0.5)
            col_spacing = w / (cols_needed + 0.3)

            idx = 0
            for row in range(rows):
                for col in range(cols_needed):
                    if idx >= n:
                        break
                    x = (col + 0.5) * col_spacing + col_spacing/2
                    y = (row + 0.5) * row_spacing
                    if row % 2 == 1:
                        x += col_spacing * 0.5
                    xs_tmp[idx] = x
                    ys_tmp[idx] = y
                    idx += 1

            # Check if all within bounds
            valid = True
            for i in range(n):
                if xs_tmp[i] < 0 or xs_tmp[i] > w or ys_tmp[i] < 0 or ys_tmp[i] > h:
                    valid = False
                    break

            if valid and idx == n:
                return xs_tmp, ys_tmp

        # Fallback: grid
        xs = np.zeros(n)
        ys = np.zeros(n)
        rows = 5
        cols = 5
        for i in range(n):
            row = i // cols
            col = i % cols
            xs[i] = (col + 0.5) / cols * w
            ys[i] = (row + 0.5) / rows * h
        return xs, ys

    def _make_grid(w, h, n, rows=5, cols=5):
        xs = np.zeros(n)
        ys = np.zeros(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            xs[i] = (col + 0.5) / cols * w
            ys[i] = (row + 0.5) / rows * h
        return xs, ys

    def _make_random(w, h, n, seed=42):
        rng = np.random.RandomState(seed)
        xs = rng.uniform(0.1*w, 0.9*w, n)
        ys = rng.uniform(0.1*h, 0.9*h, n)
        return xs, ys

    def _objective(params, w, h, W=500.0):
        """
        Objective for variable-radius packing: minimize -sum(ri) + penalties.
        params = [x1, y1, r1, x2, y2, r2, ...] (3*N)
        """
        n = N
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]

        obj = -np.sum(rs)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0

        # Boundary penalties - soft
        for i in range(n):
            xi, yi, ri = xs[i], ys[i], max(rs[i], 1e-8)

            # Left boundary: xi >= ri
            v = xi - ri
            if v < 0:
                pen = W * v * v
                obj += pen
                grad[3*i] += 2 * W * v
                grad[3*i+2] -= 2 * W * v

            # Right boundary: xi + ri <= w
            v = w - xi - ri
            if v < 0:
                pen = W * v * v
                obj += pen
                grad[3*i] += 2 * W * v
                grad[3*i+2] -= 2 * W * v

            # Bottom boundary: yi >= ri
            v = yi - ri
            if v < 0:
                pen = W * v * v
                obj += pen
                grad[3*i+1] += 2 * W * v
                grad[3*i+2] -= 2 * W * v

            # Top boundary: yi + ri <= h
            v = h - yi - ri
            if v < 0:
                pen = W * v * v
                obj += pen
                grad[3*i+1] += 2 * W * v
                grad[3*i+2] -= 2 * W * v

        # Pairwise overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_d = max(rs[i], 1e-8) + max(rs[j], 1e-8)

                if dist_sq < min_d * min_d:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    v = min_d - dist
                    if v > 0:
                        pen = W * v * v
                        obj += pen
                        coeff = 2 * W * v / dist
                        grad[3*i] -= coeff * dx
                        grad[3*i+1] -= coeff * dy
                        grad[3*i+2] -= 2 * W * v
                        grad[3*j] += coeff * dx
                        grad[3*j+1] += coeff * dy
                        grad[3*j+2] -= 2 * W * v

        return obj, grad

    def _check_valid(xs, ys, rs, w, h, tol=1e-6):
        """Check validity of variable-radius configuration."""
        for i in range(N):
            ri = rs[i]
            if ri < 1e-6:
                return False
            if xs[i] - ri < -tol or xs[i] + ri > w + tol:
                return False
            if ys[i] - ri < -tol or ys[i] + ri > h + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if d < rs[i] + rs[j] - tol:
                    return False
        return True

    def _bbox_perim(xs, ys, rs):
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        return 2.0 * ((max_x - min_x) + (max_y - min_y))

    # Try multiple aspect ratios
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w

        # Generate several initializations
        inits = []

        # Hexagonal lattice
        xs_hex, ys_hex = _make_hex_lattice(w, h, N)
        inits.append((xs_hex.copy(), ys_hex.copy()))

        # Grid
        for rows, cols in [(4, 6), (5, 5), (6, 4), (3, 7)]:
            if rows * cols >= N:
                xs_g, ys_g = _make_grid(w, h, N, rows, cols)
                inits.append((xs_g.copy(), ys_g.copy()))

        # Random
        for seed in [1, 2, 3, 42, 100, 200]:
            xs_r, ys_r = _make_random(w, h, N, seed)
            inits.append((xs_r.copy(), ys_r.copy()))

        for xs_init, ys_init in inits:
            if time.time() > deadline - 10:
                break

            # Estimate initial radius
            r_est = 0.0
            for i in range(N):
                min_dist = min(xs_init[i], w - xs_init[i], ys_init[i], h - ys_init[i])
                for j in range(i+1, N):
                    d = np.sqrt((xs_init[i]-xs_init[j])**2 + (ys_init[i]-ys_init[j])**2)
                    min_dist = min(min_dist, d/2.0)
                r_est = min(r_est if r_est > 0 else 1e9, min_dist)
            if r_est <= 0:
                r_est = 0.05 * min(w, h)

            # Variable radius optimization
            params_var = np.concatenate([xs_init, ys_init, np.full(N, r_est)])

            bounds_var = [(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N

            try:
                res_v = minimize(
                    lambda p: _objective(p, w, h),
                    params_var,
                    jac=True,
                    method='SLSQP',
                    bounds=bounds_var,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                p_v = res_v.x
                xs_v = p_v[0::3]
                ys_v = p_v[1::3]
                rs_v = p_v[2::3]

                if _check_valid(xs_v, ys_v, rs_v, w, h):
                    val_v = np.sum(rs_v)
                    if val_v > best_value:
                        best_value = val_v
                        best_result = np.column_stack([xs_v, ys_v, rs_v])
            except Exception:
                pass

            # Also try equal radius as a comparison
            params_eq = np.concatenate([np.column_stack([xs_init, ys_init]).flatten(), [r_est]])
            bounds_eq = [(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)]

            try:
                def _objective_eq(p, w, h, W=500.0):
                    n = N
                    xs = p[0:2*n:2]
                    ys = p[1:2*n:2]
                    r = max(p[-1], 1e-8)

                    obj = -r
                    grad = np.zeros_like(p)
                    grad[-1] = -1.0

                    # Boundary penalties
                    for i in range(n):
                        xi, yi = xs[i], ys[i]
                        v = xi - r
                        if v < 0:
                            pen = W * v * v
                            obj += pen
                            grad[2*i] += 2 * W * v
                            grad[-1] -= 2 * W * v

                        v = w - xi - r
                        if v < 0:
                            pen = W * v * v
                            obj += pen
                            grad[2*i] += 2 * W * v
                            grad[-1] -= 2 * W * v

                        v = yi - r
                        if v < 0:
                            pen = W * v * v
                            obj += pen
                            grad[2*i+1] += 2 * W * v
                            grad[-1] -= 2 * W * v

                        v = h - yi - r
                        if v < 0:
                            pen = W * v * v
                            obj += pen
                            grad[2*i+1] += 2 * W * v
                            grad[-1] -= 2 * W * v

                    # Pairwise overlap
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist_sq = dx*dx + dy*dy
                            min_d = 2.0 * r
                            if dist_sq < min_d * min_d:
                                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                                v = min_d - dist
                                if v > 0:
                                    pen = W * v * v
                                    obj += pen
                                    coeff = 2 * W * v / dist
                                    grad[2*i] -= coeff * dx
                                    grad[2*i+1] -= coeff * dy
                                    grad[-1] -= 2 * W * v

                    return obj, grad

                res_eq = minimize(
                    lambda p: _objective_eq(p, w, h),
                    params_eq,
                    jac=True,
                    method='SLSQP',
                    bounds=bounds_eq,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                p_opt = res_eq.x
                xs_opt = p_opt[0:2*N:2]
                ys_opt = p_opt[1:2*N:2]
                r_opt = p_opt[-1]

                if r_opt > 1e-6:
                    # Check validity
                    valid = True
                    for i in range(N):
                        if xs_opt[i] - r_opt < -1e-6 or xs_opt[i] + r_opt > w + 1e-6:
                            valid = False
                            break
                        if ys_opt[i] - r_opt < -1e-6 or ys_opt[i] + r_opt > h + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                d = np.sqrt((xs_opt[i]-xs_opt[j])**2 + (ys_opt[i]-ys_opt[j])**2)
                                if d < 2*r_opt - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break

                    if valid:
                        val = N * r_opt
                        if val > best_value:
                            best_value = val
                            best_result = np.column_stack([xs_opt, ys_opt, np.full(N, r_opt)])
            except Exception:
                pass

    # Final refinement
    if best_result is not None and time.time() < deadline - 8:
        # Try a few local perturbations
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            perturbed = best_result + np.random.randn(*best_result.shape) * 0.003
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 2.0)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 2.0)
            perturbed[:, 2] = np.clip(perturbed[:, 2], 1e-4, 0.5)

            # Quick validity check
            xs_p = perturbed[:, 0]
            ys_p = perturbed[:, 1]
            rs_p = perturbed[:, 2]
            valid = True
            for i in range(N):
                if xs_p[i] - rs_p[i] < -1e-5 or xs_p[i] + rs_p[i] > 2.0 + 1e-5:
                    valid = False
                    break
                if ys_p[i] - rs_p[i] < -1e-5 or ys_p[i] + rs_p[i] > 2.0 + 1e-5:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((xs_p[i]-xs_p[j])**2 + (ys_p[i]-ys_p[j])**2)
                        if d < rs_p[i] + rs_p[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                val = np.sum(rs_p)
                if val > best_value:
                    best_value = val
                    best_result = perturbed.copy()

    # Return result
    if best_result is None:
        # Fallback: simple grid
        xs, ys = _make_grid(1.0, 1.0, N, 5, 5)
        r = 0.08
        best_result = np.column_stack([xs, ys, np.full(N, r)])

    # Scale to ensure perimeter <= 4
    xs_f = best_result[:, 0]
    ys_f = best_result[:, 1]
    rs_f = best_result[:, 2]

    perim = _bbox_perim(xs_f, ys_f, rs_f)
    if perim > 4.0 + 1e-9:
        scale = 4.0 / perim
        cx = np.mean(xs_f)
        cy = np.mean(ys_f)
        xs_f = cx + (xs_f - cx) * scale
        ys_f = cy + (ys_f - cy) * scale
        rs_f = rs_f * scale

    return np.column_stack([xs_f, ys_f, rs_f])
