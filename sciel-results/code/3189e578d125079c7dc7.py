import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer before 150s kill)

    # --- Cell assignment: 3x3 grid, 26 circles distributed ---
    # 9 cells, 26 circles -> 8 cells get 3, 1 cell gets 2
    # Assignment: row-major, fill cells
    cell_assignment = []
    for i in range(N):
        row = i // 3
        col = i % 3
        cell_assignment.append((row, col))
    # Ensure exactly 26 entries
    assert len(cell_assignment) == N

    # Cell centers and half-sizes
    def cell_bounds(row, col):
        """Return (x_min, x_max, y_min, y_max) for cell (row, col) in 3x3 grid."""
        x_min = col / 3.0
        x_max = (col + 1) / 3.0
        y_min = row / 3.0
        y_max = (row + 1) / 3.0
        return x_min, x_max, y_min, y_max

    def _eval_packing(xy, r):
        """Evaluate a packing: returns (total_radius, is_valid)."""
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
                return 0.0, False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 0.0, False
        return np.sum(r), True

    def _solve_cell(circles_in_cell, x_min, x_max, y_min, y_max, other_circles_global, deadline):
        """
        Solve a small sub-problem: optimize positions and radii of circles_in_cell
        within the cell bounds, with soft penalties for overlap with other_circles_global.
        
        circles_in_cell: list of indices into the global arrays
        other_circles_global: (xy, r) of all other circles (for boundary penalties)
        """
        idx = circles_in_cell
        k = len(idx)
        if k == 0:
            return

        # Local variables: positions and radii for this cell
        x0 = np.array([xy_g[idx[i], 0] for i in range(k)])
        y0 = np.array([xy_g[idx[i], 1] for i in range(k)])
        r0 = np.array([r_g[idx[i]] for i in range(k)])

        # Other circles (global indices not in this cell)
        mask = np.ones(N, dtype=bool)
        for i in idx:
            mask[i] = False
        other_idx = np.where(mask)[0]

        def cell_objective(params):
            """Minimize negative sum of radii + penalties for overlaps and wall violations."""
            if time.time() > deadline:
                return 1e6
            cx = params[0:k]
            cy = params[k:2*k]
            cr = params[2*k:3*k]
            cr = np.maximum(cr, 1e-6)

            penalty = 0.0
            w = 500.0  # penalty weight

            # Wall constraints (cell boundaries)
            for i in range(k):
                xi, yi, ri = cx[i], cy[i], cr[i]
                penalty += max(0.0, ri - (xi - x_min))**2
                penalty += max(0.0, ri - (x_max - xi))**2
                penalty += max(0.0, ri - (yi - y_min))**2
                penalty += max(0.0, ri - (y_max - yi))**2

            # Overlap within cell
            for i in range(k):
                for j in range(i+1, k):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = cr[i] + cr[j]
                    if dist < min_d:
                        penalty += (min_d - dist)**2

            # Overlap with other cells' circles (only if close)
            for oi in other_idx:
                ox, oy, orr = xy_g[oi, 0], xy_g[oi, 1], r_g[oi]
                for i in range(k):
                    dx = cx[i] - ox
                    dy = cy[i] - oy
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = cr[i] + orr
                    if dist < min_d:
                        penalty += (min_d - dist)**2

            return -np.sum(cr) + w * penalty

        bounds = []
        for i in range(k):
            bounds.append((x_min + 1e-4, x_max - 1e-4))
            bounds.append((y_min + 1e-4, y_max - 1e-4))
            bounds.append((1e-4, 0.3))

        params0 = np.concatenate([x0, y0, r0])

        try:
            res = minimize(cell_objective, params0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-12})
            if res.success or res.fun < cell_objective(params0):
                cx = res.x[0:k]
                cy = res.x[k:2*k]
                cr = np.maximum(res.x[2*k:3*k], 1e-4)
                for i in range(k):
                    xy_g[idx[i], 0] = cx[i]
                    xy_g[idx[i], 1] = cy[i]
                    r_g[idx[i]] = cr[i]
        except Exception:
            pass

    # --- Global arrays ---
    xy_g = np.zeros((N, 2))
    r_g = np.zeros(N)

    # Initial placement: center of each cell with small radius
    for i in range(N):
        row, col = cell_assignment[i]
        x_min, x_max, y_min, y_max = cell_bounds(row, col)
        # Spread circles within cell
        pos_in_cell = i % 3  # 0, 1, 2 within cell
        # 3 circles per cell: place at thirds
        if pos_in_cell == 0:
            px, py = x_min + (x_max - x_min) * 0.25, y_min + (y_max - y_min) * 0.5
        elif pos_in_cell == 1:
            px, py = x_min + (x_max - x_min) * 0.5, y_min + (y_max - y_min) * 0.75
        else:
            px, py = x_min + (x_max - x_min) * 0.75, y_min + (y_max - y_min) * 0.25
        xy_g[i, 0] = px
        xy_g[i, 1] = py
        r_g[i] = 0.05  # initial small radius

    # --- Hierarchical optimization loop ---
    # Phase 1: Solve each cell independently (no inter-cell coupling yet)
    for phase in range(3):
        if time.time() > deadline - 20.0:
            break

        # Phase 1a: Independent cell solves
        for row in range(3):
            for col in range(3):
                if time.time() > deadline - 20.0:
                    break
                cell_idx = [i for i in range(N) if cell_assignment[i] == (row, col)]
                if not cell_idx:
                    continue
                x_min, x_max, y_min, y_max = cell_bounds(row, col)
                _solve_cell(cell_idx, x_min, x_max, y_min, y_max, None, deadline - 20.0)

        # Phase 1b: Boundary coordination - refine circles near cell boundaries
        # Identify "boundary circles": those within 0.15 of any cell boundary
        boundary_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            x, y = xy_g[i, 0], xy_g[i, 1]
            row, col = cell_assignment[i]
            x_min, x_max, y_min, y_max = cell_bounds(row, col)
            # Check if near any internal boundary (not the outer square boundary)
            if col > 0 and x < x_min + 0.12:
                boundary_mask[i] = True
            if col < 2 and x > x_max - 0.12:
                boundary_mask[i] = True
            if row > 0 and y < y_min + 0.12:
                boundary_mask[i] = True
            if row < 2 and y > y_max - 0.12:
                boundary_mask[i] = True

        if np.any(boundary_mask):
            bidx = np.where(boundary_mask)[0]
            # Optimize boundary circles jointly with a global penalty
            def boundary_objective(params):
                if time.time() > deadline:
                    return 1e6
                cx = params[0:len(bidx)]
                cy = params[len(bidx):2*len(bidx)]
                cr = params[2*len(bidx):3*len(bidx)]
                cr = np.maximum(cr, 1e-6)

                penalty = 0.0
                w = 800.0

                # Wall constraints (global square)
                for i in range(len(bidx)):
                    gi = bidx[i]
                    xi, yi, ri = cx[i], cy[i], cr[i]
                    penalty += max(0.0, ri - xi)**2
                    penalty += max(0.0, ri - (1.0 - xi))**2
                    penalty += max(0.0, ri - yi)**2
                    penalty += max(0.0, ri - (1.0 - yi))**2

                # Overlaps: boundary circles with each other and with all non-boundary circles
                all_idx = list(range(N))
                for i in range(len(bidx)):
                    gi = bidx[i]
                    for j in range(len(bidx)):
                        gj = bidx[j]
                        if gi >= gj:
                            continue
                        dx = cx[i] - cx[j]
                        dy = cy[i] - cy[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_d = cr[i] + cr[j]
                        if dist < min_d:
                            penalty += (min_d - dist)**2
                    # Overlap with non-boundary circles
                    for nj in range(N):
                        if boundary_mask[nj]:
                            continue
                        dx = cx[i] - xy_g[nj, 0]
                        dy = cy[i] - xy_g[nj, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_d = cr[i] + r_g[nj]
                        if dist < min_d:
                            penalty += (min_d - dist)**2

                return -np.sum(cr) + w * penalty

            params0 = np.concatenate([xy_g[bidx, 0], xy_g[bidx, 1], r_g[bidx]])
            bounds = []
            for i in range(len(bidx)):
                bounds.append((1e-4, 0.999))
                bounds.append((1e-4, 0.999))
                bounds.append((1e-4, 0.35))

            try:
                res = minimize(boundary_objective, params0, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 200, 'maxfun': 3000, 'ftol': 1e-12})
                if res.success or res.fun < boundary_objective(params0):
                    cx = res.x[0:len(bidx)]
                    cy = res.x[len(bidx):2*len(bidx)]
                    cr = np.maximum(res.x[2*len(bidx):3*len(bidx)], 1e-4)
                    xy_g[bidx, 0] = cx
                    xy_g[bidx, 1] = cy
                    r_g[bidx] = cr
            except Exception:
                pass

    # --- Phase 2: Global polish with vectorized gradient ---
    def _global_objective(params):
        """Minimize negative sum of radii + penalties for all overlaps and wall violations."""
        if time.time() > deadline:
            return 1e6
        n = len(params) // 3
        cx = params[0::3]
        cy = params[1::3]
        cr = params[2::3]
        cr = np.maximum(cr, 1e-6)

        penalty = 0.0
        w = 1000.0

        # Wall violations
        for i in range(n):
            xi, yi, ri = cx[i], cy[i], cr[i]
            penalty += max(0.0, ri - xi)**2
            penalty += max(0.0, ri - (1.0 - xi))**2
            penalty += max(0.0, ri - yi)**2
            penalty += max(0.0, ri - (1.0 - yi))**2

        # Overlap violations (vectorized)
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = cr[i] + cr[j]
                if dist < min_d:
                    penalty += (min_d - dist)**2

        return -np.sum(cr) + w * penalty

    params0 = np.zeros(3 * N)
    params0[0::3] = xy_g[:, 0]
    params0[1::3] = xy_g[:, 1]
    params0[2::3] = r_g

    bounds = []
    for i in range(N):
        bounds.append((1e-4, 0.999))
        bounds.append((1e-4, 0.999))
        bounds.append((1e-4, 0.4))

    best_params = params0.copy()
    best_total = 0.0
    best_valid = False

    # Check initial validity
    init_total, init_valid = _eval_packing(xy_g, r_g)
    if init_valid:
        best_total = init_total
        best_valid = True

    # Multiple restarts with different penalty weights
    for restart in range(4):
        if time.time() > deadline - 10.0:
            break

        pw = [1000.0, 2000.0, 500.0, 3000.0][restart]

        def _obj_pw(params, pw=pw):
            if time.time() > deadline:
                return 1e6
            n = len(params) // 3
            cx = params[0::3]
            cy = params[1::3]
            cr = params[2::3]
            cr = np.maximum(cr, 1e-6)

            penalty = 0.0
            w = pw

            for i in range(n):
                xi, yi, ri = cx[i], cy[i], cr[i]
                penalty += max(0.0, ri - xi)**2
                penalty += max(0.0, ri - (1.0 - xi))**2
                penalty += max(0.0, ri - yi)**2
                penalty += max(0.0, ri - (1.0 - yi))**2

            for i in range(n):
                for j in range(i+1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = cr[i] + cr[j]
                    if dist < min_d:
                        penalty += (min_d - dist)**2

            return -np.sum(cr) + w * penalty

        try:
            res = minimize(_obj_pw, best_params, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14})
            opt_params = res.x
            opt_xy = opt_params[0::3].reshape(-1, 2)
            opt_r = np.maximum(opt_params[1::3], 1e-4)

            total, valid = _eval_packing(opt_xy, opt_r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_params = opt_params.copy()
        except Exception:
            pass

    # --- Final high-precision polish ---
    if time.time() < deadline - 5.0 and best_valid:
        def _obj_high(params):
            if time.time() > deadline:
                return 1e6
            n = len(params) // 3
            cx = params[0::3]
            cy = params[1::3]
            cr = params[2::3]
            cr = np.maximum(cr, 1e-6)

            penalty = 0.0
            w = 5000.0

            for i in range(n):
                xi, yi, ri = cx[i], cy[i], cr[i]
                penalty += max(0.0, ri - xi)**2
                penalty += max(0.0, ri - (1.0 - xi))**2
                penalty += max(0.0, ri - yi)**2
                penalty += max(0.0, ri - (1.0 - yi))**2

            for i in range(n):
                for j in range(i+1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = cr[i] + cr[j]
                    if dist < min_d:
                        penalty += (min_d - dist)**2

            return -np.sum(cr) + w * penalty

        try:
            res = minimize(_obj_high, best_params, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-16, 'gtol': 1e-10})
            opt_params = res.x
            opt_xy = opt_params[0::3].reshape(-1, 2)
            opt_r = np.maximum(opt_params[1::3], 1e-4)

            total, valid = _eval_packing(opt_xy, opt_r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_params = opt_params.copy()
        except Exception:
            pass

    # --- Extract final result ---
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 1e-4)

    # Clip to ensure validity
    result = np.zeros((N, 3))
    for i in range(N):
        x, y, ri = final_xy[i, 0], final_xy[i, 1], final_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 1e-4)
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        result[i, 2] = ri

    # Final validation
    total, valid = _eval_packing(result[:, :2], result[:, 2])
    if not valid:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
