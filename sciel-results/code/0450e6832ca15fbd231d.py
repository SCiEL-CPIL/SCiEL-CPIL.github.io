"""Circle packing in perimeter-4 rectangle: Delaunay-based construction + optimization polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _force_relax(x, y, r, w, h, max_iter=800, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles to boundaries."""
    n = N
    dt = dt_init
    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 80
        fx += right_viol * 80
        fy -= bottom_viol * 80
        fy += top_viol * 80

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) < 1e-9:
            break

        f_pairs = overlap * 12.0
        dx_pairs = dx[iu]
        dy_pairs = dy[iu]
        ux = dx_pairs / d_pairs
        uy = dy_pairs / d_pairs
        for idx in range(len(iu[0])):
            i_idx = iu[0][idx]
            j_idx = iu[1][idx]
            fx[i_idx] += ux[idx] * f_pairs[idx]
            fy[i_idx] += uy[idx] * f_pairs[idx]
            fx[j_idx] -= ux[idx] * f_pairs[idx]
            fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = max(0.0001, dt * 0.7)
        elif max_force < 0.05:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _compute_inradii(x, y, w, h):
    """Compute maximum inscribed circle radius for each point using Delaunay triangulation."""
    n = N
    boundary_pts = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
        [w/2, 0], [w, h/2], [w/2, h], [0, h/2],
        [w/4, 0], [3*w/4, 0], [w, h/4], [w, 3*h/4],
        [3*w/4, h], [w/4, h], [0, 3*h/4], [0, h/4],
    ])
    all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])
    n_total = len(all_pts)

    try:
        tri = Delaunay(all_pts)
    except Exception:
        return np.full(n, min(w, h) / 6.0)

    inradii = np.zeros(n)
    for i in range(n):
        min_dist = min(x[i], w - x[i], y[i], h - y[i])

        for simplex in tri.simplices:
            if i in simplex:
                pts = all_pts[simplex]
                ax, ay = pts[0]
                bx, by = pts[1]
                cx, cy = pts[2]

                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

                dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                min_dist = min(min_dist, dist)

        inradii[i] = min_dist

    return inradii


def _delaunay_construction(w0, h0, max_iter=200, dt=0.002):
    """Construct initial packing using Delaunay-based inradius optimization."""
    n = N

    # Start with a jittered grid
    rows = 4
    cols = 6
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)

    for i in range(n):
        row = i // cols
        col = i % cols
        x[i] = (col + 0.5) * (w0 / cols) + np.random.uniform(-0.1, 0.1) * w0 / cols
        y[i] = (row + 0.5) * (h0 / rows) + np.random.uniform(-0.1, 0.1) * h0 / rows
        x[i] = np.clip(x[i], 0.05, w0 - 0.05)
        y[i] = np.clip(y[i], 0.05, h0 - 0.05)
        r[i] = 0.05

    # Iterate: compute inradii, then move points toward their Voronoi cell centers
    for it in range(max_iter):
        inradii = _compute_inradii(x, y, w0, h0)
        r = np.maximum(inradii, 0.01)

        boundary_pts = np.array([
            [0, 0], [w0, 0], [w0, h0], [0, h0],
            [w0/2, 0], [w0, h0/2], [w0/2, h0], [0, h0/2],
        ])
        all_pts = np.vstack([np.column_stack([x, y]), boundary_pts])

        try:
            tri = Delaunay(all_pts)
        except Exception:
            break

        new_x = x.copy()
        new_y = y.copy()

        for i in range(n):
            cx_sum = 0.0
            cy_sum = 0.0
            weight_sum = 0.0

            for simplex in tri.simplices:
                if i in simplex:
                    pts = all_pts[simplex]
                    ax, ay = pts[0]
                    bx, by = pts[1]
                    cx, cy = pts[2]

                    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-12:
                        continue
                    ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
                    uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

                    dist = np.sqrt((x[i] - ux)**2 + (y[i] - uy)**2)
                    weight = 1.0 / max(dist, 0.01)
                    cx_sum += ux * weight
                    cy_sum += uy * weight
                    weight_sum += weight

            if weight_sum > 0:
                new_x[i] = cx_sum / weight_sum
                new_y[i] = cy_sum / weight_sum

        x = x + dt * (new_x - x)
        y = y + dt * (new_y - y)
        x = np.clip(x, 0.05, w0 - 0.05)
        y = np.clip(y, 0.05, h0 - 0.05)

    r = _compute_inradii(x, y, w0, h0)
    x, y = _force_relax(x, y, r, w0, h0, max_iter=300)

    if _is_valid(x, y, r, w0, h0):
        return x, y, r
    else:
        r *= 0.9
        if _is_valid(x, y, r, w0, h0):
            return x, y, r
    return None


def _smooth_objective(z, beta=500.0):
    """Smooth objective using log-sum-exp for constraint violations."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.1 or h < 0.1 or np.any(r <= 0):
        return 1e8

    obj = -np.sum(r)

    # Smooth penalty for circle-circle overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    overlap = r_sum[iu] - d_pairs
    # log-sum-exp soft max
    max_ov = np.max(overlap)
    if max_ov > 0:
        lse = max_ov + np.log(np.sum(np.exp(overlap - max_ov))) / beta
        obj += beta * lse
    else:
        obj += 0.0

    # Smooth penalty for wall violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    max_v = np.max(all_viol)
    if max_v > 0:
        lse_v = max_v + np.log(np.sum(np.exp(all_viol - max_v))) / beta
        obj += beta * lse_v

    # Aspect ratio bounds (soft)
    if w < 0.3:
        obj += 1000.0 * (0.3 - w) ** 2
    elif w > 1.7:
        obj += 1000.0 * (w - 1.7) ** 2

    return obj


def _slsqp_optimize(x, y, r, w, h, time_budget):
    """SLSQP optimization with smooth log-sum-exp objective."""
    n = N
    z = np.zeros(3 * n + 1)
    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    z[-1] = w

    bounds = [(0.01, 2.0)] * (2 * n) + [(0.01, 1.0)] * n + [(0.3, 1.7)]
    remaining = time_budget - time.time()
    if remaining < 5:
        return None

    best_z = z.copy()
    best_val = float(np.sum(r))

    # Multi-stage: start with low beta, increase for precision
    for beta in [100.0, 300.0, 800.0]:
        remaining = time_budget - time.time()
        if remaining < 3:
            break
        maxiter = int(min(300, max(50, remaining * 20)))

        try:
            res = minimize(_smooth_objective, z, method='SLSQP',
                           bounds=bounds,
                           args=(beta,),
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            z_opt = res.x
            x_opt = z_opt[0::3][:n]
            y_opt = z_opt[1::3][:n]
            r_opt = z_opt[2::3][:n]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt

            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                val = float(np.sum(r_opt))
                if val > best_val:
                    best_val = val
                    best_z = z_opt.copy()
                    z = z_opt.copy()
                else:
                    # Try slight radius shrink
                    r_repair = r_opt * 0.995
                    if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                        val2 = float(np.sum(r_repair))
                        if val2 > best_val:
                            best_val = val2
                            z_new = z_opt.copy()
                            z_new[2::3][:n] = r_repair
                            best_z = z_new.copy()
                            z = z_new.copy()
            else:
                # Try shrinking radii to recover validity
                for scale in [0.99, 0.98, 0.97, 0.95]:
                    r_try = r_opt * scale
                    if _is_valid(x_opt, y_opt, r_try, w_opt, h_opt):
                        val = float(np.sum(r_try))
                        if val > best_val:
                            best_val = val
                            z_new = z_opt.copy()
                            z_new[2::3][:n] = r_try
                            best_z = z_new.copy()
                            z = z_new.copy()
                        break
        except Exception:
            pass

    if best_val > float(np.sum(r)):
        z_final = best_z
        x_f = z_final[0::3][:n]
        y_f = z_final[1::3][:n]
        r_f = z_final[2::3][:n]
        w_f = z_final[-1]
        h_f = 2.0 - w_f
        if _is_valid(x_f, y_f, r_f, w_f, h_f):
            return x_f, y_f, r_f, w_f, h_f

    return None


def _polish_positions(x, y, r, w, h, time_budget):
    """Polish positions via force-directed method with adaptive step, then grow radii."""
    n = N
    x = x.copy()
    y = y.copy()

    max_iter = int(min(3000, time_budget * 150))
    dt = 0.002
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 60
        fx += right_viol * 60
        fy -= bottom_viol * 60
        fy += top_viol * 60

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]
        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        max_overlap = np.max(overlap)
        if max_overlap > 1e-10:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs
            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        # Adaptive step size
        max_f = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_f > 1.0:
            dt = max(0.0005, dt * 0.7)
        elif max_f < 0.02:
            dt = min(0.005, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Aggressive radius growth with binary search per circle
    for _ in range(10):
        if time.time() > time_budget:
            break
        r_trial = r * 1.002
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Per-circle expansion
    for i in range(n):
        if time.time() > time_budget:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            r_max = min(r_max, r_allowed)
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-7:
            r[i] = r_max

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _expand_individual(x, y, r, w, h, time_budget):
    """Aggressive per-circle radius expansion with multiple passes."""
    n = N
    improved = False
    for _pass in range(5):
        if time.time() > time_budget:
            break
        any_improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                r_max = min(r_max, r_allowed)
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-6:
                r[i] = r_max
                any_improved = True
                improved = True
        if not any_improved:
            break

    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None


def _hexagonal_initial(w0, h0):
    """Generate hexagonal lattice initial guess."""
    n = N
    z = np.zeros(3 * n + 1)
    rows_needed = int(np.ceil(N / 6)) + 1
    col_spacing = w0 / 6.0
    row_spacing = col_spacing * np.sqrt(3) / 2.0
    idx = 0
    for row in range(rows_needed):
        if idx >= N:
            break
        y_pos = (row + 0.5) * row_spacing
        if y_pos > h0 - 0.05:
            break
        offset = (col_spacing / 2.0) if (row % 2 == 1) else 0.0
        for col in range(6):
            if idx >= N:
                break
            x_pos = (col + 0.5) * col_spacing + offset
            if x_pos < 0.05 or x_pos > w0 - 0.05:
                continue
            r0 = min(col_spacing, row_spacing) * 0.48
            z[3 * idx] = x_pos
            z[3 * idx + 1] = y_pos
            z[3 * idx + 2] = r0
            idx += 1
    if idx < N:
        for i in range(idx, N):
            z[3 * i] = w0 / 2.0
            z[3 * i + 1] = h0 / 2.0
            z[3 * i + 2] = 0.05
    z[-1] = w0
    return z


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    # Phase 1: Delaunay-based construction for various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]

    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try Delaunay construction
        result = _delaunay_construction(w0, h0, max_iter=100)
        if result is not None:
            x, y, r = result
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

        # Also try grid construction as fallback
        best_grid = None
        best_grid_r = 0
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            cell_w = w0 / cols
            cell_h = h0 / rows
            r_grid = min(cell_w, cell_h) * 0.48
            if r_grid > best_grid_r:
                best_grid_r = r_grid
                best_grid = (rows, cols)

        if best_grid is not None:
            rows, cols = best_grid
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)

            for i in range(N):
                row = i // cols
                col = i % cols
                x[i] = (col + 0.5) * (w0 / cols)
                y[i] = (row + 0.5) * (h0 / rows)
                r[i] = min(w0 / cols, h0 / rows) * 0.45

            x, y = _force_relax(x, y, r, w0, h0, max_iter=300)

            if not _is_valid(x, y, r, w0, h0):
                r *= 0.95
                if not _is_valid(x, y, r, w0, h0):
                    continue

            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # Phase 2: Optimization
    if best_result is not None and time.time() < deadline - 30:
        x_best, y_best, r_best, w_best, h_best = best_result
        opt_result = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline)
        if opt_result is not None:
            x_opt, y_opt, r_opt, w_opt, h_opt = opt_result
            val = float(np.sum(r_opt))
            if val > best_value:
                best_value = val
                best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)

    # Phase 3: Polish
    if best_result is not None and time.time() < deadline - 10:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 5
        polished = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Phase 4: Final expansion
    if best_result is not None and time.time() < deadline - 3:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 1
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    # Phase 5: Second SLSQP pass on best result if time allows
    if best_result is not None and time.time() < deadline - 15:
        x_best, y_best, r_best, w_best, h_best = best_result
        opt_result2 = _slsqp_optimize(x_best, y_best, r_best, w_best, h_best, deadline - 5)
        if opt_result2 is not None:
            x_o2, y_o2, r_o2, w_o2, h_o2 = opt_result2
            val2 = float(np.sum(r_o2))
            if val2 > best_value:
                best_value = val2
                best_result = (x_o2.copy(), y_o2.copy(), r_o2.copy(), w_o2, h_o2)

    # Phase 6: Final polish after second optimization
    if best_result is not None and time.time() < deadline - 5:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining2 = deadline - (time.time() - start_time) - 2
        polished2 = _polish_positions(x_best, y_best, r_best, w_best, h_best, remaining2)
        if polished2 is not None:
            x_p2, y_p2, r_p2 = polished2
            val3 = float(np.sum(r_p2))
            if val3 > best_value:
                best_value = val3
                best_result = (x_p2, y_p2, r_p2, w_best, h_best)

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
