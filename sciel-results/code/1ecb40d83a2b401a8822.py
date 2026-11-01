"""Circle packing in perimeter-4 rectangle: Grid cellular decomposition + smooth optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

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


def _smooth_overlap_penalty(x, y, r, w, h, penalty_weight=5000.0, soft_scale=50.0):
    """Smooth pairwise overlap and boundary penalty using soft-min via log-sum-exp."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    
    # Soft overlap: use smooth approximation of max(0, r_sum - d)
    overlap_raw = r_sum_pairs - d_pairs
    # Soft-max via log-sum-exp for smoothness
    soft_overlap = np.logaddexp(0, soft_scale * overlap_raw) / soft_scale
    
    # Boundary violations
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h
    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    soft_boundary = np.logaddexp(0, soft_scale * all_viol) / soft_scale
    
    return penalty_weight * (np.sum(soft_overlap) + np.sum(soft_boundary))


def _objective(z, penalty_weight=5000.0, soft_scale=50.0):
    """Smooth objective for SLSQP."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.15 or h < 0.15 or np.any(r <= 0.001):
        return 1e8

    obj = -np.sum(r)
    obj += _smooth_overlap_penalty(x, y, r, w, h, penalty_weight, soft_scale)

    # Aspect ratio penalty
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _cell_inradius(x0, y0, x1, y1, cx, cy):
    """Compute the inradius of a rectangular cell for a circle centered at (cx, cy)."""
    return min(cx - x0, x1 - cx, cy - y0, y1 - cy)


def _generate_cellular_layout(w, h, rows, cols, stagger=False):
    """Generate a cellular decomposition layout with one circle per cell."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    cell_w = w / cols
    cell_h = h / rows
    
    for i in range(N):
        row = i // cols
        col = i % cols
        
        if row >= rows:
            # Handle remainder: place in last row
            row = rows - 1
            col = i - (rows - 1) * cols
            if col >= cols:
                col = cols - 1
        
        # Cell boundaries
        cx_cell = (col + 0.5) * cell_w
        cy_cell = (row + 0.5) * cell_h
        
        # Stagger offset for hexagonal-like packing
        if stagger and (row % 2 == 1):
            cx_cell += 0.25 * cell_w
        
        # Clamp to cell
        cx_cell = np.clip(cx_cell, x0 + cell_w * 0.15 if False else 0.15 * cell_w, 
                          (col + 0.85) * cell_w)
        cy_cell = np.clip(cy_cell, 0.15 * cell_h, (row + 0.85) * cell_h)
        
        x[i] = cx_cell
        y[i] = cy_cell
        
        # Inradius of the cell
        r[i] = _cell_inradius(col * cell_w, row * cell_h, 
                              (col + 1) * cell_w, (row + 1) * cell_h, 
                              cx_cell, cy_cell) * 0.95
    
    return x, y, r


def _force_relax(x, y, r, w, h, max_iter=600, dt_init=0.002):
    """Force-directed relaxation to resolve overlaps."""
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
            dt = min(0.004, dt * 1.1)

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    return x, y


def _high_precision_polish(x, y, r, w, h, time_budget):
    """High-precision polish: iterative radius growth with position adjustment."""
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Phase 1: Fine force relaxation
    dt = 0.0005
    max_iter = int(min(3000, time_budget * 80))
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

        if np.max(overlap) > 1e-11:
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

        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Phase 2: Binary search radius growth per circle
    for i in range(n):
        if time.time() > time_budget:
            break
        # Find max radius for circle i given others fixed
        r_lo = r[i]
        r_hi = min(x[i], w - x[i], y[i], h - y[i])
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_hi = min(r_hi, d - r[j])
        r_hi = max(r_hi, r_lo)
        
        if r_hi > r_lo + 1e-8:
            # Binary search
            for _ in range(30):
                if time.time() > time_budget:
                    break
                r_mid = (r_lo + r_hi) / 2
                r_test = r.copy()
                r_test[i] = r_mid
                if _is_valid(x, y, r_test, w, h):
                    r_lo = r_mid
                else:
                    r_hi = r_mid
            r[i] = r_lo * 0.999999

    if _is_valid(x, y, r, w, h):
        return x, y, r
    return None


def _expand_individual(x, y, r, w, h, time_budget):
    """Try to expand each circle individually by finding optimal position."""
    n = N
    improved = False
    for i in range(n):
        if time.time() > time_budget:
            break
        # Try small random perturbations to find better position
        best_r = r[i]
        best_x, best_y = x[i], y[i]
        for _ in range(50):
            if time.time() > time_budget:
                break
            dx = np.random.uniform(-0.05, 0.05)
            dy = np.random.uniform(-0.05, 0.05)
            tx = np.clip(x[i] + dx, 0.01, w - 0.01)
            ty = np.clip(y[i] + dy, 0.01, h - 0.01)
            
            r_max = min(tx, w - tx, ty, h - ty)
            for j in range(n):
                if i == j:
                    continue
                ddx = tx - x[j]
                ddy = ty - y[j]
                d = np.sqrt(ddx * ddx + ddy * ddy)
                r_max = min(r_max, d - r[j])
            r_max = max(0.0, r_max)
            
            if r_max > best_r + 1e-6:
                best_r = r_max
                best_x, best_y = tx, ty
        
        if best_r > r[i] + 1e-6:
            x[i] = best_x
            y[i] = best_y
            r[i] = best_r * 0.9999
            improved = True

    if _is_valid(x, y, r, w, h):
        return x, y, r, improved
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Cellular decomposition: try various aspect ratios and grid configurations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]

    for w0 in aspect_ratios:
        if time.time() > deadline - 25:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try multiple grid configurations
        for rows in range(3, 8):
            if time.time() > deadline - 25:
                break
            cols = int(np.ceil(N / rows))
            if cols < 3 or cols > 10:
                continue
            
            # Generate cellular layout
            for stagger in [False, True]:
                if time.time() > deadline - 25:
                    break
                x, y, r = _generate_cellular_layout(w0, h0, rows, cols, stagger)
                
                # Force relaxation
                x, y = _force_relax(x, y, r, w0, h0, max_iter=400)

                if not _is_valid(x, y, r, w0, h0):
                    r *= 0.92
                    if not _is_valid(x, y, r, w0, h0):
                        continue

                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w0, h0)

    # SLSQP optimization on best result
    if best_result is not None and time.time() < deadline - 35:
        x_best, y_best, r_best, w_best, h_best = best_result

        z = np.zeros(3 * N + 1)
        z[0::3][:N] = x_best
        z[1::3][:N] = y_best
        z[2::3][:N] = r_best
        z[-1] = w_best

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(150, remaining * 30)))
        try:
            res = minimize(_objective, z, method='SLSQP',
                           bounds=[(0.01, 2.0)] * (2 * N) + [(0.01, 1.0)] * N + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            z_opt = res.x
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt

            if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
                val = float(np.sum(r_opt))
                if val > best_value:
                    best_value = val
                    best_result = (x_opt.copy(), y_opt.copy(), r_opt.copy(), w_opt, h_opt)
            else:
                r_repair = r_opt * 0.95
                if _is_valid(x_opt, y_opt, r_repair, w_opt, h_opt):
                    val = float(np.sum(r_repair))
                    if val > best_value:
                        best_value = val
                        best_result = (x_opt.copy(), y_opt.copy(), r_repair.copy(), w_opt, h_opt)
        except Exception:
            pass

    # High-precision polish
    if best_result is not None and time.time() < deadline - 12:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 6
        polished = _high_precision_polish(x_best, y_best, r_best, w_best, h_best, remaining)
        if polished is not None:
            x_p, y_p, r_p = polished
            val = float(np.sum(r_p))
            if val > best_value:
                best_value = val
                best_result = (x_p, y_p, r_p, w_best, h_best)

    # Final individual expansion
    if best_result is not None and time.time() < deadline - 4:
        x_best, y_best, r_best, w_best, h_best = best_result
        remaining = deadline - (time.time() - start_time) - 2
        expanded = _expand_individual(x_best, y_best, r_best, w_best, h_best, remaining)
        if expanded is not None:
            x_e, y_e, r_e, _ = expanded
            val = float(np.sum(r_e))
            if val > best_value:
                best_value = val
                best_result = (x_e, y_e, r_e, w_best, h_best)

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result

    # Final safety: ensure strict validity
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
