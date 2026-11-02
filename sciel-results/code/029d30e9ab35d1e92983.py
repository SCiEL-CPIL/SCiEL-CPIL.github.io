"""Circle packing in perimeter-4 rectangle: SOCP-style exact constraint optimization via SLSQP."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _perimeter(w, h):
    return 2.0 * (w + h)


def _radius_sum(circles):
    return float(np.sum(circles[:, 2]))


def _is_valid(circles, tol=1e-9):
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    W = max_x - min_x
    H = max_y - min_y
    if W + H > 2.0 + tol:
        return False

    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            r_sum = rs[i] + rs[j]
            if dist_sq < r_sum * r_sum - tol:
                return False
    return True


def _is_valid_from_z(z, tol=1e-9):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r <= 0):
        return False

    for i in range(N):
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False

    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _radius_sum_from_z(z):
    r = z[2::3][:N]
    return float(np.sum(r))


def _exact_objective(z, penalty_weight=5000.0):
    """
    Objective for SLSQP: minimize -sum(r) + hard penalties for constraint violations.
    This mimics an SOCP formulation where constraints are exactly enforced.
    """
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise non-overlap: |p_i - p_j| >= r_i + r_j
    # Violation = max(0, (r_i + r_j)^2 - |p_i - p_j|^2)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum * r_sum

    iu = np.triu_indices(N, k=1)
    overlap_viol = np.maximum(r_sum_sq[iu] - d_sq[iu], 0.0)
    obj += penalty_weight * np.sum(overlap_viol)

    # Boundary constraints:
    # x_i >= r_i, x_i + r_i <= w, y_i >= r_i, y_i + r_i <= h
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_viol = left_viol + right_viol + bottom_viol + top_viol
    obj += penalty_weight * np.sum(boundary_viol ** 2)

    # Aspect ratio bounds (soft)
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _force_relax(z, max_iter=300, dt_init=0.003):
    """Force-directed relaxation with vectorized pairwise forces."""
    z = z.copy().astype(float)
    dt = dt_init
    for it in range(max_iter):
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(N)
        fy = np.zeros(N)

        # Vectorized repulsion between overlapping circles
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d_sq, 1e-12))
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(r_sum - d, 0.0)

        iu = np.triu_indices(N, k=1)
        dx_u = dx[iu]
        dy_u = dy[iu]
        d_u = d[iu]
        ov_u = overlap[iu]
        mask = d_u > 1e-12
        safe_d = np.where(mask, d_u, 1.0)
        force = ov_u * 10.0
        fx[iu[0]] += force * dx_u / safe_d
        fy[iu[0]] += force * dy_u / safe_d
        fx[iu[1]] -= force * dx_u / safe_d
        fy[iu[1]] -= force * dy_u / safe_d

        # Boundary forces (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx += 50.0 * (left_viol - right_viol)
        fy += 50.0 * (bottom_viol - top_viol)

        # Apply forces
        z[0::3][:N] += dt * fx
        z[1::3][:N] += dt * fy

        # Clamp
        for i in range(N):
            z[3 * i] = np.clip(z[3 * i], r[i] * 0.5, w - r[i] * 0.5)
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i] * 0.5, h - r[i] * 0.5)

        # Decay step
        dt *= 0.995
        if dt < 1e-6:
            break

    return z


def _grid_cell_layout(w, h, n_cols, n_rows, margin=0.95):
    """
    Grid-based cellular decomposition.
    """
    total_cells = n_cols * n_rows
    if total_cells < N:
        return None

    z = np.zeros(3 * N + 1)
    cell_w = w / n_cols
    cell_h = h / n_rows

    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            r = 0.5 * min(cell_w, cell_h) * margin
            z[3 * count] = cx
            z[3 * count + 1] = cy
            z[3 * count + 2] = r
            count += 1
    z[-1] = w
    return z


def _slsqp_refine(z0, time_budget=15.0, penalty_weight=5000.0):
    """SLSQP refinement with exact constraint penalties."""
    start = time.time()
    maxiter = int(min(800, max(200, time_budget * 60)))

    try:
        res = minimize(
            _exact_objective, z0, method='SLSQP',
            bounds=[(0.01, 2.0)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100}
        )
        if res.success or res.fun < _exact_objective(z0):
            return res.x
    except Exception:
        pass
    return z0


def _contact_graph_refine(z0, time_budget=10.0):
    """
    Refine using contact graph: identify touching pairs and solve linearized system.
    This is a structural alternative to SLP using exact contact constraints.
    """
    start = time.time()
    z = z0.copy().astype(float)
    
    for slp_iter in range(50):
        if time.time() - start > time_budget:
            break
        
        x = z[0::3][:N].copy()
        y = z[1::3][:N].copy()
        r = z[2::3][:N].copy()
        w = z[-1]
        h = 2.0 - w
        
        # Identify contact pairs
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
        r_sum = r[:, None] + r[None, :]
        
        tol = 0.01
        contact_mask = (np.abs(d - r_sum) < tol * np.maximum(r_sum, 0.01)) & (d > 1e-6)
        np.fill_diagonal(contact_mask, False)
        
        wall_left = np.abs(x - r) < tol * r
        wall_right = np.abs(x + r - w) < tol * r
        wall_bottom = np.abs(y - r) < tol * r
        wall_top = np.abs(y + r - h) < tol * r
        
        # Build linear system for small adjustments
        n_vars = 3 * N + 1
        c_obj = np.zeros(n_vars)
        c_obj[2::3][:N] = -1.0
        
        A_rows = []
        b_rows = []
        
        iu = np.triu_indices(N, k=1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            if contact_mask[i, j]:
                dxij = x[i] - x[j]
                dyij = y[i] - y[j]
                rij = r[i] + r[j]
                if rij > 1e-6:
                    row = np.zeros(n_vars)
                    row[3*i] = 2 * dxij
                    row[3*i + 1] = 2 * dyij
                    row[3*j] = -2 * dxij
                    row[3*j + 1] = -2 * dyij
                    row[3*i + 2] = -2 * rij
                    row[3*j + 2] = -2 * rij
                    A_rows.append(row)
                    b_rows.append(0.0)
        
        for i in range(N):
            if wall_left[i]:
                row = np.zeros(n_vars)
                row[3*i] = 1.0
                row[3*i + 2] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            if wall_right[i]:
                row = np.zeros(n_vars)
                row[3*i] = 1.0
                row[3*i + 2] = 1.0
                row[-1] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            if wall_bottom[i]:
                row = np.zeros(n_vars)
                row[3*i + 1] = 1.0
                row[3*i + 2] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
            if wall_top[i]:
                row = np.zeros(n_vars)
                row[3*i + 1] = 1.0
                row[3*i + 2] = 1.0
                row[-1] = 1.0
                A_rows.append(row)
                b_rows.append(0.0)
        
        if not A_rows:
            break
        
        A_arr = np.array(A_rows)
        b_arr = np.array(b_rows)
        A_ub = -A_arr
        b_ub = -b_arr
        bounds = [(-0.03, 0.03)] * (3 * N) + [(-0.03, 0.03)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                step = 0.7
                z[0::3][:N] += step * res.x[0::3][:N]
                z[1::3][:N] += step * res.x[1::3][:N]
                z[2::3][:N] += step * res.x[2::3][:N]
                z[-1] += step * res.x[-1]
                
                z[2::3][:N] = np.maximum(z[2::3][:N], 0.01)
                z[-1] = np.clip(z[-1], 0.3, 1.7)
                
                z = _force_relax(z, max_iter=30, dt_init=0.001)
                
                if not _is_valid_from_z(z):
                    z = z0.copy()
                    break
        except Exception:
            break
    
    return z


def _expansion_polish(z, time_budget=5.0):
    """
    Try to expand radii while maintaining validity.
    """
    z = z.copy().astype(float)
    start = time.time()
    best_z = z.copy()
    best_val = _radius_sum_from_z(z)

    for step in range(80):
        if time.time() - start > time_budget:
            break
        
        z_trial = z.copy()
        z_trial[2::3][:N] = z[2::3][:N] * 1.015

        if _is_valid_from_z(z_trial):
            z = z_trial
            val = _radius_sum_from_z(z)
            if val > best_val:
                best_val = val
                best_z = z.copy()
        else:
            z_relaxed = _force_relax(z_trial, max_iter=80, dt_init=0.002)
            if _is_valid_from_z(z_relaxed):
                z = z_relaxed
                val = _radius_sum_from_z(z)
                if val > best_val:
                    best_val = val
                    best_z = z.copy()
            else:
                break

    return best_z


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    best_z = None
    best_val = 0.0

    # --- Phase 1: Grid-cell decompositions with diverse aspect ratios ---
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 1.4, 0.7]

    for w0 in aspect_ratios:
        if time.time() > deadline - 40:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        for n_cols in range(3, 7):
            for n_rows in range(3, 7):
                if n_cols * n_rows < N:
                    continue
                z = _grid_cell_layout(w0, h0, n_cols, n_rows, margin=0.92)
                if z is None:
                    continue

                z_relaxed = _force_relax(z, max_iter=200, dt_init=0.003)

                if not _is_valid_from_z(z_relaxed):
                    continue

                val = _radius_sum_from_z(z_relaxed)
                if val > best_val:
                    best_val = val
                    best_z = z_relaxed.copy()

    # --- Phase 2: SLSQP refinement with exact constraints ---
    if best_z is not None:
        if time.time() < deadline - 30:
            refined = _slsqp_refine(best_z, time_budget=15.0, penalty_weight=5000.0)
            if _is_valid_from_z(refined):
                val = _radius_sum_from_z(refined)
                if val > best_val:
                    best_val = val
                    best_z = refined
            else:
                z_repair = refined.copy()
                z_repair[2::3][:N] *= 0.95
                if _is_valid_from_z(z_repair):
                    val = _radius_sum_from_z(z_repair)
                    if val > best_val:
                        best_val = val
                        best_z = z_repair

    # --- Phase 3: Contact graph refinement ---
    if best_z is not None and time.time() < deadline - 20:
        slp_result = _contact_graph_refine(best_z, time_budget=12.0)
        if _is_valid_from_z(slp_result):
            val = _radius_sum_from_z(slp_result)
            if val > best_val:
                best_val = val
                best_z = slp_result.copy()

    # --- Phase 4: Expansion polish ---
    if best_z is not None and time.time() < deadline - 12:
        expanded = _expansion_polish(best_z, time_budget=7.0)
        if _is_valid_from_z(expanded):
            val = _radius_sum_from_z(expanded)
            if val > best_val:
                best_val = val
                best_z = expanded

    # --- Phase 5: Second SLP pass after expansion ---
    if best_z is not None and time.time() < deadline - 8:
        slp_result2 = _contact_graph_refine(best_z, time_budget=5.0)
        if _is_valid_from_z(slp_result2):
            val = _radius_sum_from_z(slp_result2)
            if val > best_val:
                best_val = val
                best_z = slp_result2.copy()

    # --- Phase 6: Final SLSQP pass ---
    if best_z is not None and time.time() < deadline - 5:
        refined2 = _slsqp_refine(best_z, time_budget=4.0, penalty_weight=8000.0)
        if _is_valid_from_z(refined2):
            val = _radius_sum_from_z(refined2)
            if val > best_val:
                best_val = val
                best_z = refined2
        else:
            z_repair = refined2.copy()
            z_repair[2::3][:N] *= 0.96
            if _is_valid_from_z(z_repair):
                val = _radius_sum_from_z(z_repair)
                if val > best_val:
                    best_val = val
                    best_z = z_repair

    # --- Phase 7: Final force polish ---
    if best_z is not None and time.time() < deadline - 2:
        polished = _force_relax(best_z, max_iter=100, dt_init=0.001)
        if _is_valid_from_z(polished):
            val = _radius_sum_from_z(polished)
            if val > best_val:
                best_val = val
                best_z = polished

    # Fallback
    if best_z is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])

    # Format output
    x = best_z[0::3][:N]
    y = best_z[1::3][:N]
    r = best_z[2::3][:N]

    # Safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-9)
        circles_trial = np.column_stack((x, y, r_trial))
        if _is_valid(circles_trial):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
