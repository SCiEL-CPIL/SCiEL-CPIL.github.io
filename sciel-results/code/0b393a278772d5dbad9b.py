"""Circle packing in perimeter-4 rectangle: grid-cell + force-directed + SLSQP + constraint refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0


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


def _vectorized_objective(z, penalty_weight=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_pairs = r_sum[iu]
    overlap = np.maximum(r_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _granular_polish(z, time_budget=10.0):
    """Per-circle radial expansion with vectorized force relaxation."""
    z = z.copy().astype(float)
    start = time.time()
    best_z = z.copy()
    best_val = _radius_sum_from_z(z)
    
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w
    
    # Precompute pairwise distances
    dx_mat = x[:, None] - x[None, :]
    dy_mat = y[:, None] - y[None, :]
    d_mat = np.sqrt(dx_mat * dx_mat + dy_mat * dy_mat)
    np.fill_diagonal(d_mat, np.inf)
    
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        improved = False
        # Try expanding each circle individually
        for i in range(N):
            # Max radius for circle i given current positions
            max_r_i = min(x[i], w - x[i], y[i], h - y[i])
            for j in range(N):
                if i != j:
                    max_r_i = min(max_r_i, d_mat[i, j] - r[j])
            
            if max_r_i > r[i] * 1.001:
                # Try to expand circle i
                trial_r = min(max_r_i * 0.999, r[i] * 1.05)
                if trial_r > r[i]:
                    r_trial = r.copy()
                    r_trial[i] = trial_r
                    
                    # Check validity with force relaxation
                    z_trial = z.copy()
                    z_trial[2::3][:N] = r_trial
                    
                    z_relaxed = _force_relax(z_trial, max_iter=50, dt_init=0.002)
                    if _is_valid_from_z(z_relaxed):
                        val = _radius_sum_from_z(z_relaxed)
                        if val > best_val:
                            best_val = val
                            best_z = z_relaxed.copy()
                            z = z_relaxed.copy()
                            x = z[0::3][:N].copy()
                            y = z[1::3][:N].copy()
                            r = z[2::3][:N].copy()
                            # Recompute distance matrix
                            dx_mat = x[:, None] - x[None, :]
                            dy_mat = y[:, None] - y[None, :]
                            d_mat = np.sqrt(dx_mat * dx_mat + dy_mat * dy_mat)
                            np.fill_diagonal(d_mat, np.inf)
                            improved = True
                            break
        
        if not improved:
            break
    
    return best_z


def _force_relax(z, max_iter=300, dt_init=0.003):
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

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    if d < 1e-12:
                        dx = 0.01
                        dy = 0.01
                        d = np.sqrt(dx * dx + dy * dy)
                    force = (r_sum - d) * 10.0
                    fx[i] += force * dx / d
                    fy[i] += force * dy / d
                    fx[j] -= force * dx / d
                    fy[j] -= force * dy / d

        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 50.0
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 50.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 50.0
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 50.0

        z[0::3][:N] += dt * fx
        z[1::3][:N] += dt * fy

        for i in range(N):
            z[3 * i] = np.clip(z[3 * i], r[i] * 0.5, w - r[i] * 0.5)
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i] * 0.5, h - r[i] * 0.5)

        dt *= 0.995
        if dt < 1e-6:
            break

    return z


def _grid_cell_layout(w, h, n_cols, n_rows, margin=0.92, stagger=False):
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
            if stagger and row % 2 == 1:
                cx = (col + 0.25) * cell_w
            else:
                cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            r = 0.5 * min(cell_w, cell_h) * margin
            z[3 * count] = cx
            z[3 * count + 1] = cy
            z[3 * count + 2] = r
            count += 1
    z[-1] = w
    return z


def _radius_sum_from_z(z):
    r = z[2::3][:N]
    return float(np.sum(r))


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


def _slsqp_refine(z0, time_budget=15.0):
    start = time.time()
    remaining = time_budget - (time.time() - start)
    maxiter = int(min(600, max(150, remaining * 50)))

    try:
        res = minimize(
            _vectorized_objective, z0, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80}
        )
        if res.success or res.fun < _vectorized_objective(z0):
            return res.x
    except Exception:
        pass
    return z0


def _constraint_refine(z0, time_budget=10.0):
    """Fix positions, optimize radii and width to maximize sum(r) with non-overlap."""
    start = time.time()
    x0 = z0[0::3][:N].copy()
    y0 = z0[1::3][:N].copy()
    w0 = z0[-1]

    def obj_rc(rw):
        r = rw[:N]
        w = rw[N]
        h = 2.0 - w
        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        val = -np.sum(r)

        dx = x0[:, None] - x0[None, :]
        dy = y0[:, None] - y0[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
        r_pairs = r_sum[iu]
        overlap = np.maximum(r_pairs - d_pairs, 0.0)
        val += 1000.0 * np.sum(overlap * overlap)

        left_viol = np.maximum(r - x0, 0.0)
        right_viol = np.maximum(x0 + r - w, 0.0)
        bottom_viol = np.maximum(r - y0, 0.0)
        top_viol = np.maximum(y0 + r - h, 0.0)
        val += 1000.0 * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )
        return val

    r0 = z0[2::3][:N].copy()
    z_rc = np.concatenate([r0, [w0]])

    maxiter = int(min(400, max(100, time_budget * 40)))
    try:
        res = minimize(
            obj_rc, z_rc, method='SLSQP',
            bounds=[(0.01, 1.0)] * N + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60}
        )
        if res.fun < obj_rc(z_rc):
            r_new = res.x[:N]
            w_new = res.x[N]
            z_new = z0.copy()
            z_new[2::3][:N] = r_new
            z_new[-1] = w_new
            return z_new
    except Exception:
        pass
    return z0


def _expansion_polish(z, time_budget=8.0):
    z = z.copy().astype(float)
    start = time.time()
    scale = 1.0
    best_z = z.copy()
    best_val = _radius_sum_from_z(z)

    for step in range(100):
        if time.time() - start > time_budget:
            break
        scale *= 1.012
        z_trial = z.copy()
        z_trial[2::3][:N] = z[2::3][:N] * scale

        if _is_valid_from_z(z_trial):
            z = z_trial
            val = _radius_sum_from_z(z)
            if val > best_val:
                best_val = val
                best_z = z.copy()
        else:
            z_relaxed = _force_relax(z_trial, max_iter=100, dt_init=0.002)
            if _is_valid_from_z(z_relaxed):
                z = z_relaxed
                val = _radius_sum_from_z(z)
                if val > best_val:
                    best_val = val
                    best_z = z.copy()
            else:
                # Try smaller scale
                scale /= 1.012
                scale *= 1.005
                z_trial = z.copy()
                z_trial[2::3][:N] = z[2::3][:N] * scale
                if _is_valid_from_z(z_trial):
                    z = z_trial
                    val = _radius_sum_from_z(z)
                    if val > best_val:
                        best_val = val
                        best_z = z.copy()
                else:
                    break

    return best_z


def _high_precision_polish(z, time_budget=5.0):
    """Tight-tolerance SLSQP for final precision."""
    start = time.time()
    remaining = time_budget - (time.time() - start)
    maxiter = int(min(400, max(80, remaining * 40)))

    def obj_tight(zv):
        return _vectorized_objective(zv, penalty_weight=8000.0)

    try:
        res = minimize(
            obj_tight, z, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-15, 'maxls': 120}
        )
        if res.fun < obj_tight(z):
            return res.x
    except Exception:
        pass
    return z


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_z = None
    best_val = 0.0

    # --- Phase 1: Grid-cell decompositions with diverse aspect ratios and staggers ---
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 1.4, 0.7, 0.9, 1.1, 0.95, 1.05, 0.88, 1.12]

    for w0 in aspect_ratios:
        if time.time() > deadline - 50:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        for n_cols in range(3, 8):
            for n_rows in range(3, 8):
                if n_cols * n_rows < N:
                    continue
                for stagger in [False, True]:
                    if time.time() > deadline - 45:
                        break
                    for margin in [0.88, 0.92, 0.95]:
                        z = _grid_cell_layout(w0, h0, n_cols, n_rows, margin=margin, stagger=stagger)
                        if z is None:
                            continue

                        z_relaxed = _force_relax(z, max_iter=250, dt_init=0.003)

                        if not _is_valid_from_z(z_relaxed):
                            continue

                        val = _radius_sum_from_z(z_relaxed)
                        if val > best_val:
                            best_val = val
                            best_z = z_relaxed.copy()

    # --- Phase 2: SLSQP refinement on top candidate ---
    if best_z is not None:
        if time.time() < deadline - 35:
            refined = _slsqp_refine(best_z, time_budget=20.0)
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

    # --- Phase 3: Constraint-based refinement (fix positions, optimize radii+w) ---
    if best_z is not None and time.time() < deadline - 25:
        constrained = _constraint_refine(best_z, time_budget=10.0)
        if _is_valid_from_z(constrained):
            val = _radius_sum_from_z(constrained)
            if val > best_val:
                best_val = val
                best_z = constrained

    # --- Phase 4: Expansion polish ---
    if best_z is not None and time.time() < deadline - 20:
        expanded = _expansion_polish(best_z, time_budget=10.0)
        if _is_valid_from_z(expanded):
            val = _radius_sum_from_z(expanded)
            if val > best_val:
                best_val = val
                best_z = expanded
    
    # --- Phase 4.5: Granular polish ---
    if best_z is not None and time.time() < deadline - 12:
        granular = _granular_polish(best_z, time_budget=8.0)
        if _is_valid_from_z(granular):
            val = _radius_sum_from_z(granular)
            if val > best_val:
                best_val = val
                best_z = granular

    # --- Phase 5: Final SLSQP pass ---
    if best_z is not None and time.time() < deadline - 12:
        refined2 = _slsqp_refine(best_z, time_budget=8.0)
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

    # --- Phase 6: High-precision polish ---
    if best_z is not None and time.time() < deadline - 6:
        polished = _high_precision_polish(best_z, time_budget=5.0)
        if _is_valid_from_z(polished):
            val = _radius_sum_from_z(polished)
            if val > best_val:
                best_val = val
                best_z = polished

    # --- Phase 7: Final force polish ---
    if best_z is not None and time.time() < deadline - 3:
        final_polish = _force_relax(best_z, max_iter=80, dt_init=0.0005)
        if _is_valid_from_z(final_polish):
            val = _radius_sum_from_z(final_polish)
            if val > best_val:
                best_val = val
                best_z = final_polish
    
    # --- Phase 8: Final micro-expansion ---
    if best_z is not None and time.time() < deadline - 1:
        z_micro = best_z.copy()
        for _ in range(20):
            z_micro[2::3][:N] *= 1.002
            if _is_valid_from_z(z_micro):
                val = _radius_sum_from_z(z_micro)
                if val > best_val:
                    best_val = val
                    best_z = z_micro.copy()
            else:
                break

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
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        circles_trial = np.column_stack((x, y, r_trial))
        if _is_valid(circles_trial):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
