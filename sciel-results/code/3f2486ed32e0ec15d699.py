"""Circle packing in perimeter-4 rectangle: grid-cell decomposition + force-directed + SLSQP refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE_MARGIN = 10.0


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


def _vectorized_objective(z, penalty_weight=500.0):
    """Vectorized objective: minimize -sum(r) + penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    r_pairs = r_sum[iu]
    overlap = np.maximum(r_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary penalties
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

    # Aspect ratio soft constraint
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=300, dt_init=0.003):
    """Force-directed relaxation to resolve overlaps and push circles into bounds."""
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

        # Repulsion between overlapping circles
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

        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 50.0
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 50.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 50.0
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 50.0

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
    Grid-based cellular decomposition: partition w x h rectangle into
    n_cols x n_rows cells, place one circle per cell with radius = inradius * margin.
    This guarantees disjoint interiors by construction.
    Returns z vector of shape (3N + 1,).
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
            # Inradius of the cell is min(cell_w, cell_h)/2; use margin for safety
            r = 0.5 * min(cell_w, cell_h) * margin
            z[3 * count] = cx
            z[3 * count + 1] = cy
            z[3 * count + 2] = r
            count += 1
    z[-1] = w
    return z


def _expansion_polish(z, time_budget=5.0):
    """
    Try to expand radii while maintaining validity.
    Uses iterative radius scaling with force relaxation.
    """
    z = z.copy().astype(float)
    start = time.time()
    scale = 1.0
    best_z = z.copy()
    best_val = _radius_sum_from_z(z)

    for step in range(50):
        if time.time() - start > time_budget:
            break
        # Try to increase all radii by a small factor
        scale *= 1.02
        z_trial = z.copy()
        z_trial[2::3][:N] = z[2::3][:N] * scale

        # Check if still valid
        if _is_valid_from_z(z_trial):
            z = z_trial
            val = _radius_sum_from_z(z)
            if val > best_val:
                best_val = val
                best_z = z.copy()
        else:
            # Relax to resolve overlaps
            z_relaxed = _force_relax(z_trial, max_iter=100, dt_init=0.002)
            if _is_valid_from_z(z_relaxed):
                z = z_relaxed
                val = _radius_sum_from_z(z)
                if val > best_val:
                    best_val = val
                    best_z = z.copy()
            else:
                break

    return best_z


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
    """SLSQP refinement on the unified parameter vector."""
    start = time.time()
    remaining = time_budget - (time.time() - start)
    maxiter = int(min(500, max(100, remaining * 40)))

    try:
        res = minimize(
            _vectorized_objective, z0, method='SLSQP',
            bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
            options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60}
        )
        if res.success or res.fun < _vectorized_objective(z0):
            return res.x
    except Exception:
        pass
    return z0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_z = None
    best_val = 0.0

    # --- Phase 1: Grid-cell decompositions with diverse aspect ratios ---
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 1.4, 0.7]

    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Try multiple grid decompositions
        for n_cols in range(3, 7):
            for n_rows in range(3, 7):
                if n_cols * n_rows < N:
                    continue
                z = _grid_cell_layout(w0, h0, n_cols, n_rows, margin=0.92)
                if z is None:
                    continue

                # Force relax to settle
                z_relaxed = _force_relax(z, max_iter=200, dt_init=0.003)

                # Check validity
                if not _is_valid_from_z(z_relaxed):
                    continue

                val = _radius_sum_from_z(z_relaxed)
                if val > best_val:
                    best_val = val
                    best_z = z_relaxed.copy()

    # --- Phase 2: SLSQP refinement on top candidates ---
    if best_z is not None:
        if time.time() < deadline - 25:
            refined = _slsqp_refine(best_z, time_budget=15.0)
            if _is_valid_from_z(refined):
                val = _radius_sum_from_z(refined)
                if val > best_val:
                    best_val = val
                    best_z = refined
            else:
                # Try shrink repair
                z_repair = refined.copy()
                z_repair[2::3][:N] *= 0.95
                if _is_valid_from_z(z_repair):
                    val = _radius_sum_from_z(z_repair)
                    if val > best_val:
                        best_val = val
                        best_z = z_repair

    # --- Phase 3: Expansion polish ---
    if best_z is not None and time.time() < deadline - 15:
        expanded = _expansion_polish(best_z, time_budget=8.0)
        if _is_valid_from_z(expanded):
            val = _radius_sum_from_z(expanded)
            if val > best_val:
                best_val = val
                best_z = expanded

    # --- Phase 4: Final SLSQP pass ---
    if best_z is not None and time.time() < deadline - 8:
        refined2 = _slsqp_refine(best_z, time_budget=5.0)
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

    # --- Phase 5: Final force polish ---
    if best_z is not None and time.time() < deadline - 3:
        polished = _force_relax(best_z, max_iter=100, dt_init=0.001)
        if _is_valid_from_z(polished):
            val = _radius_sum_from_z(polished)
            if val > best_val:
                best_val = val
                best_z = polished

    # Fallback
    if best_z is None:
        # Simple 5x5 grid in unit square
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
