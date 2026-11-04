"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# Deadline guard: 142s wall-clock (8s safety margin under 150s)
_DEADLINE = 142.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _soft_objective(x, n=N, mu=5000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)

    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Boundary penalty gradient
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


def _is_valid(x, n=N, tol=1e-7):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _project_feasible(x, n=N, max_iter=1000, tol=1e-12):
    """
    Project a configuration to feasibility by iteratively resolving violations.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)

    for _ in range(max_iter):
        if _remaining() < 1.0:
            break
        xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()

        # Fix boundary violations
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        too_big = rs > min_d
        if np.any(too_big):
            rs[too_big] = min_d[too_big] * (1 - 1e-10)
            pts[:, 2] = rs

        # Fix overlap violations
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)

        if not np.any(ov > tol):
            break

        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)

        reduction = overlap_per_circle / 2.0 + tol
        rs = np.maximum(0.005, rs - reduction)
        pts[:, 2] = rs

    return pts.ravel()


def _grow_radii(x, n=N, max_iter=800, tol=1e-9):
    """
    Try to grow radii while maintaining feasibility using a local search approach.
    Vectorized per-circle max radius computation.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)

    for _ in range(max_iter):
        if _remaining() < 2.0:
            break

        xs = pts[:, 0].copy()
        ys = pts[:, 1].copy()
        rs = pts[:, 2].copy()

        improved = False

        # Compute max possible radius for each circle (vectorized)
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        max_r_boundary = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))

        # For each circle i, find min over j != i of (dist(i,j) - r_j)
        # Build pairwise distance matrix
        dx_mat = xs[:, None] - xs[None, :]
        dy_mat = ys[:, None] - ys[None, :]
        dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
        np.fill_diagonal(dist_mat, np.inf)

        # max_r_overlap[i] = min_j (dist_mat[i,j] - rs[j])
        # dist_mat[i,j] - rs[j] for all j
        constraint_mat = dist_mat - rs[None, :]  # shape (n, n), constraint_mat[i,j] = dist(i,j) - r_j
        max_r_overlap = np.min(constraint_mat, axis=1)

        # Combine with boundary
        max_r = np.minimum(max_r_boundary, max_r_overlap)
        max_r = np.maximum(max_r, 0.001)

        # Find circles that can grow
        growable = max_r > rs + 1e-12
        if not np.any(growable):
            break

        # Try to grow all growable circles at once
        # But we need to be careful: growing one circle may constrain others
        # Do it greedily: grow the most constrained ones first (smallest max_r improvement)
        order = np.argsort(max_r[growable] - rs[growable])
        grow_indices = np.where(growable)[0][order]

        for i in grow_indices:
            if _remaining() < 2.0:
                break
            target_r = max_r[i] * (1 - 1e-10)
            old_r = rs[i]
            rs[i] = target_r
            pts[i, 2] = target_r

            # Quick validity check
            if not _is_valid(pts.ravel(), n, tol=tol):
                rs[i] = old_r
                pts[i, 2] = old_r
            else:
                improved = True
                # Recompute max_r for remaining circles
                d_left = xs
                d_bot = ys
                d_right = 1.0 - xs
                d_top = 1.0 - ys
                max_r_boundary = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
                dx_mat = xs[:, None] - xs[None, :]
                dy_mat = ys[:, None] - ys[None, :]
                dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
                np.fill_diagonal(dist_mat, np.inf)
                constraint_mat = dist_mat - rs[None, :]
                max_r_overlap = np.min(constraint_mat, axis=1)
                max_r = np.minimum(max_r_boundary, max_r_overlap)
                max_r = np.maximum(max_r, 0.001)

        if not improved:
            break

    return pts.ravel()


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est

    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * x_spacing
            if i % 2 == 1:
                x += x_spacing / 2
            xs.append(x)
            ys.append(y)
            rs.append(r_est)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    rs_arr = np.array(rs)

    min_x = np.min(xs_arr - rs_arr)
    max_x = np.max(xs_arr + rs_arr)
    min_y = np.min(ys_arr - rs_arr)
    max_y = np.max(ys_arr + rs_arr)

    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)

    xs_arr = (xs_arr - min_x) * scale
    ys_arr = (ys_arr - min_y) * scale
    rs_arr = rs_arr * scale

    xs_arr += (1.0 - width * scale) / 2
    ys_arr += (1.0 - height * scale) / 2

    x = np.zeros(3 * n)
    x[0::3] = xs_arr
    x[1::3] = ys_arr
    x[2::3] = rs_arr

    return x


def _structured_init(n=N, pattern='spiral'):
    """Structured initializations."""
    xs, ys, rs = [], [], []

    if pattern == 'spiral':
        cx, cy = 0.5, 0.5
        r_base = 0.08
        for i in range(n):
            angle = i * 2.4
            dist = 0.03 * np.sqrt(i + 1)
            x = cx + dist * np.cos(angle)
            y = cy + dist * np.sin(angle)
            xs.append(x)
            ys.append(y)
            rs.append(r_base)
    elif pattern == 'diagonal':
        side = int(np.ceil(np.sqrt(n)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                x = (i + 0.5) / side + 0.02 * (j - side / 2)
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1
    else:
        side = int(np.ceil(np.sqrt(n)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                xs.append(x)
                ys.append(y)
                rs.append(r)
                count += 1

    x = np.zeros(3 * n)
    x[0::3] = xs[:n]
    x[1::3] = ys[:n]
    x[2::3] = rs[:n]
    return x


def _random_init(n=N, seed=None):
    """Random initial guess."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, 0.9, n)
    ys = rng.uniform(0.1, 0.9, n)
    rs = rng.uniform(0.05, 0.12, n)
    x = np.zeros(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def _grid_init(n=N):
    """Grid initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    x = np.zeros(3 * n)
    x[0::3] = xs[:n]
    x[1::3] = ys[:n]
    x[2::3] = rs[:n]
    return x


def _repulsion_relax_init(n=N, seed=42):
    """
    Initialize with repulsion relaxation: start with random positions,
    iteratively push circles apart and grow radii to find a good initial packing.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.15, 0.85, n)
    ys = rng.uniform(0.15, 0.85, n)
    rs = np.full(n, 0.04)

    i_idx, j_idx = np.triu_indices(n, k=1)

    for it in range(300):
        if _remaining() < 3.0:
            break

        # Repulsion: push overlapping circles apart
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)

        if np.any(ov > 1e-12):
            mask = ov > 1e-12
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            ov_m = ov[mask]
            dist_m = dist[mask]
            safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
            ux = dx[mask] / safe_dist
            uy = dy[mask] / safe_dist

            # Move centers apart
            push = ov_m * 0.5 * 0.8  # damped
            np.add.at(xs, i_m, -push * ux)
            np.add.at(ys, i_m, -push * uy)
            np.add.at(xs, j_m, push * ux)
            np.add.at(ys, j_m, push * uy)

        # Keep inside bounds
        xs = np.clip(xs, 0.05, 0.95)
        ys = np.clip(ys, 0.05, 0.95)

        # Try to grow radii
        for i in range(n):
            d_left = xs[i]
            d_bot = ys[i]
            d_right = 1.0 - xs[i]
            d_top = 1.0 - ys[i]
            max_r_b = min(d_left, d_bot, d_right, d_top)

            max_r_o = max_r_b
            for j in range(n):
                if i == j:
                    continue
                dx_ij = xs[i] - xs[j]
                dy_ij = ys[i] - ys[j]
                dist_ij = np.sqrt(dx_ij ** 2 + dy_ij ** 2)
                limit = dist_ij - rs[j]
                if limit < max_r_o:
                    max_r_o = limit

            target_r = max_r_o * 0.999
            if target_r > rs[i]:
                rs[i] = target_r

    x = np.zeros(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def _optimize_from(x0, n=N, maxiter=500, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        res = minimize(
            _soft_objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0


def _polish(x, n=N):
    """High-precision polish pass with multiple stages."""
    if _remaining() < 5.0:
        return x

    best_x = x.copy()
    best_score = _score(x, n)

    # Stage 1: Moderate mu optimization
    for mu in [5000.0, 10000.0, 20000.0, 50000.0]:
        if _remaining() < 4.0:
            break
        try:
            res = minimize(
                lambda v, m=mu: _soft_objective(v, n, m),
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            x_opt = res.x

            if not _is_valid(x_opt, n, tol=1e-8):
                x_proj = _project_feasible(x_opt, n)
                if _is_valid(x_proj, n, tol=1e-8):
                    x_opt = x_proj

            score = _score(x_opt, n)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            pass

    # Stage 2: Grow radii
    if _remaining() > 4.0:
        try:
            x_grown = _grow_radii(best_x, n, max_iter=500)
            if _is_valid(x_grown, n, tol=1e-8):
                score = _score(x_grown, n)
                if score > best_score:
                    best_score = score
                    best_x = x_grown.copy()
        except Exception:
            pass

    # Stage 3: Another round of optimization after growing
    if _remaining() > 4.0:
        for mu in [50000.0, 100000.0]:
            if _remaining() < 3.0:
                break
            try:
                res = minimize(
                    lambda v, m=mu: _soft_objective(v, n, m),
                    best_x,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100}
                )
                x_opt = res.x

                if not _is_valid(x_opt, n, tol=1e-8):
                    x_proj = _project_feasible(x_opt, n)
                    if _is_valid(x_proj, n, tol=1e-8):
                        x_opt = x_proj

                score = _score(x_opt, n)
                if score > best_score:
                    best_score = score
                    best_x = x_opt.copy()
            except Exception:
                pass

    # Stage 4: Final grow radii
    if _remaining() > 3.0:
        try:
            x_grown = _grow_radii(best_x, n, max_iter=300)
            if _is_valid(x_grown, n, tol=1e-8):
                score = _score(x_grown, n)
                if score > best_score:
                    best_score = score
                    best_x = x_grown.copy()
        except Exception:
            pass

    return best_x


def construct():
    global _START
    _START = time.time()

    best_x = None
    best_score = -np.inf

    # Generate diverse initializations
    inits = []

    # Hex packing
    inits.append(_hex_packing_init(N))

    # Structured inits
    inits.append(_structured_init(N, 'spiral'))
    inits.append(_structured_init(N, 'diagonal'))
    inits.append(_grid_init(N))

    # Repulsion relaxation init
    inits.append(_repulsion_relax_init(N, seed=42))

    # Random inits
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        inits.append(_random_init(N, seed=seed))

    for init_x in inits:
        if _remaining() < 12.0:
            break

        # Optimize
        x_opt = _optimize_from(init_x, N, maxiter=600, ftol=1e-14, gtol=1e-10)

        # Project to feasibility if needed
        if not _is_valid(x_opt, N, tol=1e-8):
            x_proj = _project_feasible(x_opt, N)
            if _is_valid(x_proj, N, tol=1e-8):
                x_opt = x_proj

        score = _score(x_opt, N)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    # Polish the best solution
    if best_x is not None and _remaining() > 10.0:
        best_x = _polish(best_x, N)
        best_score = _score(best_x, N)

    # If still no valid solution, fall back to grid
    if best_x is None or not _is_valid(best_x, N):
        best_x = _grid_init(N)

    return best_x.reshape(N, 3)
