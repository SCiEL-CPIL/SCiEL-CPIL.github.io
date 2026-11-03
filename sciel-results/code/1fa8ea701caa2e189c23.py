import numpy as np
import time
import math

N = 200

# Global deadline guard: evaluator kills at 150s; keep a safety margin.
_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Vectorized packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Boundary constraint: distance from each centre to the nearest side.
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    # Pairwise distances via squared distances (cheaper, no sqrt on the full matrix).
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _generate_hex_candidates(spacing):
    """Dense hexagonal lattice of candidate centres inside [-0.5,0.5]^2."""
    dy = spacing * math.sqrt(3.0) / 2.0
    max_coord = 0.5
    points = []
    row = 0
    y = -max_coord
    while y <= max_coord:
        x_offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        x = -max_coord + x_offset
        while x <= max_coord:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append((x, y))
            x += spacing
        y += dy
        row += 1
    return np.array(points, dtype=float) if points else np.zeros((0, 2))


def _greedy_select(candidates, n_select):
    """
    Greedy best-fit: repeatedly pick the candidate that maximises the minimum
    distance to (already-selected points, boundary). Returns (selected, score).
    """
    m = len(candidates)
    if m == 0:
        return np.zeros((0, 2)), 0.0
    if m <= n_select:
        sel = candidates.copy()
        return sel, _compute_radius(sel)

    # Per-candidate boundary distance (distance to nearest side).
    boundary = np.min(0.5 - np.abs(candidates), axis=1)
    # score[i] = min(boundary_i, min distance to selected so far).
    score = boundary.copy()
    sel_idx = []

    for _ in range(n_select):
        if _time_left() < 5.0:
            break
        best = int(np.argmax(score))
        sel_idx.append(best)
        p = candidates[best]
        # Distance from every candidate to the newly selected point.
        diff = candidates - p
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[best] = np.inf
        # New score is the tighter of old score and distance to new point.
        score = np.minimum(score, d)

    sel = candidates[sel_idx]
    return sel, _compute_radius(sel)


def _local_refine(pos, start_budget):
    """
    Iterative local refinement: repeatedly try to move the 'worst' point
    (the one limiting the radius) to a nearby better position.
    """
    pos = np.array(pos, dtype=float, copy=True)
    cur_r = _compute_radius(pos)
    rng = np.random.RandomState(7)
    step = 0.006

    while _time_left() > 8.0:
        improved = False
        # Find the limiting point: the one whose boundary or nearest-neighbour
        # distance is closest to the current radius.
        n = len(pos)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        nn_dist = np.sqrt(np.min(dist_sq, axis=1))
        boundary = np.min(0.5 - np.abs(pos), axis=1)
        # A point limits the radius if min(nn_dist/2, boundary) is near cur_r.
        limit = np.minimum(nn_dist * 0.5, boundary)
        worst = int(np.argmin(limit))

        # Try several candidate moves for the worst point.
        best_move = None
        best_r = cur_r
        for _ in range(40):
            if _time_left() < 8.0:
                break
            delta = rng.randn(2) * step
            new_pos = pos.copy()
            new_pos[worst] = pos[worst] + delta
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            r = _compute_radius(new_pos)
            if r > best_r + 1e-12:
                best_r = r
                best_move = new_pos
        if best_move is not None:
            pos = best_move
            cur_r = best_r
            improved = True
            step = min(step * 1.05, 0.02)
        else:
            step *= 0.7
            if step < 1e-6:
                break
            # Occasionally try a broader random kick to escape local minima.
            if rng.rand() < 0.1:
                k = rng.randint(n)
                new_pos = pos.copy()
                new_pos[k] = pos[k] + rng.randn(2) * 0.02
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                r = _compute_radius(new_pos)
                if r > cur_r:
                    pos = new_pos
                    cur_r = r
                    improved = True
    return pos, cur_r


def _dynamics_optimize(pts, time_limit=50.0):
    """
    Optimize packing using repulsive force dynamics.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)

    # Initial radius estimate
    r_curr = _compute_radius(pts)

    # Parameters
    damping = 0.9
    step_size = 0.005
    force_k = 1.0

    # We want to maximize the minimum distance.
    # We apply forces that push particles away from each other and walls.

    for step in range(10000):
        if time.time() - start_time > time_limit:
            break

        # 1. Compute Forces
        forces = np.zeros_like(pts)

        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))

        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)

        inv_dist_sq = 1.0 / (dists * dists)

        inv_dist_cubed = inv_dist_sq / dists
        np.fill_diagonal(inv_dist_cubed, 0.0)

        sum_inv_dist_cubed = np.sum(inv_dist_cubed, axis=1)
        weighted_pts = inv_dist_cubed @ pts

        force_pair = force_k * (pts * sum_inv_dist_cubed[:, np.newaxis] - weighted_pts)

        # Wall repulsion
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])

        wall_force_x = np.zeros(n)
        wall_force_y = np.zeros(n)

        mask_x = dx < r_curr
        k_wall = 5.0
        wall_force_x[mask_x] = -k_wall * (r_curr - dx[mask_x]) * np.sign(pts[mask_x, 0])

        mask_y = dy < r_curr
        wall_force_y[mask_y] = -k_wall * (r_curr - dy[mask_y]) * np.sign(pts[mask_y, 1])

        total_forces = force_pair
        total_forces[:, 0] += wall_force_x
        total_forces[:, 1] += wall_force_y

        # 2. Update Positions
        if not hasattr(_dynamics_optimize, 'momentum'):
            _dynamics_optimize.momentum = np.zeros_like(pts)

        momentum = _dynamics_optimize.momentum
        momentum = damping * momentum + step_size * total_forces
        pts += momentum

        # 3. Constraint Projection
        pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)

        # 4. Update Radius
        r_new = _compute_radius(pts)
        if r_new > r_curr:
            r_curr = r_new
            step_size *= 0.999
        else:
            pass

        # Cool down step size over time
        step_size *= 0.9995

    _dynamics_optimize.momentum = np.zeros_like(pts)
    return pts


def construct():
    best_pos = None
    best_r = 0.0

    # 1) Greedy selection from a dense hexagonal lattice.
    spacings = [0.04, 0.03, 0.025]
    for sp in spacings:
        if _time_left() < 15.0:
            break
        try:
            candidates = _generate_hex_candidates(sp)
            if len(candidates) < N:
                continue
            sel, r = _greedy_select(candidates, N)
            if len(sel) >= N and r > best_r:
                best_r = r
                best_pos = sel.copy()
        except Exception:
            continue

    # Fallback: coarse uniform grid if greedy produced nothing usable.
    if best_pos is None or best_r < 0.02:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_pos = grid.copy()

    if best_pos is None:
        # Absolute safe fallback.
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)

    # 2) Local refinement to polish the greedy solution.
    if _time_left() > 10.0:
        refined, r = _local_refine(best_pos, _time_left())
        if r > best_r:
            best_r = r
            best_pos = refined

    # 3) Dynamics optimization for further improvement.
    if _time_left() > 15.0:
        alloc_time = min(30.0, _time_left() - 10.0)
        if alloc_time > 5.0:
            pts_opt = _dynamics_optimize(best_pos, time_limit=alloc_time)
            r = _compute_radius(pts_opt)
            if r > best_r:
                best_r = r
                best_pos = pts_opt.copy()

    # Final safety: guarantee validity (finite, inside square, correct shape).
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        # Repair shape by padding/truncating with a valid grid.
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_pos = grid
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pos
