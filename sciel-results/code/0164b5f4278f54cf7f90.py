"""Circle packing in perimeter-4 rectangle: SLSQP + Voronoi-based expansion + polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid (vectorized)."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True


def _max_radius_voronoi(x, y, r, w, h, i):
    """Compute maximum radius for circle i using Voronoi cell inradius."""
    # First get a quick upper bound from simple method
    r_simple = _max_radius_simple(x, y, r, w, h, i)

    # Only use Voronoi if the simple estimate is tight (likely constrained by neighbors)
    # Check if circle i is close to any wall or other circle
    wall_dist = min(x[i], w - x[i], y[i], h - y[i])
    min_neighbor_dist = float('inf')
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        if d - r[j] < min_neighbor_dist:
            min_neighbor_dist = d - r[j]

    # If the limiting factor is clearly a wall (not a neighbor), skip Voronoi
    if wall_dist < min_neighbor_dist - 0.005 and wall_dist < r_simple + 0.001:
        return r_simple

    pts = list(zip(x, y))
    boundary_pts = [
        (0, 0), (w, 0), (0, h), (w, h),
        (0, h / 2), (w, h / 2), (w / 2, 0), (w / 2, h),
        (w * 0.25, 0), (w * 0.75, 0), (w * 0.25, h), (w * 0.75, h),
        (0, h * 0.25), (0, h * 0.75), (w, h * 0.25), (w, h * 0.75),
    ]
    all_pts = np.array(pts + boundary_pts)

    try:
        vor = Voronoi(all_pts)
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            return r_simple

        vertices = vor.vertices[region]
        if len(vertices) < 3:
            return r_simple

        min_dist = float('inf')
        for k in range(len(vertices)):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % len(vertices)]
            dist = _point_to_segment_dist(x[i], y[i], v1[0], v1[1], v2[0], v2[1])
            if dist < min_dist:
                min_dist = dist

        r_voronoi = min_dist
    except Exception:
        return r_simple

    # Also consider other circles' radii (they can't overlap)
    r_max = r_voronoi
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed

    return max(0.0, r_max)


def _max_radius_simple(x, y, r, w, h, i):
    """Fallback: compute max radius by distance to walls and other circles."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _smooth_objective(z, penalty_weight=1000.0, beta=50.0):
    """Smooth objective using log-sum-exp for differentiable penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap: use smooth max via log-sum-exp
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    violations = r_sum_pairs - d_pairs

    # Smooth penalty: beta * log(sum(exp(violations/beta))) approximates max(violations)
    # Use stable log-sum-exp
    v_shifted = violations - np.max(violations)
    smooth_max = np.max(violations) + (1.0 / beta) * np.log(np.sum(np.exp(v_shifted)))
    obj += penalty_weight * smooth_max * np.max(violations)

    # Boundary penalties (smooth)
    left_v = r - x
    right_v = x + r - w
    bottom_v = r - y
    top_v = y + r - h

    for v in [left_v, right_v, bottom_v, top_v]:
        v_shifted = v - np.max(v)
        sm = np.max(v) + (1.0 / beta) * np.log(np.sum(np.exp(v_shifted)))
        if sm > 0:
            obj += penalty_weight * sm * sm

    # Aspect ratio soft constraint
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _force_relax(z, max_iter=500):
    """Force-directed relaxation to resolve initial overlaps (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

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
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-8:
            f_pairs = overlap * 15.0
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

        if max_overlap < 1e-8:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _voronoi_expansion_polish(z, time_budget):
    """
    Polish using Voronoi-based radius computation for accurate max radius estimation.
    Iteratively expands each circle to its Voronoi-limited maximum, then adjusts positions.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # Compute max radius using Voronoi
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-7:
                # Binary search for true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(22):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
            else:
                # Try to move circle i to a position allowing larger radius
                # Find most constraining neighbor/wall
                min_gap = float('inf')
                blocker_idx = -1
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        blocker_idx = j

                wall_gaps = [x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]]
                min_wall_gap = min(wall_gaps)

                if min_gap < 0.015 or min_wall_gap < 0.015:
                    # Generate candidate move directions
                    directions = []
                    if blocker_idx >= 0:
                        dx = x[i] - x[blocker_idx]
                        dy = y[i] - y[blocker_idx]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-8:
                            directions.append((dx / d, dy / d))

                    if wall_gaps[0] < 0.015:
                        directions.append((1.0, 0.0))
                    if wall_gaps[1] < 0.015:
                        directions.append((-1.0, 0.0))
                    if wall_gaps[2] < 0.015:
                        directions.append((0.0, 1.0))
                    if wall_gaps[3] < 0.015:
                        directions.append((0.0, -1.0))

                    # Diagonal directions
                    for dx_dir, dy_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        norm = np.sqrt(2.0)
                        directions.append((dx_dir / norm, dy_dir / norm))

                    for dx_dir, dy_dir in directions:
                        for step in [0.03, 0.015, 0.008, 0.004]:
                            new_x = x[i] + dx_dir * step
                            new_y = y[i] + dy_dir * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    # Final uniform expansion
    for _ in range(200):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0001
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            r_trial = r * 1.00002
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
            else:
                break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _polish_result(z, time_budget):
    """Polish with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _high_precision_polish(z, time_budget):
    """
    High-precision polish: iterative radius growth with position adjustment.
    Uses Voronoi for accurate radius bounds and adaptive step sizing.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Phase 1: Force-directed position refinement
    dt = 0.0005
    max_iter = int(min(6000, time_budget * 250))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-12:
            f_pairs = overlap * 30.0
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

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.0002
        elif max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

        if max_overlap < 1e-10 and max_force < 0.01:
            break

    # Phase 2: Voronoi-guided individual radius expansion
    for _ in range(60):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            r_max = _max_radius_voronoi(x, y, r, w, h, i)

            if r_max > r[i] + 1e-9:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h, tol=1e-10):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-10:
                    r[i] = lo
                    improved = True
            else:
                # Try to move circle i
                min_gap = float('inf')
                blocker_idx = -1
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    gap = d - (r[i] + r[j])
                    if gap < min_gap:
                        min_gap = gap
                        blocker_idx = j

                wall_gaps = [x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]]
                min_wall_gap = min(wall_gaps)

                if min_gap < 0.01 or min_wall_gap < 0.01:
                    directions = []
                    if blocker_idx >= 0:
                        dx = x[i] - x[blocker_idx]
                        dy = y[i] - y[blocker_idx]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-8:
                            directions.append((dx / d, dy / d))

                    if wall_gaps[0] < 0.01:
                        directions.append((1.0, 0.0))
                    if wall_gaps[1] < 0.01:
                        directions.append((-1.0, 0.0))
                    if wall_gaps[2] < 0.01:
                        directions.append((0.0, 1.0))
                    if wall_gaps[3] < 0.01:
                        directions.append((0.0, -1.0))

                    for dx_dir, dy_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        norm = np.sqrt(2.0)
                        directions.append((dx_dir / norm, dy_dir / norm))

                    for dx_dir, dy_dir in directions:
                        for step in [0.02, 0.01, 0.005]:
                            new_x = x[i] + dx_dir * step
                            new_y = y[i] + dy_dir * step
                            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                                x[i] = new_x
                                y[i] = new_y
                                improved = True
                                break
                        if improved:
                            break

        if not improved:
            break

    # Phase 3: Final simultaneous squeeze
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * 1.00005
        if _is_valid(x, y, r_trial, w, h, tol=1e-10):
            r = r_trial
        else:
            r_trial = r * 1.00001
            if _is_valid(x, y, r_trial, w, h, tol=1e-10):
                r = r_trial
            else:
                break

    if _is_valid(x, y, r, w, h, tol=1e-10):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _epigraph_radius_growth(z, time_budget):
    """
    Epigraph-based radius growth: introduce auxiliary variable t and solve
    for max t such that all constraints hold with radii scaled by t.
    Uses SLSQP with smooth constraints.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Scale radii down slightly to ensure feasibility
    scale = 0.99
    r_base = r * scale

    # Solve for maximum uniform scale factor using SLSQP
    # Variables: [x_0..x_n-1, y_0..y_n-1, t] where t is the scale factor
    def epigraph_obj(v):
        xv = v[0:n]
        yv = v[n:2*n]
        t = v[2*n]
        if t < 0.1 or t > 2.0:
            return 1e6
        rv = r_base * t
        hw = 2.0 - w

        obj = -t * np.sum(r_base)

        # Overlap penalties
        dx = xv[:, None] - xv[None, :]
        dy = yv[:, None] - yv[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = rv[:, None] + rv[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        violations = r_sum[iu] - d_pairs
        penalty = np.sum(np.maximum(violations, 0.0) ** 2) * 800.0
        obj += penalty

        # Boundary penalties
        left_v = np.maximum(rv - xv, 0.0)
        right_v = np.maximum(xv + rv - w, 0.0)
        bottom_v = np.maximum(rv - yv, 0.0)
        top_v = np.maximum(yv + rv - hw, 0.0)
        obj += 800.0 * (np.sum(left_v**2) + np.sum(right_v**2) +
                        np.sum(bottom_v**2) + np.sum(top_v**2))

        return obj

    bounds = [(0, 2)] * (2 * n) + [(0.5, 1.5)]
    v0 = np.concatenate([x, y, [1.0]])

    maxiter = int(min(500, time_budget * 100))
    try:
        res = minimize(epigraph_obj, v0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})

        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2*n]
        t_opt = v_opt[2*n]
        r_opt = r_base * t_opt

        if _is_valid(x_opt, y_opt, r_opt, w, h, tol=1e-9):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x_opt
            z_new[1::3][:n] = y_opt
            z_new[2::3][:n] = r_opt
            z_new[-1] = w
            return z_new
    except Exception:
        pass
    return None


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using smooth log-sum-exp objective."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Smooth log-sum-exp penalty for overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        violations = r_sum[iu] - d_pairs

        # Stable log-sum-exp
        v_max = np.max(violations)
        if v_max > 0:
            v_shifted = violations - v_max
            smooth_max = v_max + (1.0 / 40.0) * np.log(np.sum(np.exp(v_shifted)))
            obj += 600.0 * smooth_max * v_max
        else:
            obj += 600.0 * np.sum(np.maximum(violations, 0.0) ** 2)

        # Boundary penalties (smooth)
        for v in [r - x, x + r - w, r - y, y + r - h]:
            v_max = np.max(v)
            if v_max > 0:
                v_shifted = v - v_max
                sm = v_max + (1.0 / 40.0) * np.log(np.sum(np.exp(v_shifted)))
                obj += 600.0 * sm * v_max
            else:
                obj += 600.0 * np.sum(np.maximum(v, 0.0) ** 2)

        return obj

    bounds = [(0, 2)] * (3 * n) + [(0.3, 1.7)]

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 120})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - z_opt[-1]

        # Repair if slightly invalid
        if not _is_valid(x, y, r, w, h):
            for _ in range(20):
                r = r * 0.999
                if _is_valid(x, y, r, w, h):
                    z_opt[2::3][:n] = r
                    break
            else:
                return None

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # Grid-based with various aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 0.9, 1.1]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random with better radius estimation
        rng = np.random.default_rng(42)
        for seed in range(15):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            # Use Poisson-disk-like random placement
            placed = []
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                attempts = 0
                while attempts < 50:
                    cx = rng.uniform(r0, w0 - r0)
                    cy = rng.uniform(r0, h0 - r0)
                    ok = True
                    for (px, py, pr) in placed:
                        if (cx - px)**2 + (cy - py)**2 < (r0 + pr + 0.01)**2:
                            ok = False
                            break
                    if ok:
                        placed.append((cx, cy, r0))
                        z[3 * i] = cx
                        z[3 * i + 1] = cy
                        z[3 * i + 2] = r0
                        break
                    attempts += 1
                else:
                    # Fallback: place at grid position
                    row, col = divmod(i, side)
                    cx = (col + 0.5) / side * w0
                    cy = (row + 0.5) / side * h0
                    r0 = min(w0 / side, h0 / side) * 0.4
                    z[3 * i] = cx
                    z[3 * i + 1] = cy
                    z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Hexagonal packing
        for w0 in [0.9, 1.0, 1.1, 1.05]:
            h0 = 2.0 - w0
            if h0 < 0.4:
                continue
            z = np.zeros(3 * N + 1)
            rows = 5
            cols = 5
            spacing_x = w0 / (cols + 0.5)
            spacing_y = h0 / (rows + 0.5)
            for i in range(N):
                row, col = divmod(i, cols)
                cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * spacing_x
                cy = (row + 0.5) * spacing_y
                r0 = min(spacing_x, spacing_y) * 0.48
                cx = min(max(cx, r0), w0 - r0)
                cy = min(max(cy, r0), h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        z0_relaxed = _force_relax(z0, max_iter=300)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(400, max(100, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Polish ---
    if best_result is not None and time.time() < deadline - 20:
        # 1. Force-directed polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 12)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Voronoi-based expansion
        if best_result is not None and time.time() < deadline - 10:
            expanded = _voronoi_expansion_polish(best_result, deadline - (time.time() - start_time) - 5)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Epigraph-based radius growth (new)
        if best_result is not None and time.time() < deadline - 8:
            epi = _epigraph_radius_growth(best_result, deadline - (time.time() - start_time) - 4)
            if epi is not None:
                valid, val = check_validity(epi)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi

        # 4. Constraint-based SLSQP refinement with smooth objective
        if best_result is not None and time.time() < deadline - 5:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 5. Final high-precision polish
        if best_result is not None and time.time() < deadline - 2:
            final_polished = _high_precision_polish(best_result, deadline - (time.time() - start_time))
            if final_polished is not None:
                valid, val = check_validity(final_polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_polished

        # 6. One more round of epigraph + constraint refinement
        if best_result is not None and time.time() < deadline - 1:
            epi2 = _epigraph_radius_growth(best_result, 1.0)
            if epi2 is not None:
                valid, val = check_validity(epi2)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi2

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
