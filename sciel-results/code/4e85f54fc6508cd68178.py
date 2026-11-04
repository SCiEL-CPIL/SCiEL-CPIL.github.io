"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: LECV-based sequential placement + Voronoi largest-empty-circle local search."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26

# EVOLVE-BLOCK-START


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _largest_empty_circle_at(px, py, others_xy, others_r):
    """
    Compute the maximum radius of a circle centered at (px, py) that fits
    inside [0,1]^2 and is disjoint from all other circles.
    Uses circle-circle intersection points for exact Voronoi vertex candidates.
    """
    n = len(others_xy)
    # Wall clearance
    r_wall = min(px, 1.0 - px, py, 1.0 - py)
    if r_wall < 0:
        r_wall = 0.0

    # Distance to each other circle
    best_r = r_wall

    if n == 0:
        return best_r

    # Distance to each other circle (tangent distance)
    dists = np.sqrt((others_xy[:, 0] - px) ** 2 + (others_xy[:, 1] - py) ** 2)
    clearances = dists - others_r
    best_r = max(best_r, np.max(clearances))

    # Circle-circle intersection points as Voronoi vertex candidates
    # For each pair of other circles, find intersection points of the two
    # "expanded" circles (radius = r_i + candidate_r) — but we need the
    # point equidistant to (px,py) and two other circles.
    # Actually, the LEC center candidates are:
    # 1. (px, py) itself (already handled via clearances)
    # 2. Points where the expanded circles of two others meet
    #    i.e., points P where |P - c_i| = |P - c_j| = r_i + r (and r = |P - c_i| - r_i)
    #    But we're fixing the center at (px,py), so the LEC radius is just
    #    min over all constraints. The "largest empty circle" at a FIXED center
    #    is simply min(wall_clear, min_i(dist_i - r_i)).
    #
    # However, for the RELAX step, we want to find the best position for circle i
    # given all others. That's a 2D optimization. Instead, let's use a different
    # approach: for the local search, we'll use a grid + refinement, but with
    # the exact clearance computation (no sampling error in the radius).
    #
    # For the SEQUENTIAL PLACEMENT, we need to find the largest empty circle
    # in the free space. That's where Voronoi + circle intersections help.

    return best_r


def _largest_empty_circle_global(others_xy, others_r, deadline):
    """
    Find the largest empty circle in [0,1]^2 given existing circles.
    Uses Voronoi diagram of circle centers + circle-circle intersection points.
    Returns (best_x, best_y, best_r).
    """
    n = len(others_xy)
    if n == 0:
        return 0.5, 0.5, 0.5

    # Candidate centers:
    # 1. Voronoi vertices of the circle centers
    # 2. Circle-circle intersection points (of the actual circles, not expanded)
    # 3. Points where a circle is tangent to a wall
    # 4. A grid for safety

    candidates = []

    # Voronoi vertices
    try:
        vor = Voronoi(others_xy)
        for v in vor.vertices:
            if 0.0 <= v[0] <= 1.0 and 0.0 <= v[1] <= 1.0:
                candidates.append(v)
    except Exception:
        pass

    # Circle-circle intersection points
    for i in range(n):
        for j in range(i + 1, n):
            c1 = others_xy[i]
            c2 = others_xy[j]
            r1 = others_r[i]
            r2 = others_r[j]
            d = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            if d < 1e-12 or d > r1 + r2 + 1e-9:
                continue
            if d < abs(r1 - r2) - 1e-9:
                continue
            # Intersection of circles (c1,r1) and (c2,r2)
            a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
            h_sq = r1 * r1 - a * a
            if h_sq < 0:
                continue
            h = np.sqrt(h_sq)
            mx = c1[0] + a * (c2[0] - c1[0]) / d
            my = c1[1] + a * (c2[1] - c1[1]) / d
            # Two intersection points
            ix1 = mx + h * (c2[1] - c1[1]) / d
            iy1 = my - h * (c2[0] - c1[0]) / d
            ix2 = mx - h * (c2[1] - c1[1]) / d
            iy2 = my + h * (c2[0] - c1[0]) / d
            if 0.0 <= ix1 <= 1.0 and 0.0 <= iy1 <= 1.0:
                candidates.append([ix1, iy1])
            if 0.0 <= ix2 <= 1.0 and 0.0 <= iy2 <= 1.0:
                candidates.append([ix2, iy2])

    # Wall-tangent points: for each circle, the point on the circle closest to each wall
    # Actually, the LEC center on a wall tangent is where the circle touches a wall.
    # The center of the LEC touching the left wall at (0, y) with radius r means
    # the center is at (r, y). This is already covered by the grid/Voronoi.

    # Grid candidates for robustness
    grid_size = 50
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    for x in gx:
        for y in gy:
            candidates.append([x, y])

    if not candidates:
        return 0.5, 0.5, 0.0

    candidates = np.array(candidates)

    # Compute clearance for each candidate
    best_r = 0.0
    best_pos = (0.5, 0.5)

    for cx, cy in candidates:
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        r_wall = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if r_wall < 0:
            r_wall = 0.0
        dists = np.sqrt((others_xy[:, 0] - cx) ** 2 + (others_xy[:, 1] - cy) ** 2)
        clearances = dists - others_r
        r_cand = min(r_wall, np.min(clearances)) if n > 0 else r_wall
        if r_cand > best_r:
            best_r = r_cand
            best_pos = (cx, cy)

    return best_pos[0], best_pos[1], max(best_r, 0.0)


def _sequential_placement(deadline, seed=0):
    """
    Sequentially place N circles using LECV.
    Start with a few large circles, then fill in the gaps.
    """
    rng = np.random.RandomState(seed)
    placed_xy = []
    placed_r = []

    # Place first circle at center
    placed_xy.append([0.5, 0.5])
    placed_r.append(0.5)

    for i in range(1, N):
        if time.time() > deadline:
            break
        if len(placed_xy) == 0:
            x, y, r = 0.5, 0.5, 0.5
        else:
            xy_arr = np.array(placed_xy)
            r_arr = np.array(placed_r)
            x, y, r = _largest_empty_circle_global(xy_arr, r_arr, deadline)

        if r < 0.001:
            # Try a random perturbation
            for _ in range(20):
                x_try = rng.uniform(0.05, 0.95)
                y_try = rng.uniform(0.05, 0.95)
                if len(placed_xy) == 0:
                    r_try = min(x_try, 1.0 - x_try, y_try, 1.0 - y_try)
                else:
                    xy_arr = np.array(placed_xy)
                    r_arr = np.array(placed_r)
                    dists = np.sqrt((xy_arr[:, 0] - x_try) ** 2 + (xy_arr[:, 1] - y_try) ** 2)
                    clearances = dists - r_arr
                    r_try = min(min(x_try, 1.0 - x_try, y_try, 1.0 - y_try), np.min(clearances))
                if r_try > r:
                    x, y, r = x_try, y_try, r_try

        if r > 0.0005:
            placed_xy.append([x, y])
            placed_r.append(r)

    if len(placed_xy) < N:
        # Pad with tiny circles
        while len(placed_xy) < N:
            placed_xy.append([0.5, 0.5])
            placed_r.append(0.001)

    return np.array(placed_xy[:N]), np.array(placed_r[:N])


def _lecv_local_search(xy, r, deadline, max_sweeps=15):
    """
    Local search: for each circle, find the best position using LECV.
    The best position for circle i is the LEC of the space occupied by all other circles,
    but we also allow the circle to stay at its current position if that's better.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Remove circle i, find LEC in the remaining space
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Find the LEC
            lx, ly, lr = _largest_empty_circle_global(others_xy, others_r, deadline)

            # Also consider staying at current position with current radius
            cur_r = r[i]
            cur_x, cur_y = xy[i, 0], xy[i, 1]
            # Check if current position is valid with current radius
            if len(others_xy) > 0:
                dists = np.sqrt((others_xy[:, 0] - cur_x) ** 2 + (others_xy[:, 1] - cur_y) ** 2)
                clearances = dists - others_r
                cur_feasible_r = min(min(cur_x, 1.0 - cur_x, cur_y, 1.0 - cur_y), np.min(clearances))
            else:
                cur_feasible_r = min(cur_x, 1.0 - cur_x, cur_y, 1.0 - cur_y)

            # Choose the better option
            if lr > cur_feasible_r + 1e-10:
                xy[i, 0] = lx
                xy[i, 1] = ly
                r[i] = lr
                improved = True
            elif cur_feasible_r > cur_r + 1e-10:
                # Stay at current position but increase radius
                r[i] = cur_feasible_r
                improved = True

        # Evaluate after sweep
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _slsqp_polish(xy, r, deadline, max_iter=500):
    """
    SLSQP polish with hard constraints.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    # Objective: maximize sum(r) => minimize -sum(r)
    def objective(v):
        return -np.sum(v[2*n:3*n])

    # Gradient
    def gradient(v):
        g = np.zeros(3 * n)
        g[2*n:3*n] = -1.0
        return g

    # Inequality constraints: g(v) >= 0
    # Wall constraints: x_i >= r_i, 1-x_i >= r_i, y_i >= r_i, 1-y_i >= r_i
    # Pairwise: dist_ij >= r_i + r_j

    constraints = []

    # Wall constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[i] - v[2*n+i],
                            'jac': lambda v, i=i: np.concatenate([np.eye(n)[i], -np.eye(n)[i], -np.eye(n)[i]])})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i],
                            'jac': lambda v, i=i: np.concatenate([-np.eye(n)[i], np.zeros(n), -np.eye(n)[i]])})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: v[n+i] - v[2*n+i],
                            'jac': lambda v, i=i: np.concatenate([np.zeros(n), np.eye(n)[i], -np.eye(n)[i]])})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq',
                            'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i],
                            'jac': lambda v, i=i: np.concatenate([np.zeros(n), -np.eye(n)[i], -np.eye(n)[i]])})

    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            def pair_fun(v, i=i, j=j):
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - v[2*n+i] - v[2*n+j]

            def pair_jac(v, i=i, j=j):
                jac = np.zeros(3*n)
                dx = v[i] - v[j]
                dy = v[n+i] - v[n+j]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-20:
                    dist = 1e-10
                    jac[i] = 0.0
                    jac[n+i] = 0.0
                    jac[j] = 0.0
                    jac[n+j] = 0.0
                else:
                    dist = np.sqrt(dist_sq)
                    jac[i] = dx / dist
                    jac[n+i] = dy / dist
                    jac[j] = -dx / dist
                    jac[n+j] = -dy / dist
                jac[2*n+i] = -1.0
                jac[2*n+j] = -1.0
                return jac

            constraints.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
    for i in range(n):
        bounds.append((0.001, 0.999))  # y
    for i in range(n):
        bounds.append((0.001, 0.5))  # r

    try:
        result = minimize(objective, v0, jac=gradient, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': max_iter, 'ftol': 1e-12})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Sequential LECV placement with multiple seeds
    for seed in range(6):
        if time.time() > deadline - 30:
            break
        try:
            xy, r = _sequential_placement(deadline, seed=seed)
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
        except Exception:
            continue

    # Phase 2: Grid initialization
    for side in [5, 6]:
        if time.time() > deadline - 20:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

    # Phase 3: LECV local search
    if best_valid and best_xy is not None:
        if time.time() < deadline - 30:
            ls_xy, ls_r, ls_total, ls_valid = _lecv_local_search(best_xy.copy(), best_r.copy(), deadline)
            if ls_valid and ls_total > best_total:
                best_total = ls_total
                best_valid = True
                best_xy = ls_xy
                best_r = ls_r

    # Phase 4: SLSQP polish
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        try:
            pol_xy, pol_r = _slsqp_polish(best_xy.copy(), best_r.copy(), deadline, max_iter=300)
            total, valid = _eval_packing(pol_xy, pol_r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = pol_xy
                best_r = pol_r
        except Exception:
            pass

    # Phase 5: Another LECV local search pass
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        try:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _lecv_local_search(best_xy.copy(), best_r.copy(), deadline, max_sweeps=8)
            if ls2_valid and ls2_total > best_total + 1e-10:
                best_total = ls2_total
                best_valid = True
                best_xy = ls2_xy
                best_r = ls2_r
        except Exception:
            pass

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
