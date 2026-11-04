"""Circle packing in unit square: contact-graph (Delaunay) local search + gradient polish."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """Return (total_radius, is_valid)."""
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
    return float(np.sum(r)), True


def _hex_init():
    """Hexagonal-style seed: rows of equal circles, alternating offset."""
    # 5 rows of 5, 6, 5, 6, 5? We need 26. Use 5 rows: 5,6,5,6,4 = 26
    rows = [5, 6, 5, 6, 4]
    pts = []
    n_rows = len(rows)
    # vertical spacing: rows centered
    # We'll place rows at y = (k+0.5)/n_rows
    # radius will be determined later; start with small
    for k, cnt in enumerate(rows):
        y = (k + 0.5) / n_rows
        for j in range(cnt):
            # horizontal spacing with slight offset for odd rows to mimic hex
            if k % 2 == 1:
                x = (j + 0.5) / cnt
            else:
                x = (j + 0.5) / cnt
            # nudge alternating rows by half a cell horizontally for hex feel
            if k % 2 == 1:
                x = x + (0.5 / cnt) * 0.5
                x = min(x, 1.0 - 0.01)
            x = max(0.01, min(x, 0.99))
            pts.append([x, y, 0.02])
    while len(pts) < N:
        pts.append([0.5, 0.5, 0.01])
    return np.array(pts[:N])


def _max_radius_at(x, y, others_xy, others_r):
    """Max radius for a circle centered at (x,y) given others fixed."""
    wall = min(x, 1.0 - x, y, 1.0 - y)
    if wall <= 0:
        return 0.0
    if len(others_xy) == 0:
        return max(wall, 0.0)
    dx = others_xy[:, 0] - x
    dy = others_xy[:, 1] - y
    d = np.sqrt(dx * dx + dy * dy)
    clear = np.min(d - others_r)
    return max(0.0, min(wall, clear))


def _contact_graph_local_search(xy, r, deadline, grid_res=40, max_sweeps=40):
    """
    Direct contact-graph local search:
    1. Build Delaunay triangulation of centers -> neighbor list.
    2. For each circle i (in random order), maximize r_i by choosing position
       in a local grid around current center, with all other circles fixed.
       Use the Delaunay neighbors to define a 'tight' local search window.
    3. Repeat sweeps until no improvement or deadline.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    rng = np.random.default_rng(12345)

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        # Rebuild Delaunay each sweep (positions change)
        try:
            tri = Delaunay(xy)
            # Build adjacency from simplices
            adj = [set() for _ in range(n)]
            for simplex in tri.simplices:
                for a in simplex:
                    for b in simplex:
                        if a != b:
                            adj[a].add(b)
        except Exception:
            # Fallback: full neighborhood
            adj = [set(range(n)) - {i} for i in range(n)]

        order = list(range(n))
        rng.shuffle(order)
        improved = False

        for i in order:
            if time.time() > deadline:
                break
            cx, cy = xy[i, 0], xy[i, 1]
            ri = r[i]

            # Determine local search window: use neighbors' distances to bound
            neighbors = list(adj[i])
            if neighbors:
                # window = max neighbor distance + a bit, but cap to reasonable
                dists = []
                for j in neighbors:
                    dx = xy[i, 0] - xy[j, 0]
                    dy = iy = xy[i, 1] - xy[j, 1]
                    dists.append(np.sqrt(dx * dx + dy * dy))
                half_win = max(0.05, 0.7 * max(dists)) if dists else 0.1
            else:
                half_win = 0.15

            # Local grid around (cx, cy)
            half_win = min(half_win, 0.45)  # don't go outside square
            x0 = max(0.0, cx - half_win)
            x1 = min(1.0, cx + half_win)
            y0 = max(0.0, cy - half_win)
            y1 = min(1.0, cy + half_win)

            if x1 - x0 < 1e-6:
                x0, x1 = max(0.0, cx - 0.01), min(1.0, cx + 0.01)
            if y1 - y0 < 1e-6:
                y0, y1 = max(0.0, cy - 0.01), min(1.0, cy + 0.01)

            gx = np.linspace(x0, x1, grid_res)
            gy = np.linspace(y0, y1, grid_res)
            GX, GY = np.meshgrid(gx, gy, indexing='ij')
            cand = np.column_stack([GX.ravel(), GY.ravel()])

            # Others (all circles except i)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute max radius for each candidate
            best_r_cand = -1.0
            best_idx = 0
            # Vectorized: for each candidate, compute min over others of (dist - r_other)
            if len(others_xy) > 0:
                dx = cand[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = cand[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                wall_clear = np.minimum(np.minimum(cand[:, 0], 1.0 - cand[:, 0]),
                                        np.minimum(cand[:, 1], 1.0 - cand[:, 1]))
                r_cand = np.minimum(wall_clear, circle_clear)
                r_cand = np.maximum(r_cand, 0.0)
                idx = int(np.argmax(r_cand))
                best_r_cand = r_cand[idx]
                best_idx = idx
            else:
                best_r_cand = min(cx, 1.0 - cx, cy, 1.0 - cy)

            if best_r_cand > ri + 1e-10:
                xy[i, 0] = cand[best_idx, 0]
                xy[i, 1] = cand[best_idx, 1]
                r[i] = best_r_cand
                improved = True

        # After sweep, evaluate
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _soft_objective(params, n):
    """Penalized objective: -sum(r) + penalty for violations."""
    params = np.asarray(params)
    xy = params[0::3].reshape(-1, 2)
    r = np.maximum(params[1::3], 0.001)

    # Wall violations
    wall_v = 0.0
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        wall_v += max(0.0, ri - x) ** 2
        wall_v += max(0.0, ri - (1.0 - x)) ** 2
        wall_v += max(0.0, ri - y) ** 2
        wall_v += max(0.0, ri - (1.0 - y)) ** 2

    # Overlap violations
    overlap_v = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if dist < min_d:
                overlap_v += (min_d - dist) ** 2

    penalty = 1000.0 * (wall_v + overlap_v)
    return -float(np.sum(r)) + penalty


def _gradient_polish(xy, r, deadline, max_iter=300):
    """L-BFGS-B polish on the soft objective."""
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.45))

    best_total, best_valid = _eval_packing(xy, r)
    best_params = params.copy()

    try:
        result = minimize(
            _soft_objective,
            params,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 4000, 'ftol': 1e-16, 'gtol': 1e-10}
        )
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
    except Exception:
        pass

    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    return final_xy, final_r, best_total, best_valid


def _slsqp_polish(xy, r, deadline):
    """SLSQP with explicit constraints as a final polish."""
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.45))

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i * 3] - p[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i * 3] - p[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i * 3 + 1] - p[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i * 3 + 1] - p[i * 3 + 2]})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt((p[i * 3] - p[j * 3]) ** 2 +
                                                    (p[i * 3 + 1] - p[j * 3 + 1]) ** 2)
                                                    - (p[i * 3 + 2] + p[j * 3 + 2])
            })

    best_total, best_valid = _eval_packing(xy, r)
    best_params = params.copy()

    try:
        result = minimize(
            lambda p: -float(np.sum(p[2::3])),
            params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-14}
        )
        if result.success:
            opt_params = result.x
            opt_xy = opt_params[0::3].reshape(-1, 2)
            opt_r = np.maximum(opt_params[1::3], 0.001)
            total, valid = _eval_packing(opt_xy, opt_r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_params = opt_params.copy()
    except Exception:
        pass

    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    return final_xy, final_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Initialization: hexagonal seed
    init = _hex_init()
    xy = init[:, :2].copy()
    r = init[:, 2].copy()

    # Phase 1: Contact-graph local search (the core new strategy)
    if time.time() < deadline - 40.0:
        cg_xy, cg_r, cg_total, cg_valid = _contact_graph_local_search(
            xy, r, deadline - 40.0, grid_res=45, max_sweeps=50
        )
        if cg_valid and cg_total > best_total:
            best_total = cg_total
            best_valid = True
            best_xy = cg_xy
            best_r = cg_r

    # If contact search didn't produce a valid packing, fallback to simple grid
    if best_xy is None or not best_valid:
        side = 5
        result = np.zeros((N, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                result[idx, 0] = (j + 0.5) / side
                result[idx, 1] = (i + 0.5) / side
                result[idx, 2] = 0.45 / side
                idx += 1
        while idx < N:
            result[idx, 0] = 0.5
            result[idx, 1] = 0.5
            result[idx, 2] = 0.01
            idx += 1
        total, valid = _eval_packing(result[:, :2], result[:, 2])
        if valid:
            best_total = total
            best_valid = True
            best_xy = result[:, :2]
            best_r = result[:, 2]
        else:
            # Tiny circles
            for i in range(N):
                result[i, 2] = 0.01
            return result

    # Phase 2: Gradient polish (L-BFGS-B on soft objective)
    if time.time() < deadline - 20.0:
        g_xy, g_r, g_total, g_valid = _gradient_polish(
            best_xy, best_r, deadline - 20.0, max_iter=300
        )
        if g_valid and g_total > best_total:
            best_total = g_total
            best_valid = True
            best_xy = g_xy
            best_r = g_r

    # Phase 3: Another contact-graph local search pass (refine after gradient)
    if time.time() < deadline - 15.0:
        cg2_xy, cg2_r, cg2_total, cg2_valid = _contact_graph_local_search(
            best_xy, best_r, deadline - 15.0, grid_res=50, max_sweeps=30
        )
        if cg2_valid and cg2_total > best_total:
            best_total = cg2_total
            best_valid = True
            best_xy = cg2_xy
            best_r = cg2_r

    # Phase 4: SLSQP final polish
    if time.time() < deadline - 8.0:
        s_xy, s_r, s_total, s_valid = _slsqp_polish(best_xy, best_r, deadline - 8.0)
        if s_valid and s_total > best_total:
            best_total = s_total
            best_valid = True
            best_xy = s_xy
            best_r = s_r

    # Final gradient polish if time remains
    if time.time() < deadline - 3.0:
        g2_xy, g2_r, g2_total, g2_valid = _gradient_polish(
            best_xy, best_r, deadline - 3.0, max_iter=100
        )
        if g2_valid and g2_total > best_total:
            best_total = g2_total
            best_valid = True
            best_xy = g2_xy
            best_r = g2_r

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
        # Fallback to simple grid
        side = 5
        result = np.zeros((N, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                result[idx, 0] = (j + 0.5) / side
                result[idx, 1] = (i + 0.5) / side
                result[idx, 2] = 0.45 / side
                idx += 1
        while idx < N:
            result[idx, 0] = 0.5
            result[idx, 1] = 0.5
            result[idx, 2] = 0.01
            idx += 1

    return result
# EVOLVE-BLOCK-END
