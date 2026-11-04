"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
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


def _largest_empty_circle(xy, r, grid_size=100):
    """
    Find the center and radius of the largest circle that fits inside the unit square
    and is disjoint from all existing circles (xy, r).
    Uses a grid search over candidate centers.
    """
    n = len(xy)
    # Candidate centers: grid over [0,1]^2
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Wall clearance for each candidate
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    if n == 0:
        # No existing circles: pick center of square
        idx = np.argmax(wall_clear)
        return candidates[idx, 0], candidates[idx, 1], wall_clear[idx]

    # Circle clearance: for each candidate, min over all existing circles of (dist - r_j)
    # dist from candidate to circle j
    # candidates shape (M, 2), xy shape (n, 2)
    dx = candidates[:, 0:1] - xy[:, 0][np.newaxis, :]  # (M, n)
    dy = candidates[:, 1:2] - xy[:, 1][np.newaxis, :]  # (M, n)
    dists = np.sqrt(dx * dx + dy * dy)  # (M, n)
    circle_clear = np.min(dists - r[np.newaxis, :], axis=1)  # (M,)

    max_r = np.minimum(wall_clear, circle_clear)
    idx = int(np.argmax(max_r))
    return candidates[idx, 0], candidates[idx, 1], max_r[idx]


def _local_optimize_new_circle(xy_fixed, r_fixed, x0, y0, r0, deadline, n_iter=200):
    """
    Given fixed existing circles (xy_fixed, r_fixed), optimize the position and radius
    of one new circle starting from (x0, y0, r0).
    We maximize r subject to:
      - circle inside [0,1]^2
      - disjoint from all fixed circles
    Use gradient-free optimization (Nelder-Mead) on (x, y) with r computed from clearance.
    """
    n_fixed = len(xy_fixed)

    def _clearance(x, y):
        # Wall clearance
        wc = min(x, 1.0 - x, y, 1.0 - y)
        if n_fixed == 0:
            return wc
        # Circle clearance
        cc = wc
        for j in range(n_fixed):
            dx = x - xy_fixed[j, 0]
            dy = y - xy_fixed[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            cc = min(cc, dist - r_fixed[j])
        return cc

    def _objective(params):
        x, y = params
        # Penalty if outside square
        if x < 0 or x > 1 or y < 0 or y > 1:
            return -1e6
        r = _clearance(x, y)
        if r < 0:
            return -1e6 + r
        return -r  # Minimize negative radius

    x0c = max(r0, min(1.0 - r0, x0))
    y0c = max(r0, min(1.0 - r0, y0))
    r0c = max(0.001, min(r0, _clearance(x0c, y0c)))

    res = minimize(_objective, [x0c, y0c], method='Nelder-Mead',
                   options={'maxiter': n_iter, 'xatol': 1e-5, 'fatol': 1e-5})

    if res.success or res.fun < -0.001:
        x_opt, y_opt = res.x
        r_opt = _clearance(x_opt, y_opt)
        if r_opt > 0.001:
            return x_opt, y_opt, r_opt
    return x0c, y0c, r0c


def _greedy_insertion(deadline, seed=0, grid_size=80):
    """
    Sequential greedy insertion: add circles one by one, each placed at the center
    of the largest empty circle, then locally optimize its position.
    """
    rng = np.random.RandomState(seed)
    xy = np.zeros((0, 2))
    r = np.zeros(0)

    for k in range(N):
        if time.time() > deadline:
            break

        # Find largest empty circle
        cx, cy, cr = _largest_empty_circle(xy, r, grid_size=grid_size)

        # Local optimization of the new circle
        x_opt, y_opt, r_opt = _local_optimize_new_circle(xy, r, cx, cy, cr, deadline, n_iter=100)

        if r_opt > 0.001:
            xy = np.vstack([xy, [[x_opt, y_opt]]])
            r = np.append(r, r_opt)
        else:
            # Fallback: place at center with tiny radius
            xy = np.vstack([xy, [[0.5, 0.5]]])
            r = np.append(r, 0.005)

    return xy, r


def _refine_all(xy, r, deadline, grid_size=60, max_sweeps=5):
    """
    Refine all circles: iterate over each circle, optimize its position given others fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            x_opt, y_opt, r_opt = _local_optimize_new_circle(
                others_xy, others_r, xy[i, 0], xy[i, 1], r[i], deadline, n_iter=80)

            if r_opt > r[i] + 1e-8:
                xy[i, 0] = x_opt
                xy[i, 1] = y_opt
                r[i] = r_opt
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Multiple seeds for greedy insertion
    for seed in range(6):
        if time.time() > deadline:
            break

        xy, r = _greedy_insertion(deadline, seed=seed, grid_size=80)

        # Refine all circles
        out_xy, out_r, total, valid = _refine_all(xy, r, deadline, grid_size=60, max_sweeps=4)

        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Final high-resolution polish if we have time
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        final_xy, final_r, final_total, final_valid = _refine_all(
            best_xy, best_r, deadline, grid_size=80, max_sweeps=3)
        if final_valid and final_total > best_total:
            best_total = final_total
            best_xy = final_xy
            best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        total, valid = _eval_packing(result[:, :2], result[:, 2])
        if valid:
            return result
        else:
            result[:, 2] = 0.05
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
