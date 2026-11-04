"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
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
    return np.sum(r), True


def _hex_init():
    """Hexagonal packing initialization."""
    positions = []
    # Hex grid: rows offset by half spacing
    spacing = 0.18
    row = 0
    while len(positions) < N:
        offset = 0.0 if row % 2 == 0 else spacing * 0.5
        col = 0
        while len(positions) < N:
            x = 0.09 + col * spacing + offset
            y = 0.09 + row * spacing * 0.866
            if x + 0.09 <= 1.0 and y + 0.09 <= 1.0:
                positions.append([x, y, 0.09])
            col += 1
        row += 1
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.05])
    return np.array(positions[:N])


def _ring_init():
    """Concentric rings initialization."""
    positions = [[0.5, 0.5, 0.12]]
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = i * np.pi / 3
        x = 0.5 + r1 * np.cos(angle)
        y = 0.5 + r1 * np.sin(angle)
        positions.append([x, y, 0.08])
    # Ring 2: 12 circles
    r2 = 0.42
    for i in range(12):
        angle = i * 2 * np.pi / 12 + np.pi / 12
        x = 0.5 + r2 * np.cos(angle)
        y = 0.5 + r2 * np.sin(angle)
        positions.append([x, y, 0.06])
    # Pad
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.03])
    return np.array(positions[:N])


def _jittered_grid_init(seed=42):
    """Jittered grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                base_x = (i + 0.5) / side
                base_y = (j + 0.5) / side
                x = base_x + rng.uniform(-0.1, 0.1) / side
                y = base_y + rng.uniform(-0.1, 0.1) / side
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                positions.append([x, y, 0.5 / side - 0.02])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.03])
    return np.array(positions[:N])


def _make_constraints(n):
    """Create constraint functions for SLSQP: each must be >= 0."""
    constraints = []
    # Wall constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    for i in range(n):
        # x_i >= r_i  =>  x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        # 1 - x_i >= r_i  =>  1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2]})
        # y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        # 1 - y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2]})
    # Pairwise: dist_ij >= r_i + r_j  =>  dist_ij - r_i - r_j >= 0
    # We use a smooth approximation: dist^2 - (r_i+r_j)^2 >= 0 is not equivalent
    # Instead use: sqrt(dx^2+dy^2) - r_i - r_j >= 0
    # For SLSQP, let's use the squared form which is equivalent when both sides positive:
    # Actually sqrt form is fine for SLSQP
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt(
                    (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2
                ) - p[3*i+2] - p[3*j+2]
            })
    return constraints


def _slsqp_optimize(xy, r, deadline, max_iter=300):
    """
    Constrained SLSQP optimization: maximize sum(r) subject to wall and non-overlap.
    """
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.01, 0.45))  # r

    # Objective: minimize -sum(r)
    def objective(p):
        return -np.sum(p[2::3])

    # Constraints
    constraints = _make_constraints(n)

    best_params = params.copy()
    best_total, best_valid = _eval_packing(xy, r)
    if not best_valid:
        best_total = 0.0

    try:
        result = minimize(
            objective,
            params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
        )

        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[2::3], 0.01)

        # Clip to ensure wall validity
        for i in range(n):
            opt_r[i] = min(opt_r[i], opt_xy[i, 0], 1.0 - opt_xy[i, 0], opt_xy[i, 1], 1.0 - opt_xy[i, 1])
            opt_r[i] = max(opt_r[i], 0.01)
            opt_xy[i, 0] = np.clip(opt_xy[i, 0], opt_r[i], 1.0 - opt_r[i])
            opt_xy[i, 1] = np.clip(opt_xy[i, 1], opt_r[i], 1.0 - opt_r[i])

        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_params = opt_params.copy()

    except Exception:
        pass

    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[2::3], 0.01)
    return final_xy, final_r, best_total


def _pairwise_local_search(xy, r, deadline, max_sweeps=50):
    """
    Local search: for each pair of circles, try to increase their radii by
    moving them apart. Only accept moves that maintain validity.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    if not best_valid:
        best_total = 0.0
    best_xy = xy.copy()
    best_r = r.copy()

    rng = np.random.RandomState(123)

    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        # Try random pairs
        for _ in range(n * (n - 1) // 2):
            if time.time() > deadline:
                break
            i, j = rng.choice(n, 2, replace=False)
            i, j = int(i), int(j)

            # Current distance
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue

            # Direction from j to i
            ux, uy = dx / dist, dy / dist

            # Try to push i away from j and j away from i
            # Increase both radii by delta, move centers apart by delta
            # New: r_i' = r_i + delta, r_j' = r_j + delta
            # Move i by delta in direction u, j by delta in direction -u
            # This maintains dist' = dist + 2*delta >= r_i' + r_j' = r_i + r_j + 2*delta
            # So it's always valid for the pair! But need to check walls and other circles.

            # Max delta limited by walls
            # i moves by delta*(ux, uy): new pos = (x_i + delta*ux, y_i + delta*uy)
            # Need: x_i + delta*ux >= r_i + delta, and 1 - x_i - delta*ux >= r_i + delta
            # => delta*ux - delta >= -(x_i - r_i)  =>  delta*(ux - 1) >= -(x_i - r_i)
            # This is getting complex. Let's just try small deltas.

            max_delta = 0.02
            delta = max_delta
            while delta > 1e-5:
                new_xi = xy[i, 0] + delta * ux
                new_yi = xy[i, 1] + delta * uy
                new_xj = xy[j, 0] - delta * ux
                new_yj = xy[j, 1] - delta * uy
                new_ri = r[i] + delta
                new_rj = r[j] + delta

                # Check walls
                if (new_xi - new_ri < -1e-9 or 1 - new_xi - new_ri < -1e-9 or
                    new_yi - new_ri < -1e-9 or 1 - new_yi - new_ri < -1e-9 or
                    new_xj - new_rj < -1e-9 or 1 - new_xj - new_rj < -1e-9 or
                    new_yj - new_rj < -1e-9 or 1 - new_yj - new_rj < -1e-9):
                    delta *= 0.5
                    continue

                # Check all other circles
                valid = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Check circle k vs new i
                    dx_ki = new_xi - xy[k, 0]
                    dy_ki = new_yi - xy[k, 1]
                    if np.sqrt(dx_ki**2 + dy_ki**2) < new_ri + r[k] - 1e-9:
                        valid = False
                        break
                    # Check circle k vs new j
                    dx_kj = new_xj - xy[k, 0]
                    dy_kj = new_yj - xy[k, 1]
                    if np.sqrt(dx_kj**2 + dy_kj**2) < new_rj + r[k] - 1e-9:
                        valid = False
                        break

                if valid:
                    xy[i, 0], xy[i, 1] = new_xi, new_yi
                    xy[j, 0], xy[j, 1] = new_xj, new_yj
                    r[i], r[j] = new_ri, new_rj
                    improved = True
                    break
                else:
                    delta *= 0.5

        # Evaluate after sweep
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total


def _single_circle_max(xy, r, idx, deadline):
    """
    For circle idx, find the maximum radius and best position given others fixed.
    Uses a fine grid search around current position.
    """
    n = len(xy)
    x0, y0 = xy[idx, 0], xy[idx, 1]
    r0 = r[idx]

    # Search in a local region
    search_r = 0.15
    grid_n = 40
    best_r = 0.0
    best_x, best_y = x0, y0

    for gi in range(grid_n):
        if time.time() > deadline:
            break
        for gj in range(grid_n):
            x = x0 - search_r + 2 * search_r * gi / (grid_n - 1)
            y = y0 - search_r + 2 * search_r * gj / (grid_n - 1)

            # Wall clearance
            wall_clear = min(x, 1 - x, y, 1 - y)
            if wall_clear <= 0:
                continue

            # Circle clearance
            min_clear = wall_clear
            for k in range(n):
                if k == idx:
                    continue
                dx = x - xy[k, 0]
                dy = y - xy[k, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                clear = dist - r[k]
                if clear < min_clear:
                    min_clear = clear
                    if min_clear <= 0:
                        break

            if min_clear > best_r:
                best_r = min_clear
                best_x, best_y = x, y

    if best_r > r0 + 1e-10:
        xy[idx, 0], xy[idx, 1] = best_x, best_y
        r[idx] = best_r
        return True
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0

    # Multiple initializations
    inits = [
        _hex_init(),
        _ring_init(),
        _jittered_grid_init(42),
        _jittered_grid_init(137),
        _jittered_grid_init(999),
    ]

    for init_arr in inits:
        if time.time() > deadline - 20.0:
            break

        xy = init_arr[:, :2].copy()
        r = init_arr[:, 2].copy()

        # Ensure initial validity (shrink if needed)
        total, valid = _eval_packing(xy, r)
        if not valid:
            # Shrink all radii until valid
            for _ in range(50):
                r *= 0.9
                total, valid = _eval_packing(xy, r)
                if valid:
                    break
            if not valid:
                continue

        # Step 1: SLSQP constrained optimization
        if time.time() < deadline - 15.0:
            out_xy, out_r, out_total = _slsqp_optimize(xy, r, deadline - 15.0, max_iter=200)
            if out_total > best_total:
                best_total = out_total
                best_xy = out_xy.copy()
                best_r = out_r.copy()

        # Step 2: Pairwise local search
        if time.time() < deadline - 10.0:
            ps_xy, ps_r, ps_total = _pairwise_local_search(best_xy, best_r, deadline - 10.0, max_sweeps=30)
            if ps_total > best_total:
                best_total = ps_total
                best_xy = ps_xy.copy()
                best_r = ps_r.copy()

        # Step 3: Single circle maximization (multiple passes)
        if time.time() < deadline - 5.0:
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                any_improved = False
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    if _single_circle_max(best_xy, best_r, i, deadline - 5.0):
                        any_improved = True
                if not any_improved:
                    break
            # Re-evaluate
            total, valid = _eval_packing(best_xy, best_r)
            if valid:
                best_total = total

    # If no valid result, fallback
    if best_xy is None or best_total <= 0:
        side = int(np.ceil(np.sqrt(N)))
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
