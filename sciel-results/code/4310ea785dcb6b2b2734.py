"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 110.0  # seconds, safety margin before 120s kill


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=1000, tol=1e-10):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by shrinking radii if needed
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            # Clamp position to keep circle inside rectangle
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - tol:
                    # Push apart
                    if dist < 1e-12:
                        # Random nudge
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    # Distribute push based on relative sizes
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            # If circle can't fit at current position, shrink it
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _smooth_max_grad(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, gsmax


def _objective_and_grad(params, W, H, boundary_weight=200.0, overlap_weight=200.0, eps=1e-4):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum(r))/dr

    # Boundary penalties (smooth)
    # x - r >= 0  => violation = -(x - r) = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W  => violation = x + r - W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: for each pair, violation = r_i + r_j - dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v, eps)
            obj += overlap_weight * smax ** 2

            factor = 2.0 * overlap_weight * smax * gsmax
            d_obj_dx_i = factor * (-dx / dist)
            d_obj_dy_i = factor * (-dy / dist)
            d_obj_dx_j = factor * (dx / dist)
            d_obj_dy_j = factor * (dy / dist)
            d_obj_dr_i = factor * 1.0
            d_obj_dr_j = factor * 1.0

            grad[3 * i] += d_obj_dx_i
            grad[3 * i + 1] += d_obj_dy_i
            grad[3 * i + 2] += d_obj_dr_i
            grad[3 * j] += d_obj_dx_j
            grad[3 * j + 1] += d_obj_dy_j
            grad[3 * j + 2] += d_obj_dr_j

    return obj, grad


def _local_optimize(circles, W, H, max_iter=2000, xatol=1e-10, fatol=1e-12,
                    boundary_weight=200.0, overlap_weight=200.0):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    def obj_grad(params):
        return _objective_and_grad(params, W, H, boundary_weight, overlap_weight)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-10, 'maxls': 100}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=10000):
    """High-precision polish pass with L-BFGS-B and tight tolerances."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def obj_grad(params):
        return _objective_and_grad(params, W, H, boundary_weight=500.0, overlap_weight=500.0, eps=1e-5)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _hex_initialization(W, H):
    """Create a hexagonal-like initial configuration."""
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * H / total_rows
        for col_j in range(count):
            avg_w = W / 5.0
            offset = 0.5 * avg_w if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * W / 5.0 + offset
            r = 0.35 * min(W / 5.0, H / total_rows)
            circles.append([x, y, r])
    return np.array(circles[:N])


def _random_initialization(W, H, rng):
    """Create a random initial configuration with moderate circles."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _successive_convex_relaxation(circles, W, H, max_sweeps=30, tol=1e-7):
    """
    Successive Convex Relaxation using L-BFGS-B with linearized constraints.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    best_val = -np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    boundary_weight = 200.0
    overlap_weight = 50.0

    for sweep in range(max_sweeps):
        def lin_objective_and_grad(params):
            p = params.reshape(n, 3)
            px = p[:, 0]
            py = p[:, 1]
            pr = p[:, 2]

            obj = -np.sum(pr)
            grad = np.zeros_like(params)
            grad[2::3] = -1.0

            eps = 1e-4

            # Boundary penalties
            for v_expr, d_x, d_y, d_r in [
                (pr - px, -1, 0, 1),
                (pr - py, 0, -1, 1),
                (px + pr - W, 1, 0, 1),
                (py + pr - H, 0, 1, 1),
                (-pr, 0, 0, -1),
            ]:
                v = v_expr
                smax, gsmax = _smooth_max_grad(v, eps)
                obj += boundary_weight * np.sum(smax ** 2)
                if d_x != 0:
                    grad[0::3] += 2.0 * boundary_weight * smax * gsmax * d_x
                if d_y != 0:
                    grad[1::3] += 2.0 * boundary_weight * smax * gsmax * d_y
                grad[2::3] += 2.0 * boundary_weight * smax * gsmax * d_r

            # Linearized overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dx0 = x[i] - x[j]
                    dy0 = y[i] - y[j]
                    dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
                    if dist0 < 1e-10:
                        dist0 = 1e-10
                        dx0 = 1e-10
                        dy0 = 0.0
                    gx_i = dx0 / dist0
                    gy_i = dy0 / dist0

                    # Linearized distance
                    lin_dist = dist0 + gx_i * (px[i] - x[i]) + gy_i * (py[i] - y[i]) \
                                 - gx_i * (px[j] - x[j]) - gy_i * (py[j] - y[j])
                    v = (pr[i] + pr[j]) - lin_dist
                    smax, gsmax = _smooth_max_grad(v, eps)
                    obj += overlap_weight * smax ** 2

                    factor = 2.0 * overlap_weight * smax * gsmax
                    grad[3 * i] += factor * (-gx_i)
                    grad[3 * i + 1] += factor * (-gy_i)
                    grad[3 * i + 2] += factor * 1.0
                    grad[3 * j] += factor * (gx_i)
                    grad[3 * j + 1] += factor * (gy_i)
                    grad[3 * j + 2] += factor * 1.0

            return obj, grad

        params = np.column_stack([x, y, r]).flatten()
        result = minimize(
            lin_objective_and_grad, params,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-11, 'gtol': 1e-9, 'maxls': 100}
        )
        new_params = result.x.reshape(n, 3)
        new_x = new_params[:, 0]
        new_y = new_params[:, 1]
        new_r = new_params[:, 2]

        new_x = np.clip(new_x, new_r, W - new_r)
        new_y = np.clip(new_y, new_r, H - new_r)
        new_r = np.clip(new_r, 0.001, r_max)

        new_val = -np.sum(new_r)
        if new_val < best_val - 1e-8:
            best_val = new_val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        if np.max(np.abs(new_x - x)) < tol and np.max(np.abs(new_y - y)) < tol and np.max(np.abs(new_r - r)) < tol:
            break

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r]), -best_val


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    # Start with grid
    circles = _grid_initialization(W, H)
    best_val = -np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Try hexagonal initialization
    circles_hex = _hex_initialization(W, H)
    hex_val = -np.sum(circles_hex[:, 2])
    if hex_val > best_val:
        best_val = hex_val
        best_circles = circles_hex.copy()

    # Try local optimization with L-BFGS-B
    circles_opt = _local_optimize(circles, W, H, max_iter=2000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)

    # Separate any remaining overlaps
    circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
    if valid_opt:
        s = np.sum(circles_opt[:, 2])
        if s > -best_val:
            best_val = -s
            best_circles = circles_opt.copy()

    # Try hexagonal local optimization
    if time.time() - start < time_budget - 5:
        circles_hex_opt = _local_optimize(circles_hex, W, H, max_iter=2000)
        circles_hex_opt[:, 0] = np.clip(circles_hex_opt[:, 0], circles_hex_opt[:, 2], W - circles_hex_opt[:, 2])
        circles_hex_opt[:, 1] = np.clip(circles_hex_opt[:, 1], circles_hex_opt[:, 2], H - circles_hex_opt[:, 2])
        circles_hex_opt[:, 2] = np.clip(circles_hex_opt[:, 2], 0.001, min(W, H) / 2)
        circles_hex_opt, valid_hex = _separate_circles(circles_hex_opt, W, H)
        if valid_hex:
            s = np.sum(circles_hex_opt[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = circles_hex_opt.copy()

    # Try successive convex relaxation if time permits
    if time.time() - start < time_budget - 5:
        sc_circles, sc_score = _successive_convex_relaxation(best_circles, W, H, max_sweeps=25)
        sc_circles[:, 0] = np.clip(sc_circles[:, 0], sc_circles[:, 2], W - sc_circles[:, 2])
        sc_circles[:, 1] = np.clip(sc_circles[:, 1], sc_circles[:, 2], H - sc_circles[:, 2])
        sc_circles[:, 2] = np.clip(sc_circles[:, 2], 0.001, min(W, H) / 2)
        sc_circles, valid_sc = _separate_circles(sc_circles, W, H)
        if valid_sc:
            s = np.sum(sc_circles[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = sc_circles.copy()

    # Final polish if time permits
    if time.time() - start < time_budget - 2:
        polished = _polish(best_circles, W, H, max_iter=5000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = polished.copy()

    return best_circles, -best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios - broader search including elongated
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate more time to promising aspect ratios (near square)
        if 0.9 <= ar <= 1.1:
            time_budget = 20.0
        elif 0.8 <= ar <= 1.3:
            time_budget = 15.0
        else:
            time_budget = 12.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Try random restarts if time permits
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining > 5:
            n_restarts = max(1, int(remaining / 3))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1500)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
                circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
                if valid_rand:
                    s = np.sum(circ_opt[:, 2])
                    if s > best_score:
                        best_score = s
                        best_result = circ_opt.copy()
                        best_W, best_H = W, H

    # Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished = _polish(best_result, best_W, best_H, max_iter=12000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Final fallback: simple grid in square
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, best_W, best_H):
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
# EVOLVE-BLOCK-END
