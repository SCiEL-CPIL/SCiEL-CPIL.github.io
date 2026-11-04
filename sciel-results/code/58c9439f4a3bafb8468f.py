"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill

# Adaptive smooth-max: start coarse, tighten over iterations
_SMOOTH_EPS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]


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


def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
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

        # Resolve boundary violations by clamping
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
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


def _objective_transformed(params, W, H, eps=1e-4, overlap_weight=200.0):
    """
    Objective in transformed variables: u, v in [0,1], r > 0.
    x = r + u * (W - 2r)
    y = r + v * (H - 2r)
    """
    n = N
    u = params[0::3]
    v = params[1::3]
    r = params[2::3]
    
    x = r + u * (W - 2 * r)
    y = r + v * (H - 2 * r)
    
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
    
    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]
    
    v_vec = r_sum - dist_ij
    
    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax
        
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
    obj += overlap_weight * np.sum(smax_vec ** 2)
    
    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    
    nx_ij = dx_ij / dist_ij
    ny_ij = dy_ij / dist_ij
    
    grad_u_i = factor_vec * nx_ij * (W - 2 * r[iu])
    grad_v_i = factor_vec * ny_ij * (H - 2 * r[iu])
    grad_r_i = factor_vec * (1.0 + nx_ij * (1.0 - 2.0 * u[iu]) + ny_ij * (1.0 - 2.0 * v[iu]))
    
    grad_u_j = -factor_vec * nx_ij * (W - 2 * r[ju])
    grad_v_j = -factor_vec * ny_ij * (H - 2 * r[ju])
    grad_r_j = factor_vec * (1.0 - nx_ij * (1.0 - 2.0 * u[ju]) - ny_ij * (1.0 - 2.0 * v[ju]))
    
    np.add.at(grad, 3 * iu, grad_u_i)
    np.add.at(grad, 3 * iu + 1, grad_v_i)
    np.add.at(grad, 3 * iu + 2, grad_r_i)
    
    np.add.at(grad, 3 * ju, grad_u_j)
    np.add.at(grad, 3 * ju + 1, grad_v_j)
    np.add.at(grad, 3 * ju + 2, grad_r_j)
    
    return obj, grad

def _transformed_to_circles(params, W, H):
    n = N
    u = params[0::3]
    v = params[1::3]
    r = params[2::3]
    x = r + u * (W - 2 * r)
    y = r + v * (H - 2 * r)
    return np.column_stack([x, y, r])

def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
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
    grad[2::3] = -1.0

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: vectorized
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
    obj += overlap_weight * np.sum(smax_vec ** 2)

    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij

    grad_x_i = factor_vec * nx_ij
    grad_y_i = factor_vec * ny_ij
    grad_x_j = -factor_vec * nx_ij
    grad_y_j = -factor_vec * ny_ij

    np.add.at(grad, 3 * iu, grad_x_i)
    np.add.at(grad, 3 * iu + 1, grad_y_i)
    np.add.at(grad, 3 * iu + 2, factor_vec)
    np.add.at(grad, 3 * ju, grad_x_j)
    np.add.at(grad, 3 * ju + 1, grad_y_j)
    np.add.at(grad, 3 * ju + 2, factor_vec)

    return obj, grad


def _local_optimize_transformed(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize using transformed variables."""
    # Convert circles to transformed params
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # u = (x - r) / (W - 2r)
    # v = (y - r) / (H - 2r)
    # Handle division by zero if W == 2r
    denom_x = W - 2 * r
    denom_y = H - 2 * r
    
    # Clip r to be slightly less than W/2, H/2 to avoid singularities
    r_safe = np.clip(r, 1e-6, min(W, H) / 2 - 1e-6)
    denom_x = W - 2 * r_safe
    denom_y = H - 2 * r_safe
    
    u = (x - r_safe) / denom_x
    v = (y - r_safe) / denom_y
    
    # Clip u, v to [0, 1]
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    params = np.column_stack([u, v, r_safe]).flatten()
    
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, r_max - 1e-6))

    result = minimize(
        _objective_transformed, params, args=(W, H, 1e-4, 200.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    
    # Convert back to circles
    circles_opt = _transformed_to_circles(result.x, W, H)
    return circles_opt

def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _objective_and_grad, params, args=(W, H, 1e-4, 200.0, 200.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _polish_transformed(circles, W, H, max_iter=8000):
    """Multi-pass polish with progressively tighter smooth-max eps using transformed variables."""
    # Initialize transformed params
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    r_safe = np.clip(r, 1e-6, min(W, H) / 2 - 1e-6)
    denom_x = W - 2 * r_safe
    denom_y = H - 2 * r_safe
    u = (x - r_safe) / denom_x
    v = (y - r_safe) / denom_y
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    params = np.column_stack([u, v, r_safe]).flatten()
    
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, r_max - 1e-6))

    # Progressive refinement: 5 passes with decreasing eps
    eps_schedule = _SMOOTH_EPS
    n_passes = len(eps_schedule)
    iter_per_pass = max(500, max_iter // n_passes)

    for p in range(n_passes):
        eps_val = eps_schedule[p]
        result = minimize(
            _objective_transformed, params, args=(W, H, eps_val, 200.0),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iter_per_pass, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
        )
        params = result.x

    return _transformed_to_circles(params, W, H)

def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish with progressively tighter smooth-max eps."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Progressive refinement: 5 passes with decreasing eps
    eps_schedule = _SMOOTH_EPS
    n_passes = len(eps_schedule)
    iter_per_pass = max(500, max_iter // n_passes)

    for p in range(n_passes):
        eps_val = eps_schedule[p]
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps_val, 200.0, 200.0),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iter_per_pass, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
        )
        params = result.x

    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=12000):
    """Ultra-tight final polish for the best candidate."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Use the tightest eps with very tight tolerances
    result = minimize(
        _objective_and_grad, params, args=(W, H, 1e-7, 500.0, 500.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """Create a hexagonal packing initial configuration."""
    circles = []
    n_cols = 5
    n_rows = 5
    points = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * 1.0 + (0.5 if row % 2 else 0.0)
            y = row * np.sqrt(3) / 2.0
            points.append([x, y])
            
    points = np.array(points[:N])
    
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    if span_x < 1e-6: span_x = 1.0
    if span_y < 1e-6: span_y = 1.0
    
    scale_x = W / span_x
    scale_y = H / span_y
    scale = min(scale_x, scale_y)
    
    points[:, 0] = (points[:, 0] - min_x) * scale + (W - span_x * scale) / 2
    points[:, 1] = (points[:, 1] - min_y) * scale + (H - span_y * scale) / 2
    
    r = 0.5 * scale * 0.95
    
    for p in points:
        circles.append([p[0], p[1], r])
        
    return np.array(circles)

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


def _successive_convex_relaxation(circles, W, H, max_sweeps=20, tol=1e-6):
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

            def _smooth_max_grad(v):
                smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
                gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
                return smax, gsmax

            # Boundary penalties
            boundary_weight = 200.0
            for v_expr, d_x, d_y, d_r in [
                (r - x, -1, 0, 1),
                (r - y, 0, -1, 1),
                (x + r - W, 1, 0, 1),
                (y + r - H, 0, 1, 1),
                (-r, 0, 0, -1),
            ]:
                v = v_expr
                smax, gsmax = _smooth_max_grad(v)
                obj += boundary_weight * np.sum(smax ** 2)
                if d_x != 0:
                    grad[0::3] += 2.0 * boundary_weight * smax * gsmax * d_x
                if d_y != 0:
                    grad[1::3] += 2.0 * boundary_weight * smax * gsmax * d_y
                grad[2::3] += 2.0 * boundary_weight * smax * gsmax * d_r

            # Linearized overlap penalties
            overlap_weight = 200.0
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
                    smax, gsmax = _smooth_max_grad(v)
                    obj += overlap_weight * smax ** 2

                    d_obj = 2.0 * overlap_weight * smax * gsmax
                    grad[3 * i] += d_obj * (-gx_i)
                    grad[3 * i + 1] += d_obj * (-gy_i)
                    grad[3 * i + 2] += d_obj * 1.0
                    grad[3 * j] += d_obj * (gx_i)
                    grad[3 * j + 1] += d_obj * (gy_i)
                    grad[3 * j + 2] += d_obj * 1.0

            return obj, grad

        params = np.column_stack([x, y, r]).flatten()
        result = minimize(
            lin_objective_and_grad, params,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
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
    # Start with hexagonal
    circles = _hexagonal_initialization(W, H)
    best_val = -np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Try local optimization with L-BFGS-B (Transformed)
    circles_opt = _local_optimize_transformed(circles, W, H, max_iter=3000)
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

    # Try successive convex relaxation if time permits
    if time.time() - start < time_budget - 5:
        sc_circles, sc_score = _successive_convex_relaxation(best_circles, W, H, max_sweeps=15)
        sc_circles[:, 0] = np.clip(sc_circles[:, 0], sc_circles[:, 2], W - sc_circles[:, 2])
        sc_circles[:, 1] = np.clip(sc_circles[:, 1], sc_circles[:, 2], H - sc_circles[:, 2])
        sc_circles[:, 2] = np.clip(sc_circles[:, 2], 0.001, min(W, H) / 2)
        sc_circles, valid_sc = _separate_circles(sc_circles, W, H)
        if valid_sc:
            s = np.sum(sc_circles[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = sc_circles.copy()

    # Final polish if time permits (Transformed)
    if time.time() - start < time_budget - 2:
        polished = _polish_transformed(best_circles, W, H, max_iter=8000)
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

    # Try different aspect ratios - focus on near-square and slightly elongated
    aspect_ratios = [0.85, 1.0, 1.15, 1.35, 1.6]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate more time to promising aspect ratios (near square)
        if 0.9 <= ar <= 1.1:
            time_budget = 22.0
        elif 0.8 <= ar <= 1.4:
            time_budget = 16.0
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
            n_restarts = max(1, int(remaining / 4))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1000)
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

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished = _high_precision_polish(best_result, best_W, best_H, max_iter=12000)
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
