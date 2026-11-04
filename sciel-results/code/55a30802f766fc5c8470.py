"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


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
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
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


def _smooth_max(v, eps=1e-4):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v)) with adaptive eps."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, grad_smax


def _objective_and_grad(params, W, H, eps_boundary=1e-4, eps_overlap=1e-4):
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
    boundary_weight = 200.0

    # x - r >= 0  => violation = -(x - r) = r - x
    v = r - x
    smax, gsmax = _smooth_max(v, eps_boundary)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max(v, eps_boundary)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W  => violation = x + r - W
    v = x + r - W
    smax, gsmax = _smooth_max(v, eps_boundary)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max(v, eps_boundary)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max(v, eps_boundary)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: vectorized computation
    overlap_weight = 200.0
    # Compute all pairwise distances
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    # Avoid division by zero
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

    # Only use upper triangle (i < j)
    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _smooth_max(v_vec, eps_overlap)
    obj += overlap_weight * np.sum(smax_vec ** 2)

    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij

    # Accumulate gradients using np.add.at for repeated indices
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


def _local_optimize(circles, W, H, max_iter=3000, eps_boundary=1e-4, eps_overlap=1e-4):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    def _obj(params, W, H):
        return _objective_and_grad(params, W, H, eps_boundary, eps_overlap)

    result = minimize(
        _obj, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """
    Three-pass polish: coarse, medium, fine.
    """
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse
    result = minimize(
        lambda p, W, H: _objective_and_grad(p, W, H, 1e-3, 1e-3), params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        lambda p, W, H: _objective_and_grad(p, W, H, 1e-5, 1e-5), params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 200}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        lambda p, W, H: _objective_and_grad(p, W, H, 1e-6, 1e-6), params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 400}
    )
    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=10000):
    """
    High-precision final polish with very tight tolerances and adaptive eps.
    """
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-8, W - 1e-8))
        bounds.append((1e-8, H - 1e-8))
        bounds.append((1e-8, r_max))

    # Multi-stage with decreasing eps
    stages = [
        (1e-4, max_iter // 5, 1e-12, 1e-10),
        (1e-6, max_iter // 3, 1e-14, 1e-12),
        (1e-8, max_iter, 1e-16, 1e-14),
    ]

    for eps, iters, ftol, gtol in stages:
        def _obj(p, W, H, eps=eps):
            return _objective_and_grad(p, W, H, eps, eps)

        result = minimize(
            _obj, params, args=(W, H),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iters, 'ftol': ftol, 'gtol': gtol, 'maxls': 500}
        )
        params = result.x

    return params.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """Create a hexagonal (honeycomb) initial configuration."""
    circles = []
    # Start with a reasonable radius estimate
    # For hexagonal packing in a rectangle, approximate density
    area = W * H
    # Hexagonal packing density ~ 0.9069 for infinite packing
    # For 21 circles, estimate average radius
    avg_r = np.sqrt(area / (N * np.pi)) * 0.55  # conservative estimate
    
    # Build hexagonal grid
    row_spacing = avg_r * np.sqrt(3)
    col_spacing = avg_r * 2.0
    offset = col_spacing / 2.0
    
    row = 0
    while len(circles) < N:
        col = 0
        while len(circles) < N:
            if row % 2 == 0:
                cx = col * col_spacing + avg_r
            else:
                cx = col * col_spacing + offset + avg_r
            
            cy = row * row_spacing + avg_r
            
            if cx + avg_r <= W + 1e-6 and cy + avg_r <= H + 1e-6:
                if cx - avg_r >= -1e-6 and cy - avg_r >= -1e-6:
                    circles.append([cx, cy, avg_r])
            
            col += 1
        row += 1
    
    if len(circles) < N:
        # Fill remaining with smaller circles in gaps
        for i in range(len(circles), N):
            r = avg_r * 0.3
            x = np.random.uniform(r, W - r)
            y = np.random.uniform(r, H - r)
            circles.append([x, y, r])
    
    return np.array(circles[:N])


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


def _force_directed_layout(W, H, max_iter=5000, time_budget=15.0):
    """
    Force-directed layout: circles repel each other and are attracted to center.
    Radii grow when circles are well-separated.
    """
    start = time.time()
    n = N
    x = np.random.uniform(0.1, W - 0.1, n)
    y = np.random.uniform(0.1, H - 0.1, n)
    r = np.full(n, min(W, H) / (2 * np.sqrt(N)) * 0.5)
    r_max = min(W, H) / 2

    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    iu, ju = np.triu_indices(n, k=1)

    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)

        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                min_dist = r[i] + r[j]
                
                if dist < min_dist * 1.5:
                    # Repulsive force
                    force = 1.0 / (dist * dist + 1e-8)
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                else:
                    # Attraction to allow radius growth
                    force = 0.1 * (min_dist - dist) / (min_dist + 1e-8)
                    fx[i] -= force * dx / dist
                    fy[i] -= force * dy / dist
                    fx[j] += force * dx / dist
                    fy[j] += force * dy / dist

                # Radius growth force when well-separated
                if dist > min_dist * 1.2:
                    growth = 0.01 * (dist - min_dist) / (dist + 1e-8)
                    fr[i] += growth
                    fr[j] += growth

        # Attraction to center
        cx_center = W / 2
        cy_center = H / 2
        for i in range(n):
            fx[i] += 0.05 * (cx_center - x[i])
            fy[i] += 0.05 * (cy_center - y[i])

        # Update positions
        dt = 0.01
        x += dt * fx
        y += dt * fy
        r += dt * fr

        # Project onto feasible set
        r = np.clip(r, 1e-4, r_max)
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)

        # Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist ** 2:
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

        # Re-clamp
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        val = np.sum(r)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    return np.column_stack([best_x, best_y, best_r])


def _gradient_flow(circles, W, H, dt=0.01, max_steps=5000, time_budget=10.0):
    """
    Semi-implicit Euler gradient flow with projection onto feasible set.
    """
    start = time.time()
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    r_max = min(W, H) / 2

    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    iu, ju = np.triu_indices(n, k=1)

    for step in range(max_steps):
        if time.time() - start > time_budget:
            break

        params = np.column_stack([x, y, r]).flatten()
        obj, grad = _objective_and_grad(params, W, H)

        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        effective_dt = dt / (1.0 + grad_norm * 0.1)

        new_params = params - effective_dt * grad
        new_x = new_params[0::3]
        new_y = new_params[1::3]
        new_r = new_params[2::3]

        new_r = np.clip(new_r, 1e-4, r_max)
        new_x = np.clip(new_x, new_r, W - new_r)
        new_y = np.clip(new_y, new_r, H - new_r)

        for i in range(n):
            for j in range(i + 1, n):
                dx = new_x[i] - new_x[j]
                dy = new_y[i] - new_y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = new_r[i] + new_r[j]
                if dist_sq < min_dist ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = new_r[i] + new_r[j]
                    wi = new_r[i] / total_r if total_r > 0 else 0.5
                    wj = new_r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    new_x[i] += wi * push * dx / dist
                    new_y[i] += wi * push * dy / dist
                    new_x[j] -= wj * push * dx / dist
                    new_y[j] -= wj * push * dy / dist

        for i in range(n):
            max_r_x = min(new_x[i], W - new_x[i])
            max_r_y = min(new_y[i], H - new_y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if new_r[i] > max_r:
                new_r[i] = max_r
            new_x[i] = np.clip(new_x[i], new_r[i], W - new_r[i])
            new_y[i] = np.clip(new_y[i], new_r[i], H - new_r[i])

        val = np.sum(new_r)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r])


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    best_val = -np.inf
    best_circles = None

    # Try multiple initializations
    inits = []
    
    # Grid initialization
    circ_grid = _grid_initialization(W, H)
    inits.append(('grid', circ_grid))
    
    # Hexagonal initialization
    circ_hex = _hexagonal_initialization(W, H)
    inits.append(('hex', circ_hex))
    
    # Force-directed layout
    if time_budget > 15:
        circ_fd = _force_directed_layout(W, H, max_iter=3000, time_budget=time_budget * 0.3)
        inits.append(('fd', circ_fd))
    
    rng = np.random.default_rng(123)
    circ_rand = _random_initialization(W, H, rng)
    inits.append(('rand', circ_rand))

    for name, circ_init in inits:
        if time.time() - start > time_budget - 5:
            break
        
        # Local optimization
        circ_opt = _local_optimize(circ_init, W, H, max_iter=2000)
        circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
        circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
        circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
        
        circ_opt, valid_opt = _separate_circles(circ_opt, W, H)
        if valid_opt:
            s = np.sum(circ_opt[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circ_opt.copy()
    
    # Gradient flow on best
    if best_circles is not None and time.time() - start < time_budget - 8:
        gf_circles = _gradient_flow(best_circles, W, H, dt=0.005, max_steps=3000, time_budget=time_budget - 12)
        gf_circles[:, 0] = np.clip(gf_circles[:, 0], gf_circles[:, 2], W - gf_circles[:, 2])
        gf_circles[:, 1] = np.clip(gf_circles[:, 1], gf_circles[:, 2], H - gf_circles[:, 2])
        gf_circles[:, 2] = np.clip(gf_circles[:, 2], 0.001, min(W, H) / 2)
        gf_circles, valid_gf = _separate_circles(gf_circles, W, H)
        if valid_gf:
            s = np.sum(gf_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = gf_circles.copy()
    
    # Polish
    if best_circles is not None and time.time() - start < time_budget - 3:
        polished = _polish(best_circles, W, H, max_iter=4000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_val:
                best_val = s
                best_circles = polished.copy()

    if best_circles is None:
        best_circles = _grid_initialization(W, H)
        best_val = np.sum(best_circles[:, 2])

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2, 2.8]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Allocate time based on aspect ratio
        if 0.9 <= ar <= 1.1:
            time_budget = 15.0
        elif 0.7 <= ar <= 1.3:
            time_budget = 12.0
        else:
            time_budget = 8.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Random restarts if time permits
        remaining = DEADLINE - 15 - (time.time() - start_time)
        if remaining > 4:
            n_restarts = max(1, int(remaining / 4))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 15:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1200)
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

    # High-precision final polish
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _high_precision_polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Final fallback
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

    # Ensure validity
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
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
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


def _objective_and_grad(params, W, H):
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
    eps = 1e-4
    boundary_weight = 200.0

    def _smooth_max_grad(v):
        """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = -(x - r) = r - x
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

    # x + r <= W  => violation = x + r - W
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

    # Overlap penalties: vectorized computation
    overlap_weight = 200.0
    # Compute all pairwise distances
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    # Avoid division by zero
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
    
    # Only use upper triangle (i < j)
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
    
    # Accumulate gradients using np.add.at for repeated indices
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


def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """
    Three-pass polish: coarse, medium, fine.
    """
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-13, 'gtol': 1e-10, 'maxls': 200}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 400}
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


def _gradient_flow(circles, W, H, dt=0.01, max_steps=5000, time_budget=10.0):
    """
    Semi-implicit Euler gradient flow with projection onto feasible set.
    The velocity is the negative gradient of the penalty objective (i.e., gradient of sum(r)).
    After each step, project onto the feasible set (clamp to rectangle, resolve overlaps).
    """
    start = time.time()
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    r_max = min(W, H) / 2

    best_val = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Precompute indices for vectorized distance computation
    iu, ju = np.triu_indices(n, k=1)

    for step in range(max_steps):
        if time.time() - start > time_budget:
            break

        # Compute gradient of -sum(r) + penalties (i.e., the negative of the desired velocity)
        # We want to maximize sum(r), so we move in the direction of +grad(sum(r)) subject to constraints.
        # Equivalently, minimize -sum(r) + penalties. The gradient of the objective points in the direction of steepest increase.
        # We want to move AGAINST the gradient of the objective (i.e., in the direction of -grad(obj)).
        # But since obj = -sum(r) + penalties, -grad(obj) = grad(sum(r)) - grad(penalties).
        # For simplicity, we use the analytical gradient from _objective_and_grad and move in the negative direction.

        params = np.column_stack([x, y, r]).flatten()
        obj, grad = _objective_and_grad(params, W, H)

        # Semi-implicit Euler: x_{k+1} = x_k - dt * grad(obj)
        # But we also want to increase r, so the -1.0 in grad[2::3] from -sum(r) already handles that.
        # Actually, we want to maximize sum(r), so we should move in the direction that increases r.
        # The gradient of the objective (which includes -sum(r)) has -1.0 for r components.
        # So -grad(obj) has +1.0 for r components, which is what we want.

        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        # Scale step size to avoid large jumps
        effective_dt = dt / (1.0 + grad_norm * 0.1)

        new_params = params - effective_dt * grad
        new_x = new_params[0::3]
        new_y = new_params[1::3]
        new_r = new_params[2::3]

        # Project onto feasible set: clamp positions and radii
        new_r = np.clip(new_r, 1e-4, r_max)
        new_x = np.clip(new_x, new_r, W - new_r)
        new_y = np.clip(new_y, new_r, H - new_r)

        # Resolve overlaps using separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = new_x[i] - new_x[j]
                dy = new_y[i] - new_y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = new_r[i] + new_r[j]
                if dist_sq < min_dist ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = new_r[i] + new_r[j]
                    wi = new_r[i] / total_r if total_r > 0 else 0.5
                    wj = new_r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    new_x[i] += wi * push * dx / dist
                    new_y[i] += wi * push * dy / dist
                    new_x[j] -= wj * push * dx / dist
                    new_y[j] -= wj * push * dy / dist

        # Re-clamp after separation
        for i in range(n):
            max_r_x = min(new_x[i], W - new_x[i])
            max_r_y = min(new_y[i], H - new_y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if new_r[i] > max_r:
                new_r[i] = max_r
            new_x[i] = np.clip(new_x[i], new_r[i], W - new_r[i])
            new_y[i] = np.clip(new_y[i], new_r[i], H - new_r[i])

        # Check improvement
        val = np.sum(new_r)
        if val > best_val:
            best_val = val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r])


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    # Start with grid
    circles = _grid_initialization(W, H)
    best_val = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Try local optimization with L-BFGS-B
    circles_opt = _local_optimize(circles, W, H, max_iter=2000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)

    # Separate any remaining overlaps
    circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
    if valid_opt:
        s = np.sum(circles_opt[:, 2])
        if s > best_val:
            best_val = s
            best_circles = circles_opt.copy()

    # Try gradient flow if time permits
    if time.time() - start < time_budget - 5:
        gf_circles = _gradient_flow(best_circles, W, H, dt=0.005, max_steps=3000, time_budget=time_budget - 10)
        gf_circles[:, 0] = np.clip(gf_circles[:, 0], gf_circles[:, 2], W - gf_circles[:, 2])
        gf_circles[:, 1] = np.clip(gf_circles[:, 1], gf_circles[:, 2], H - gf_circles[:, 2])
        gf_circles[:, 2] = np.clip(gf_circles[:, 2], 0.001, min(W, H) / 2)
        gf_circles, valid_gf = _separate_circles(gf_circles, W, H)
        if valid_gf:
            s = np.sum(gf_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = gf_circles.copy()

    # Final polish if time permits
    if time.time() - start < time_budget - 2:
        polished = _polish(best_circles, W, H, max_iter=4000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_val:
                best_val = s
                best_circles = polished.copy()

    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios - broader search including elongated
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2, 2.8]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate more time to promising aspect ratios (near square)
        if 0.9 <= ar <= 1.1:
            time_budget = 18.0
        elif 0.7 <= ar <= 1.3:
            time_budget = 14.0
        else:
            time_budget = 10.0
        
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
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
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
