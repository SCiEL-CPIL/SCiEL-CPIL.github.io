"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=3000, tol=1e-12):
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


def _soft_max(v, scale):
    """Soft max of v and 0 using log-sum-exp: scale * log(1 + exp(v/scale))."""
    v_safe = np.clip(v / scale, -50, 50)
    smax = scale * np.log1p(np.exp(v_safe))
    sig = 1.0 / (1.0 + np.exp(-v_safe))
    return smax, sig


def _objective_and_grad(params, W, H, eps=1e-4, boundary_weight=200.0, overlap_weight=200.0):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Uses log-sum-exp based smooth max for better gradient behavior.
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

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _soft_max(v, eps)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _soft_max(v, eps)
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
    smax_vec, gsmax_vec = _soft_max(v_vec, eps)
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


def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B with analytical gradients to locally optimize. Two-stage: coarse then fine."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Stage 1: coarser eps for faster convergence
    result = minimize(
        _objective_and_grad, params, args=(W, H, 5e-4, 150.0, 150.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200}
    )
    params = result.x

    # Stage 2: tighter eps for precision
    result = minimize(
        _objective_and_grad, params, args=(W, H, 1e-5, 400.0, 400.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish with progressively tighter smooth-max eps and increasing penalty."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Progressive refinement: 6 passes with decreasing eps and increasing weight
    eps_schedule = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    weight_schedule = [200.0, 300.0, 400.0, 500.0, 600.0, 800.0]
    n_passes = len(eps_schedule)
    iter_per_pass = max(400, max_iter // n_passes)

    for p in range(n_passes):
        eps_val = eps_schedule[p]
        w_val = weight_schedule[p]
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps_val, w_val, w_val),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iter_per_pass, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300}
        )
        params = result.x

    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=12000):
    """Ultra-tight final polish for the best candidate with very small eps."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Four very tight passes with tiny eps and escalating penalties
    for eps_val, w_val in [(2e-5, 500.0), (1e-5, 800.0), (5e-6, 1200.0), (2e-6, 2000.0)]:
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps_val, w_val, w_val),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter // 4, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 400}
        )
        params = result.x

    return result.x.reshape(N, 3)


def _radius_equalization(circles, W, H, max_rounds=200, time_budget=8.0):
    """
    Local search that tries to equalize radii by transferring from large to small circles.
    For a fixed total area constraint, equal radii maximize sum when packing is tight.
    """
    start = time.time()
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    best_sum = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())

    for round_num in range(max_rounds):
        if time.time() - start > time_budget:
            break

        improved = False
        # Sort by radius: try to grow the smallest, shrink the largest
        sorted_idx = np.argsort(r)
        small_i = sorted_idx[0]
        large_j = sorted_idx[-1]

        # Try to transfer a small amount of radius from large_j to small_i
        transfer = (r[large_j] - r[small_i]) * 0.05
        if transfer < 1e-6:
            continue

        new_r_small = r[small_i] + transfer
        new_r_large = r[large_j] - transfer

        if new_r_large < 1e-4:
            continue

        # Check if small_i can accommodate the new radius at its current position
        max_r_small = _max_radius_for_circle(small_i, x, y, r, W, H)
        if new_r_small <= max_r_small + 1e-9:
            r[small_i] = new_r_small
            r[large_j] = new_r_large
            # Clamp positions
            x[small_i] = np.clip(x[small_i], r[small_i], W - r[small_i])
            y[small_i] = np.clip(y[small_i], r[small_i], H - r[small_i])
            x[large_j] = np.clip(x[large_j], r[large_j], W - r[large_j])
            y[large_j] = np.clip(y[large_j], r[large_j], H - r[large_j])
            improved = True

        if not improved:
            # Try moving small_i to a better position
            for dx, dy in [(0.002, 0), (-0.002, 0), (0, 0.002), (0, -0.002),
                           (0.0015, 0.0015), (-0.0015, 0.0015),
                           (0.0015, -0.0015), (-0.0015, -0.0015)]:
                nx, ny = x[small_i] + dx, y[small_i] + dy
                if nx < r[small_i] or nx > W - r[small_i] or ny < r[small_i] or ny > H - r[small_i]:
                    continue
                temp_x, temp_y = x.copy(), y.copy()
                temp_x[small_i], temp_y[small_i] = nx, ny
                new_max_r = _max_radius_for_circle(small_i, temp_x, temp_y, r, W, H)
                if new_max_r > r[small_i] + 1e-7:
                    x[small_i], y[small_i] = nx, ny
                    grow = min(new_max_r - r[small_i], (r[large_j] - r[small_i]) * 0.03)
                    if grow > 1e-6 and r[large_j] - grow > 1e-4:
                        r[small_i] += grow
                        r[large_j] -= grow
                        x[small_i] = np.clip(x[small_i], r[small_i], W - r[small_i])
                        y[small_i] = np.clip(y[small_i], r[small_i], H - r[small_i])
                        x[large_j] = np.clip(x[large_j], r[large_j], W - r[large_j])
                        y[large_j] = np.clip(y[large_j], r[large_j], H - r[large_j])
                        improved = True
                        break

        current_sum = np.sum(r)
        if current_sum > best_sum + 1e-10:
            best_sum = current_sum
            best_state = (x.copy(), y.copy(), r.copy())
        elif current_sum < best_sum - 1e-6:
            x, y, r = best_state

    result = np.column_stack([best_state[0], best_state[1], best_state[2]])
    return result


def _max_radius_for_circle(i, x, y, r, W, H):
    """Compute the maximum radius circle i can have given current positions of all circles."""
    max_r = min(x[i], W - x[i], y[i], H - y[i])
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - r[j])
    return max(1e-6, max_r)


def _local_search_radius_increase(circles, W, H, max_rounds=50, time_budget=10.0):
    """
    Structural change: Direct local search that tries to increase individual radii.
    """
    start = time.time()
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    best_sum = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    rng = np.random.default_rng(123)
    
    for round_num in range(max_rounds):
        if time.time() - start > time_budget:
            break
        
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() - start > time_budget:
                break
            
            max_r = _max_radius_for_circle(i, x, y, r, W, H)
            
            if max_r > r[i] + 1e-8:
                new_r = min(max_r, r[i] * 1.05 + 0.001)
                r[i] = new_r
                x[i] = np.clip(x[i], r[i], W - r[i])
                y[i] = np.clip(y[i], r[i], H - r[i])
                improved = True
            else:
                for dx, dy in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                               (0.0007, 0.0007), (-0.0007, 0.0007),
                               (0.0007, -0.0007), (-0.0007, -0.0007)]:
                    nx, ny = x[i] + dx, y[i] + dy
                    if nx < r[i] or nx > W - r[i] or ny < r[i] or ny > H - r[i]:
                        continue
                    temp_x, temp_y = x.copy(), y.copy()
                    temp_x[i], temp_y[i] = nx, ny
                    new_max_r = _max_radius_for_circle(i, temp_x, temp_y, r, W, H)
                    if new_max_r > r[i] + 1e-8:
                        x[i], y[i] = nx, ny
                        r[i] = min(new_max_r, r[i] * 1.02 + 0.0005)
                        improved = True
                        break
        
        if not improved:
            sorted_idx = np.argsort(r)
            i, j = sorted_idx[0], sorted_idx[1]
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-9:
                move_frac = 0.002
                nx = x[i] + move_frac * dx
                ny = y[i] + move_frac * dy
                if r[i] <= nx <= W - r[i] and r[i] <= ny <= H - r[i]:
                    valid = True
                    for k in range(N):
                        if k == i:
                            continue
                        d = np.sqrt((nx - x[k])**2 + (ny - y[k])**2)
                        if d < r[i] + r[k] - 1e-9:
                            valid = False
                            break
                    if valid:
                        x[i], y[i] = nx, ny
                        for idx in [i, j]:
                            max_r = _max_radius_for_circle(idx, x, y, r, W, H)
                            if max_r > r[idx] + 1e-8:
                                r[idx] = min(max_r, r[idx] * 1.01 + 0.0003)
                                improved = True
                                break
        
        current_sum = np.sum(r)
        if current_sum > best_sum + 1e-10:
            best_sum = current_sum
            best_state = (x.copy(), y.copy(), r.copy())
        elif current_sum < best_sum - 1e-6:
            x, y, r = best_state
    
    result = np.column_stack([best_state[0], best_state[1], best_state[2]])
    return result


def _hexagonal_initialization(W, H):
    """
    Create a hexagonal lattice initial configuration.
    """
    circles = []
    best_circles = None
    best_count = 0

    for trial_r in np.linspace(0.08, 0.25, 20):
        dx = 2 * trial_r * 1.001
        dy = 2 * trial_r * np.sqrt(3) / 2 * 1.001

        row = 0
        y_pos = trial_r + dy / 2
        while y_pos + trial_r <= H + 1e-9 and len(circles) < N:
            offset = 0.0 if row % 2 == 0 else dx / 2
            x_pos = trial_r + offset
            row += 1
            while x_pos + trial_r <= W + 1e-9 and len(circles) < N:
                circles.append([x_pos, y_pos, trial_r])
                x_pos += dx
            y_pos += dy

        if len(circles) > best_count:
            best_count = len(circles)
            best_circles = circles.copy()

        circles = []

    if best_circles is None or len(best_circles) == 0:
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                circles.append([cx, cy, r])

    if len(circles) > N:
        circles = circles[:N]

    while len(circles) < N:
        best_x, best_y, best_r = W/2, H/2, 0.01
        for _ in range(50):
            tx = np.random.uniform(0.05, W - 0.05)
            ty = np.random.uniform(0.05, H - 0.05)
            min_dist = min(W, H)
            for c in circles:
                d = np.sqrt((tx - c[0])**2 + (ty - c[1])**2)
                min_dist = min(min_dist, d - c[2])
            min_dist = min(min_dist, tx, W - tx, ty, H - ty)
            if min_dist > best_r:
                best_r = min_dist * 0.9
                best_x, best_y = tx, ty
        best_r = max(0.01, best_r)
        circles.append([best_x, best_y, best_r])

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
    """Create a random initial configuration."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _row_configuration_initialization(W, H):
    """
    Initialize using specific row configurations that sum to 21.
    This is inspired by the hexagonal row approach from the inspiration.
    """
    # Try different row configurations
    row_configs = [
        [5, 4, 5, 4, 3],
        [4, 5, 4, 5, 3],
        [5, 5, 4, 4, 3],
        [6, 4, 4, 4, 3],
        [4, 4, 5, 4, 4],
        [5, 4, 4, 5, 3],
    ]
    
    best_circles = None
    best_sum = -np.inf
    
    for rows in row_configs:
        if sum(rows) != N:
            continue
        
        circles = []
        total_rows = len(rows)
        
        # Estimate optimal radius
        r_est = np.sqrt(W * H / (N * np.pi)) * 0.9
        
        for row_i, count in enumerate(rows):
            y = (row_i + 0.5) * H / total_rows
            for col_j in range(count):
                # Offset alternating rows for hexagonal packing
                offset = 0.0 if row_i % 2 == 0 else W / (2 * max(rows[0], 1))
                x = (col_j + 0.5) * W / max(rows[0], 1) + offset
                # Ensure within bounds
                x = np.clip(x, r_est, W - r_est)
                y = np.clip(y, r_est, H - r_est)
                circles.append([x, y, r_est])
        
        if len(circles) == N:
            # Check validity and compute sum
            circles_arr = np.array(circles)
            circles_arr[:, 0] = np.clip(circles_arr[:, 0], circles_arr[:, 2], W - circles_arr[:, 2])
            circles_arr[:, 1] = np.clip(circles_arr[:, 1], circles_arr[:, 2], H - circles_arr[:, 2])
            
            # Separate to ensure validity
            circles_sep, valid = _separate_circles(circles_arr, W, H, max_iter=500)
            if valid:
                s = np.sum(circles_sep[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = circles_sep
    
    if best_circles is None:
        # Fallback to hexagonal
        best_circles = _hexagonal_initialization(W, H)
    
    return best_circles


def _force_directed_layout(circles, W, H, max_iter=5000):
    """
    Force-directed layout to improve packing.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    target_r = np.sqrt(W * H / N) * 0.5

    for it in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dx = np.random.uniform(-1e-6, 1e-6)
                    dy = np.random.uniform(-1e-6, 1e-6)
                    dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    force = overlap * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist

        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (r[i] - x[i]) * 50.0
            if x[i] + r[i] > W:
                fx[i] -= (x[i] + r[i] - W) * 50.0
            if y[i] - r[i] < 0:
                fy[i] += (r[i] - y[i]) * 50.0
            if y[i] + r[i] > H:
                fy[i] -= (y[i] + r[i] - H) * 50.0

        damping = 0.5
        for i in range(n):
            x[i] += damping * fx[i] * 0.01
            y[i] += damping * fy[i] * 0.01
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        for i in range(n):
            min_space = min(x[i] - r[i], W - x[i] - r[i], y[i] - r[i], H - y[i] - r[i])
            for j in range(n):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_space = min(min_space, dist - r[j])
            if min_space > r[i] + 0.001:
                r[i] = min(target_r, r[i] + 0.001)

    result = np.column_stack([x, y, r])
    return result


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()

    # Try hexagonal initialization first
    circles_hex = _hexagonal_initialization(W, H)
    best_val = -np.sum(circles_hex[:, 2])
    best_circles = circles_hex.copy()

    # Try row configuration initialization
    circles_row = _row_configuration_initialization(W, H)
    if -np.sum(circles_row[:, 2]) > best_val:
        best_val = -np.sum(circles_row[:, 2])
        best_circles = circles_row.copy()

    # Also try grid initialization
    circles_grid = _grid_initialization(W, H)
    if -np.sum(circles_grid[:, 2]) > best_val:
        best_val = -np.sum(circles_grid[:, 2])
        best_circles = circles_grid.copy()

    # Try force-directed layout
    circles_fd = _force_directed_layout(circles_hex, W, H, max_iter=3000)
    circles_fd[:, 0] = np.clip(circles_fd[:, 0], circles_fd[:, 2], W - circles_fd[:, 2])
    circles_fd[:, 1] = np.clip(circles_fd[:, 1], circles_fd[:, 2], H - circles_fd[:, 2])
    circles_fd[:, 2] = np.clip(circles_fd[:, 2], 0.001, min(W, H) / 2)
    circles_fd, valid_fd = _separate_circles(circles_fd, W, H)
    if valid_fd:
        s = np.sum(circles_fd[:, 2])
        if s > -best_val:
            best_val = -s
            best_circles = circles_fd.copy()

    # Try local optimization with L-BFGS-B on hex init
    circles_opt = _local_optimize(circles_hex, W, H, max_iter=3000)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
    if valid_opt:
        s = np.sum(circles_opt[:, 2])
        if s > -best_val:
            best_val = -s
            best_circles = circles_opt.copy()

    # Try local optimization on row config init if time permits
    if time.time() - start < time_budget - 10:
        circles_opt2 = _local_optimize(circles_row, W, H, max_iter=2000)
        circles_opt2[:, 0] = np.clip(circles_opt2[:, 0], circles_opt2[:, 2], W - circles_opt2[:, 2])
        circles_opt2[:, 1] = np.clip(circles_opt2[:, 1], circles_opt2[:, 2], H - circles_opt2[:, 2])
        circles_opt2[:, 2] = np.clip(circles_opt2[:, 2], 0.001, min(W, H) / 2)
        circles_opt2, valid_opt2 = _separate_circles(circles_opt2, W, H)
        if valid_opt2:
            s = np.sum(circles_opt2[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = circles_opt2.copy()

    # Final polish if time permits
    if time.time() - start < time_budget - 3:
        polished = _polish(best_circles, W, H, max_iter=6000)
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

    # Try different aspect ratios - focus more on near-square where packing is denser
    aspect_ratios = [0.90, 0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07, 1.12, 1.20, 1.35]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        if 0.94 <= ar <= 1.06:
            time_budget = 20.0
        elif 0.88 <= ar <= 1.15:
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
            n_restarts = max(1, int(remaining / 6))
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
    if best_result is not None and time.time() - start_time < DEADLINE - 25:
        polished = _polish(best_result, best_W, best_H, max_iter=14000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=1000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Ultra-tight final polish
    if best_result is not None and time.time() - start_time < DEADLINE - 12:
        hp = _high_precision_polish(best_result, best_W, best_H, max_iter=10000)
        hp[:, 0] = np.clip(hp[:, 0], hp[:, 2], best_W - hp[:, 2])
        hp[:, 1] = np.clip(hp[:, 1], hp[:, 2], best_H - hp[:, 2])
        hp[:, 2] = np.clip(hp[:, 2], 0.001, min(best_W, best_H) / 2)
        hp, valid_hp = _separate_circles(hp, best_W, best_H, max_iter=2000)
        if valid_hp:
            s = np.sum(hp[:, 2])
            if s > best_score:
                best_score = s
                best_result = hp.copy()

    # Radius equalization local search
    if best_result is not None and time.time() - start_time < DEADLINE - 8:
        eq = _radius_equalization(best_result, best_W, best_H,
                                   max_rounds=150,
                                   time_budget=min(6.0, DEADLINE - 8 - (time.time() - start_time)))
        eq[:, 0] = np.clip(eq[:, 0], eq[:, 2], best_W - eq[:, 2])
        eq[:, 1] = np.clip(eq[:, 1], eq[:, 2], best_H - eq[:, 2])
        eq[:, 2] = np.clip(eq[:, 2], 0.001, min(best_W, best_H) / 2)
        eq, valid_eq = _separate_circles(eq, best_W, best_H, max_iter=500)
        if valid_eq:
            s = np.sum(eq[:, 2])
            if s > best_score:
                best_score = s
                best_result = eq.copy()

    # Local search radius increase - structural change
    if best_result is not None and time.time() - start_time < DEADLINE - 4:
        ls = _local_search_radius_increase(best_result, best_W, best_H,
                                            max_rounds=60,
                                            time_budget=min(5.0, DEADLINE - 4 - (time.time() - start_time)))
        ls[:, 0] = np.clip(ls[:, 0], ls[:, 2], best_W - ls[:, 2])
        ls[:, 1] = np.clip(ls[:, 1], ls[:, 2], best_H - ls[:, 2])
        ls[:, 2] = np.clip(ls[:, 2], 0.001, min(best_W, best_H) / 2)
        ls, valid_ls = _separate_circles(ls, best_W, best_H, max_iter=500)
        if valid_ls:
            s = np.sum(ls[:, 2])
            if s > best_score:
                best_score = s
                best_result = ls.copy()

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
