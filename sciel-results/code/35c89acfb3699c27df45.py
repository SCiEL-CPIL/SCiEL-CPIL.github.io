"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0


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

        # Clamp positions to keep circles inside rectangle
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
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - tol:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-9
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

    # Boundary penalties
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


def _high_precision_polish(circles, W, H, max_iter=10000):
    """
    Progressive smooth-max eps with increasing penalty weight.
    Uses log-sum-exp based soft max for strictly positive gradients.
    """
    current = circles.copy()
    
    # Progressive eps schedule
    eps_schedule = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
    weight_schedule = [200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0]
    
    n_stages = len(eps_schedule)
    iters_per_stage = max(500, max_iter // n_stages)
    
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))
    
    for idx, (eps, bw, ow) in enumerate(zip(eps_schedule, weight_schedule, weight_schedule)):
        params = current.flatten()
        
        result = minimize(
            _objective_and_grad, params, args=(W, H, eps, bw, ow),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': iters_per_stage, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
        )
        new_circles = result.x.reshape(N, 3)
        
        # Clamp and validate
        new_circles[:, 0] = np.clip(new_circles[:, 0], new_circles[:, 2], W - new_circles[:, 2])
        new_circles[:, 1] = np.clip(new_circles[:, 1], new_circles[:, 2], H - new_circles[:, 2])
        new_circles[:, 2] = np.clip(new_circles[:, 2], 1e-4, r_max)
        
        new_circles, valid = _separate_circles(new_circles, W, H)
        if valid:
            current = new_circles
        else:
            break
    
    return current


def _hexagonal_initialization(W, H):
    """Create a hexagonal packing initial configuration."""
    circles = []
    best = None
    for spacing in np.linspace(0.08, 0.3, 30):
        r = spacing * 0.5 - 1e-4
        if r <= 0:
            continue
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        count = 0
        for row in range(int(H / dy) + 2):
            for col in range(int(W / dx) + 2):
                if count >= N:
                    break
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                if row % 2 == 1:
                    x += dx / 2
                if x - r < 0 or x + r > W or y - r < 0 or y + r > H:
                    continue
                circles_candidate = circles.copy()
                circles_candidate.append([x, y, r])
                if len(circles_candidate) == N:
                    s = sum(c[2] for c in circles_candidate)
                    if best is None or s > best[1]:
                        best = (circles_candidate, s)
                    break
            if len(circles) >= N:
                break
        if len(circles) < N:
            continue
        circles = []
    
    if best is not None:
        return np.array(best[0])
    
    return _grid_initialization(W, H)


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


def _local_search_radius_increase(circles, W, H, max_iter=5000, tol=1e-10):
    """
    Try to increase each circle's radius while maintaining validity.
    Uses bisection to find the maximum feasible radius for each circle.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    r_max_global = min(W, H) / 2
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            max_r_boundary = min(x[i], W - x[i], y[i], H - y[i], r_max_global)
            max_r = max(1e-6, max_r_boundary)
            
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - r[j])
            
            max_r = max(1e-6, max_r)
            
            if max_r > r[i] + tol:
                lo, hi = r[i], max_r
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    feasible = True
                    if x[i] - mid < -1e-9 or x[i] + mid > W + 1e-9 or y[i] - mid < -1e-9 or y[i] + mid > H + 1e-9:
                        feasible = False
                    else:
                        for j in range(n):
                            if j == i:
                                continue
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < mid + r[j] - 1e-9:
                                feasible = False
                                break
                    if feasible:
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + tol:
                    r[i] = new_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return np.column_stack([x, y, r])


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    
    inits = [
        _grid_initialization(W, H),
        _hexagonal_initialization(W, H),
    ]
    
    rng = np.random.default_rng(int(W * 10000) + int(H * 10000))
    inits.append(_random_initialization(W, H, rng))
    
    best_val = -np.inf
    best_circles = None
    
    for init in inits:
        if time.time() - start > time_budget - 5:
            break
        
        circles_opt = _local_optimize(init, W, H, max_iter=2000)
        circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
        circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
        circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
        
        circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
        if valid_opt:
            s = np.sum(circles_opt[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circles_opt.copy()
    
    if best_circles is None:
        best_circles = inits[0]
        best_val = np.sum(best_circles[:, 2])
    
    # Local radius increase search
    if time.time() - start < time_budget - 12:
        rad_circles = _local_search_radius_increase(best_circles, W, H, max_iter=3000)
        rad_circles[:, 0] = np.clip(rad_circles[:, 0], rad_circles[:, 2], W - rad_circles[:, 2])
        rad_circles[:, 1] = np.clip(rad_circles[:, 1], rad_circles[:, 2], H - rad_circles[:, 2])
        rad_circles[:, 2] = np.clip(rad_circles[:, 2], 0.001, min(W, H) / 2)
        rad_circles, valid_rad = _separate_circles(rad_circles, W, H)
        if valid_rad:
            s = np.sum(rad_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = rad_circles.copy()
    
    # High-precision polish if time permits
    if time.time() - start < time_budget - 5:
        polished = _high_precision_polish(best_circles, W, H, max_iter=5000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 1e-4, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > best_val:
                best_val = s
                best_circles = polished.copy()
    
    # Final radius increase after polish
    if time.time() - start < time_budget - 1:
        rad_circles2 = _local_search_radius_increase(best_circles, W, H, max_iter=2000)
        rad_circles2[:, 0] = np.clip(rad_circles2[:, 0], rad_circles2[:, 2], W - rad_circles2[:, 2])
        rad_circles2[:, 1] = np.clip(rad_circles2[:, 1], rad_circles2[:, 2], H - rad_circles2[:, 2])
        rad_circles2[:, 2] = np.clip(rad_circles2[:, 2], 0.001, min(W, H) / 2)
        rad_circles2, valid_rad2 = _separate_circles(rad_circles2, W, H)
        if valid_rad2:
            s = np.sum(rad_circles2[:, 2])
            if s > best_val:
                best_val = s
                best_circles = rad_circles2.copy()
    
    return best_circles, best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate more time to promising aspect ratios (near square)
        if 0.9 <= ar <= 1.1:
            time_budget = 16.0
        elif 0.8 <= ar <= 1.3:
            time_budget = 13.0
        elif 0.7 <= ar <= 1.5:
            time_budget = 11.0
        else:
            time_budget = 9.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Try random restarts if time permits
        remaining = DEADLINE - 12 - (time.time() - start_time)
        if remaining > 5:
            n_restarts = max(1, int(remaining / 5))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 12:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1000)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
                circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
                if valid_rand:
                    circ_rad = _local_search_radius_increase(circ_opt, W, H, max_iter=1000)
                    circ_rad[:, 0] = np.clip(circ_rad[:, 0], circ_rad[:, 2], W - circ_rad[:, 2])
                    circ_rad[:, 1] = np.clip(circ_rad[:, 1], circ_rad[:, 2], H - circ_rad[:, 2])
                    circ_rad[:, 2] = np.clip(circ_rad[:, 2], 0.001, min(W, H) / 2)
                    circ_rad, valid_rad = _separate_circles(circ_rad, W, H)
                    if valid_rad:
                        s = np.sum(circ_rad[:, 2])
                    else:
                        s = np.sum(circ_opt[:, 2])
                        circ_rad = circ_opt
                    if s > best_score:
                        best_score = s
                        best_result = circ_rad.copy()
                        best_W, best_H = W, H

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _high_precision_polish(best_result, best_W, best_H, max_iter=10000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 1e-4, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()
    
    # Final radius increase pass
    if best_result is not None and time.time() - start_time < DEADLINE - 2:
        rad_final = _local_search_radius_increase(best_result, best_W, best_H, max_iter=3000)
        rad_final[:, 0] = np.clip(rad_final[:, 0], rad_final[:, 2], best_W - rad_final[:, 2])
        rad_final[:, 1] = np.clip(rad_final[:, 1], rad_final[:, 2], best_H - rad_final[:, 2])
        rad_final[:, 2] = np.clip(rad_final[:, 2], 1e-4, min(best_W, best_H) / 2)
        rad_final, valid_rad_final = _separate_circles(rad_final, best_W, best_H)
        if valid_rad_final:
            s = np.sum(rad_final[:, 2])
            if s > best_score:
                best_score = s
                best_result = rad_final.copy()

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
