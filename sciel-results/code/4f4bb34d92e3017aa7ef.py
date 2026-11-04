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


def _smooth_max_grad(v, eps):
    """Returns (smooth_max(0,v), d/dv smooth_max(0,v))."""
    smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
    grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
    return smax, grad_smax


def _objective_and_grad(params, W, H, eps, boundary_weight, overlap_weight):
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

    # Boundary penalties (smooth)
    # x - r >= 0  => violation = r - x
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
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-24:
                dist = 1e-12
                dx = 1e-12
                dy = 0.0
            else:
                dist = np.sqrt(dist_sq)
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v, eps)
            obj += overlap_weight * smax ** 2

            d_obj = 2.0 * overlap_weight * smax * gsmax
            d_obj_dx_i = d_obj * (-dx / dist)
            d_obj_dy_i = d_obj * (-dy / dist)
            d_obj_dx_j = d_obj * (dx / dist)
            d_obj_dy_j = d_obj * (dy / dist)
            d_obj_dr_i = d_obj * 1.0
            d_obj_dr_j = d_obj * 1.0

            grad[3 * i] += d_obj_dx_i
            grad[3 * i + 1] += d_obj_dy_i
            grad[3 * i + 2] += d_obj_dr_i
            grad[3 * j] += d_obj_dx_j
            grad[3 * j + 1] += d_obj_dy_j
            grad[3 * j + 2] += d_obj_dr_j

    return obj, grad


def _local_optimize(circles, W, H, max_iter=2000, xatol=1e-10, fatol=1e-12, eps=1e-4, boundary_weight=100.0, overlap_weight=100.0):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def obj_grad(p):
        return _objective_and_grad(p, W, H, eps, boundary_weight, overlap_weight)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-8, 'maxls': 50}
    )
    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=10000):
    """
    Progressive smooth-max eps (0.001 -> 1e-8) with increasing penalty weight.
    This lets L-BFGS-B resolve near-tangent circle contacts more precisely.
    """
    current = circles.copy()
    
    # Progressive eps schedule - extended for higher precision
    eps_schedule = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
    weight_schedule = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0, 5000.0, 8000.0, 12000.0, 18000.0]
    
    n_stages = len(eps_schedule)
    iters_per_stage = max(500, max_iter // n_stages)
    
    for idx, (eps, bw, ow) in enumerate(zip(eps_schedule, weight_schedule, weight_schedule)):
        r_max = min(W, H) / 2
        bounds = []
        for i in range(N):
            bounds.append((1e-6, W - 1e-6))
            bounds.append((1e-6, H - 1e-6))
            bounds.append((1e-6, r_max))
        
        params = current.flatten()
        
        def obj_grad(p):
            return _objective_and_grad(p, W, H, eps, bw, ow)
        
        result = minimize(
            obj_grad, params,
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
            # Keep the previous valid configuration
            break
    
    return current


def _hexagonal_initialization(W, H):
    """Create a hexagonal packing initial configuration."""
    circles = []
    # Try different hexagonal spacings
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
    
    # Fallback: grid
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

            # Boundary penalties
            boundary_weight = 100.0
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
                if d_r != 0:
                    grad[2::3] += 2.0 * boundary_weight * smax * gsmax * d_r

            # Linearized overlap penalties
            overlap_weight = 10.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx0 = x[i] - x[j]
                    dy0 = y[i] - y[j]
                    dist0_sq = dx0 * dx0 + dy0 * dy0
                    if dist0_sq < 1e-20:
                        dist0 = 1e-10
                        dx0 = 1e-10
                        dy0 = 0.0
                    else:
                        dist0 = np.sqrt(dist0_sq)
                    gx_i = dx0 / dist0
                    gy_i = dy0 / dist0

                    lin_dist = dist0 + gx_i * (px[i] - x[i]) + gy_i * (py[i] - y[i]) \
                                 - gx_i * (px[j] - x[j]) - gy_i * (py[j] - y[j])
                    v = (pr[i] + pr[j]) - lin_dist
                    smax, gsmax = _smooth_max_grad(v, eps)
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
            # Find max radius for circle i given current positions of others
            # Constraints: 
            # 1. x[i] - r >= 0, x[i] + r <= W => r <= min(x[i], W - x[i])
            # 2. y[i] - r >= 0, y[i] + r <= H => r <= min(y[i], H - y[i])
            # 3. For each j != i: dist(i,j) >= r[i] + r[j] => r[i] <= dist(i,j) - r[j]
            
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
                # Bisection to find safe radius
                lo, hi = r[i], max_r
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    # Check if mid radius is feasible
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
                    break  # Restart sweep since geometry changed
        
        if not improved:
            break
    
    return np.column_stack([x, y, r])


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    
    # Try multiple initializations
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
        
        # Local optimization
        circles_opt = _local_optimize(init, W, H, max_iter=1500)
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
    
    # Successive convex relaxation if time permits
    if time.time() - start < time_budget - 10:
        sc_circles, sc_score = _successive_convex_relaxation(best_circles, W, H, max_sweeps=10)
        sc_circles[:, 0] = np.clip(sc_circles[:, 0], sc_circles[:, 2], W - sc_circles[:, 2])
        sc_circles[:, 1] = np.clip(sc_circles[:, 1], sc_circles[:, 2], H - sc_circles[:, 2])
        sc_circles[:, 2] = np.clip(sc_circles[:, 2], 0.001, min(W, H) / 2)
        sc_circles, valid_sc = _separate_circles(sc_circles, W, H)
        if valid_sc:
            s = np.sum(sc_circles[:, 2])
            if s > best_val:
                best_val = s
                best_circles = sc_circles.copy()
    
    # High-precision polish if time permits
    if time.time() - start < time_budget - 3:
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

    # Try different aspect ratios - expanded range for better exploration
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
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=800)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
                circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
                if valid_rand:
                    # Quick radius increase
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
