"""Circle packing in perimeter-4 rectangle - pressure field approach."""
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

        # Clamp position to keep circle inside rectangle
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
                    push = overlap + 1e-10
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation
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


def _pressure_field_objective_and_grad(params, W, H, beta=5.0):
    """
    Pressure field formulation: model circles as point sources in a 2D domain.
    The pressure at each circle center due to all other circles creates a repulsive
    force field. We minimize -sum(r) + pressure-based penalty terms.
    
    The pressure from a circle of radius r_i at position (x_i, y_i) at point (x, y)
    is modeled as: P_i(x,y) = r_i / (dist^2 + eps^2)
    
    This gives smooth, differentiable repulsion that naturally handles the packing.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    eps = 1e-4

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, gsmax

    # Boundary penalties
    for v_expr, d_x, d_y, d_r in [
        (r - x, -1, 0, 1),
        (r - y, 0, -1, 1),
        (x + r - W, 1, 0, 1),
        (y + r - H, 0, 1, 1),
        (-r, 0, 0, -1),
    ]:
        smax, gsmax = _smooth_max_grad(v_expr)
        obj += 100.0 * np.sum(smax ** 2)
        grad[0::3] += 200.0 * smax * gsmax * d_x
        grad[1::3] += 200.0 * smax * gsmax * d_y
        grad[2::3] += 200.0 * smax * gsmax * d_r

    # Pressure-based overlap penalty
    # For each pair, compute the "pressure" interaction
    pressure_weight = 50.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Pressure interaction: stronger when closer
            # P = r_i * r_j / (dist^2 + eps^2)
            # Overlap violation: r_i + r_j - dist
            v = r[i] + r[j] - dist
            smax, gsmax = _smooth_max_grad(v)
            
            # Additional pressure term that penalizes close approach
            pressure_term = (r[i] * r[j]) / (dist_sq + 0.01)
            obj += pressure_weight * (smax ** 2 + 0.1 * pressure_term)
            
            d_obj = pressure_weight * 2.0 * smax * gsmax
            d_obj_pressure = pressure_weight * 0.1 * (-2.0 * r[i] * r[j] * dx / (dist_sq + 0.01) ** 2)
            d_obj_pressure_y = pressure_weight * 0.1 * (-2.0 * r[i] * r[j] * dy / (dist_sq + 0.01) ** 2)
            
            grad[3 * i] += d_obj * (-dx / dist) + d_obj_pressure
            grad[3 * i + 1] += d_obj * (-dy / dist) + d_obj_pressure_y
            grad[3 * i + 2] += d_obj * 1.0
            grad[3 * j] += d_obj * (dx / dist) - d_obj_pressure
            grad[3 * j + 1] += d_obj * (dy / dist) - d_obj_pressure_y
            grad[3 * j + 2] += d_obj * 1.0

    return obj, grad


def _laplacian_smoothing_force(circles, W, H, sigma=0.1):
    """
    Compute Laplacian smoothing forces: each circle is pulled toward the local
    average of its neighbors, promoting uniform spacing.
    """
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    fx = np.zeros(n)
    fy = np.zeros(n)
    
    for i in range(n):
        # Find neighbors within 2*max_radius
        neighbors = []
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 3.0 * max(r):
                neighbors.append((j, dist))
        
        if len(neighbors) > 0:
            avg_x = np.mean([circles[j][0] for j, _ in neighbors])
            avg_y = np.mean([circles[j][1] for j, _ in neighbors])
            fx[i] = 0.1 * (avg_x - x[i])
            fy[i] = 0.1 * (avg_y - y[i])
    
    return fx, fy


def _local_optimize(circles, W, H, max_iter=3000, xatol=1e-12, fatol=1e-14):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _pressure_field_objective_and_grad, params, args=(W, H),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-10, 'maxls': 100}
    )
    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=10000):
    """High-precision polish with very tight tolerances."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Progressive tightening
    for tol_factor in [1e-10, 1e-12, 1e-14]:
        result = minimize(
            _pressure_field_objective_and_grad, params, args=(W, H),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': max_iter // 3, 'ftol': tol_factor, 'gtol': tol_factor * 0.1, 'maxls': 200}
        )
        params = result.x

    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """Create a hexagonal packing initial configuration."""
    circles = []
    # Try different hexagonal spacings
    for spacing in [0.15, 0.18, 0.2, 0.22]:
        r = spacing * 0.45
        rows = int(H / (spacing * np.sqrt(3) / 2)) + 1
        cols = int(W / spacing) + 1
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= N:
                    break
                x = col * spacing + (row % 2) * spacing / 2
                y = row * spacing * np.sqrt(3) / 2
                if x + r <= W and y + r <= H and x - r >= 0 and y - r >= 0:
                    circles.append([x, y, r])
                    count += 1
        if len(circles) == N:
            break
    
    if len(circles) < N:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        circles = []
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                circles.append([cx, cy, r])
    
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


def _force_directed_layout(circles, W, H, max_iter=5000, learning_rate=0.01):
    """
    Force-directed layout: treat circles as particles with repulsive forces
    from each other and attractive forces toward the center of available space.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    best_val = -np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 15:
            break
            
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle) * 1e-6
                    dy = np.sin(angle) * 1e-6
                    dist = 1e-6
                
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Strong repulsion when overlapping
                    force = 10.0 * (min_dist - dist) / dist
                else:
                    # Gentle repulsion to maintain spacing
                    force = 0.1 * (min_dist / dist) ** 4
                
                fx[i] += force * dx
                fy[i] += force * dy
                fx[j] -= force * dx
                fy[j] -= force * dy
        
        # Boundary forces
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += 5.0 * (r[i] - x[i])
            if x[i] > W - r[i]:
                fx[i] -= 5.0 * (x[i] - (W - r[i]))
            if y[i] < r[i]:
                fy[i] += 5.0 * (r[i] - y[i])
            if y[i] > H - r[i]:
                fy[i] -= 5.0 * (y[i] - (H - r[i]))
        
        # Update positions
        x += learning_rate * fx
        y += learning_rate * fy
        
        # Clamp
        for i in range(n):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
        
        # Try to increase radii
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            potential_r = min(max_r_x, max_r_y)
            if potential_r > r[i]:
                r[i] = min(potential_r, r[i] * 1.001)
        
        current_val = -np.sum(r)
        if current_val < best_val:
            best_val = current_val
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Decay learning rate
        learning_rate *= 0.999
    
    return np.column_stack([best_x, best_y, best_r]), -best_val


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found for given W, H."""
    start = time.time()
    
    # Try multiple initializations
    initializations = [
        _hexagonal_initialization(W, H),
        _grid_initialization(W, H),
    ]
    
    best_val = -np.inf
    best_circles = None
    
    for init in initializations:
        if time.time() - start > time_budget - 5:
            break
        
        # Local optimization
        circles_opt = _local_optimize(init, W, H, max_iter=2000)
        circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
        circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
        circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
        
        # Separate overlaps
        circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
        if valid_opt:
            s = np.sum(circles_opt[:, 2])
            if s > best_val:
                best_val = s
                best_circles = circles_opt.copy()
        
        # Force-directed refinement if time permits
        if time.time() - start < time_budget - 3:
            fd_circles, fd_score = _force_directed_layout(best_circles, W, H, max_iter=2000)
            fd_circles[:, 0] = np.clip(fd_circles[:, 0], fd_circles[:, 2], W - fd_circles[:, 2])
            fd_circles[:, 1] = np.clip(fd_circles[:, 1], fd_circles[:, 2], H - fd_circles[:, 2])
            fd_circles[:, 2] = np.clip(fd_circles[:, 2], 0.001, min(W, H) / 2)
            fd_circles, valid_fd = _separate_circles(fd_circles, W, H)
            if valid_fd:
                s = np.sum(fd_circles[:, 2])
                if s > best_val:
                    best_val = s
                    best_circles = fd_circles.copy()
    
    # High-precision polish if time permits
    if best_circles is not None and time.time() - start < time_budget - 2:
        polished = _high_precision_polish(best_circles, W, H, max_iter=5000)
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

    # Try different aspect ratios
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Allocate time based on how promising the aspect ratio is
        if 0.95 <= ar <= 1.05:
            time_budget = 25.0
        elif 0.85 <= ar <= 1.15:
            time_budget = 18.0
        else:
            time_budget = 12.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Random restarts if time permits
        remaining = DEADLINE - 15 - (time.time() - start_time)
        if remaining > 8:
            n_restarts = max(1, int(remaining / 6))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 15:
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

    # Final high-precision polish
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _high_precision_polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=2000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish pass
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished2 = _high_precision_polish(best_result, best_W, best_H, max_iter=5000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=2000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()

    # Fallback: simple grid in square
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
