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
    if n == 0:
        return 0.0, True
        
    # Check walls
    min_dist_to_wall = np.minimum(
        np.minimum(xy[:, 0], 1.0 - xy[:, 0]),
        np.minimum(xy[:, 1], 1.0 - xy[:, 1])
    )
    
    if np.any(r < -1e-9) or np.any(r > min_dist_to_wall + 1e-9):
        return 0.0, False

    # Check pairwise distances
    diff = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < r_sum[mask] - 1e-9):
        return 0.0, False
        
    return float(np.sum(r)), True


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

            # Refinement: if we moved, do a finer local search around the new position
            if new_r > old_r + 1e-10:
                # Create a finer grid around the best position
                step = 1.0 / grid_size
                fine_gx = np.linspace(max(0, new_x - 3*step), min(1, new_x + 3*step), 20)
                fine_gy = np.linspace(max(0, new_y - 3*step), min(1, new_y + 3*step), 20)
                FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
                fine_candidates = np.column_stack([FGX.ravel(), GY.ravel()])
                FM = len(fine_candidates)
                
                if FM > 0:
                    fine_wall_clear = np.minimum(np.minimum(fine_candidates[:, 0], 1.0 - fine_candidates[:, 0]),
                                                 np.minimum(fine_candidates[:, 1], 1.0 - fine_candidates[:, 1]))
                    if len(others_xy) > 0:
                        fdx = fine_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                        fdy = fine_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                        fdists = np.sqrt(fdx * fdx + fdy * fdy)
                        fcircle_clear = np.min(fdists - others_r[np.newaxis, :], axis=1)
                        fr_candidates = np.minimum(fine_wall_clear, fcircle_clear)
                    else:
                        fr_candidates = fine_wall_clear.copy()
                    
                    fidx = int(np.argmax(fr_candidates))
                    fnew_x, fnew_y = fine_candidates[fidx, 0], fine_candidates[fidx, 1]
                    fnew_r = fr_candidates[fidx]
                    
                    if fnew_r > new_r + 1e-10:
                        xy[i, 0] = fnew_x
                        xy[i, 1] = fnew_y
                        r[i] = fnew_r

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n):
    """
    Smooth objective: minimize -sum(r) + penalty for violations.
    v is a vector of length 3*n: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    beta = 50.0
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    # Pairwise violations
    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    wall_pen_sum = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    return -np.sum(r) + 100.0 * (wall_pen_sum + total_viol)


def _smooth_grad(v, n):
    """
    Gradient of the smooth objective using softplus derivatives.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 50.0

    # Wall terms
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        if min_idx == 0:
            grad[2*n + i] += 100.0 * s
            grad[i] -= 100.0 * s
        elif min_idx == 1:
            grad[2*n + i] += 100.0 * s
            grad[i] += 100.0 * s
        elif min_idx == 2:
            grad[2*n + i] += 100.0 * s
            grad[n + i] -= 100.0 * s
        else:
            grad[2*n + i] += 100.0 * s
            grad[n + i] += 100.0 * s

    # Pairwise terms
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-10:
                continue
            factor = 100.0 * s
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor

    # Objective: -sum(r)
    grad[2*n:3*n] -= 1.0

    return grad


def _bubble_relaxation(xy, r, deadline, dt=0.005, max_iter=5000, damping=0.9):
    """
    Physics-based relaxation: circles are pressurized bubbles.
    Each circle has internal pressure proportional to r^2, pushing it outward.
    Repulsive forces between overlapping circles push them apart.
    We simulate this with explicit Euler integration with damping.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute forces on each circle
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pressure force: each circle wants to expand, pushing against walls and other circles
        # The "pressure" is proportional to r_i^2, so the force on circle i from pressure
        # is directed away from other circles and walls.
        
        for i in range(n):
            # Force from other circles (repulsion when overlapping or close)
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    # Random small nudge
                    fx[i] += 0.01 * (np.random.random() - 0.5)
                    fy[i] += 0.01 * (np.random.random() - 0.5)
                    continue
                
                # Overlap amount
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    # Use a spring constant that scales with the radii
                    k = 10.0
                    fx[i] += k * overlap * dx / dist
                    fy[i] += k * overlap * dy / dist
            
            # Wall forces: if circle is too close to a wall, push it back
            # Left wall
            if xy[i, 0] < r[i]:
                fx[i] += 50.0 * (r[i] - xy[i, 0])
            # Right wall
            if xy[i, 0] > 1.0 - r[i]:
                fx[i] -= 50.0 * (xy[i, 0] - (1.0 - r[i]))
            # Bottom wall
            if xy[i, 1] < r[i]:
                fy[i] += 50.0 * (r[i] - xy[i, 1])
            # Top wall
            if xy[i, 1] > 1.0 - r[i]:
                fy[i] -= 50.0 * (xy[i, 1] - (1.0 - r[i]))
            
            # Pressure expansion: each circle wants to grow, so we add a small outward force
            # This is modeled by slightly increasing the effective radius in the force calculation
            # Alternatively, we can directly try to increase r if there's room
            # For now, the repulsion from other circles and walls naturally limits growth
        
        # Update positions with explicit Euler
        xy[:, 0] += dt * fx
        xy[:, 1] += dt * fy
        
        # Damping: reduce velocities (implicitly by reducing the effective step)
        # In this simple model, we just clip positions to stay in bounds
        for i in range(n):
            xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
            xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
        
        # After moving, try to grow each circle as much as possible
        for i in range(n):
            # Compute max possible radius given current position and other circles
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # r_i + r_j <= dist => r_i <= dist - r_j
                max_r = min(max_r, dist - r[j])
            max_r = max(max_r, 0.0)
            # Grow towards max_r with some factor
            target_r = max_r
            r[i] = min(target_r, r[i] * 1.01 + 0.001)  # Gradual growth
        
        # Evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _gradient_polish(xy, r, deadline, tol=1e-10, max_iter=2000):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8,
                                   'maxfun': 50000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    # Validate
    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def _high_precision_polish(xy, r, deadline):
    """
    Final high-precision polish: tight tolerance, more iterations.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                   'maxfun': 100000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + local search
    for seed in range(8):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Bubble relaxation from best result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 40:
            bubble_xy, bubble_r, bubble_total, bubble_valid = _bubble_relaxation(
                best_xy.copy(), best_r.copy(), deadline)
            if bubble_valid and bubble_total > best_total + 1e-10:
                best_total = bubble_total
                best_xy = bubble_xy
                best_r = bubble_r

    # Phase 4: Gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 30:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

        # Phase 5: Second local search pass from polished result
        if best_valid and best_xy is not None and time.time() < deadline - 20:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _best_fit_local_search(
                best_xy.copy(), best_r.copy(), deadline, grid_size=100)
            if ls2_valid and ls2_total > best_total + 1e-10:
                best_total = ls2_total
                best_xy = ls2_xy
                best_r = ls2_r

        # Phase 6: High-precision polish
        if best_valid and best_xy is not None and time.time() < deadline - 5:
            hp_xy, hp_r = _high_precision_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(hp_xy, hp_r)
            if valid and total > best_total:
                best_total = total
                best_xy = hp_xy
                best_r = hp_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
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
