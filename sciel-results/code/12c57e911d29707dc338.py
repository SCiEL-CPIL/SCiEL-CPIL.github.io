"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
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


def _physics_relax(xy, r, deadline, steps=5000, k_repel=10.0, k_attract=5.0):
    """
    Physics-based relaxation: circles repel each other and are attracted to walls.
    Each circle's radius is updated to fit in the available space.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    
    best_total, best_valid = _eval_packing(current_xy, current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    # Initial temperature for simulated annealing
    T = 0.01
    T_min = 0.0001
    cooling = 0.995
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces_x = np.zeros(n)
        forces_y = np.zeros(n)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_xy[i, 0] - current_xy[j, 0]
                dy = current_xy[i, 1] - current_xy[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                
                # Overlap or close: repel
                min_dist = current_r[i] + current_r[j]
                if dist < min_dist + 0.01:
                    # Force magnitude increases as they get closer
                    force = k_repel * (min_dist - dist + 0.01)
                    if dist > 1e-8:
                        fx = force * dx / dist
                        fy = force * dy / dist
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
        
        # Attraction to walls (to maximize radius)
        for i in range(n):
            # Distance to each wall
            dist_left = current_xy[i, 0]
            dist_right = 1.0 - current_xy[i, 0]
            dist_bottom = current_xy[i, 1]
            dist_top = 1.0 - current_xy[i, 1]
            
            # Attract to the closest wall
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            
            if min_dist > current_r[i]:
                # Pull towards the closest wall
                if min_dist == dist_left:
                    forces_x[i] -= k_attract * (min_dist - current_r[i])
                elif min_dist == dist_right:
                    forces_x[i] += k_attract * (min_dist - current_r[i])
                elif min_dist == dist_bottom:
                    forces_y[i] -= k_attract * (min_dist - current_r[i])
                else:
                    forces_y[i] += k_attract * (min_dist - current_r[i])
            
            # Add thermal noise
            forces_x[i] += T * np.random.randn()
            forces_y[i] += T * np.random.randn()
        
        # Update positions
        for i in range(n):
            current_xy[i, 0] += forces_x[i] * 0.01
            current_xy[i, 1] += forces_y[i] * 0.01
            
            # Keep within bounds
            current_xy[i, 0] = np.clip(current_xy[i, 0], 0.0, 1.0)
            current_xy[i, 1] = np.clip(current_xy[i, 1], 0.0, 1.0)
        
        # Update radii based on available space
        for i in range(n):
            # Max radius limited by walls
            max_r_wall = min(
                current_xy[i, 0],
                1.0 - current_xy[i, 0],
                current_xy[i, 1],
                1.0 - current_xy[i, 1]
            )
            
            # Max radius limited by other circles
            max_r_circles = max_r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = current_xy[i, 0] - current_xy[j, 0]
                dy = current_xy[i, 1] - current_xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # The circle i can have radius up to (dist - r_j) / 2 if centered
                # But we want to fit in the gap, so use a heuristic
                available = (dist - current_r[j]) * 0.5
                if available > 0:
                    max_r_circles = min(max_r_circles, available + current_r[i] * 0.5)
            
            # Smoothly update radius
            target_r = max(0.001, min(max_r_wall, max_r_circles))
            current_r[i] += 0.1 * (target_r - current_r[i])
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Evaluate
        total, valid = _eval_packing(current_xy, current_r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = current_xy.copy()
            best_r = current_r.copy()
    
    return best_xy, best_r, best_total, best_valid


def _simulated_annealing(xy, r, deadline, steps=20000):
    """
    Simulated annealing to escape local optima.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    
    best_total, best_valid = _eval_packing(current_xy, current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        # Random perturbation
        i = np.random.randint(n)
        dx = np.random.randn() * 0.02
        dy = np.random.randn() * 0.02
        dr = np.random.randn() * 0.01
        
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        new_xy[i, 0] = np.clip(new_xy[i, 0] + dx, 0.0, 1.0)
        new_xy[i, 1] = np.clip(new_xy[i, 1] + dy, 0.0, 1.0)
        new_r[i] = np.clip(new_r[i] + dr, 0.001, 0.5)
        
        # Evaluate
        new_total, new_valid = _eval_packing(new_xy, new_r)
        old_total = best_total if best_valid else 0.0
        
        if new_valid and new_total > best_total + 1e-10:
            current_xy = new_xy
            current_r = new_r
            best_total = new_total
            best_valid = True
            best_xy = new_xy.copy()
            best_r = new_r.copy()
        elif new_valid:
            # Accept with probability exp(-delta / T)
            delta = old_total - new_total
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-10)):
                current_xy = new_xy
                current_r = new_r
        
        # Cool down
        T = max(T_min, T * cooling)
    
    return best_xy, best_r, best_total, best_valid


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=500):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
    for i in range(n):
        bounds.append((0.001, 0.999))  # y
    for i in range(n):
        bounds.append((0.001, 0.5))  # r

    # Adaptive beta strategy: start loose, tighten up
    betas = [10.0, 50.0, 100.0]
    
    current_xy = xy.copy()
    current_r = r.copy()
    best_total, _ = _eval_packing(current_xy, current_r)

    def _smooth_objective(v, n, beta=50.0):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]

        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = r[i] + r[j] - dist
                total_viol += np.logaddexp(0.0, beta * viol) / beta

        return -np.sum(r) + 100.0 * (wall_pen + total_viol)

    def _smooth_grad(v, n, beta=50.0):
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        r = v[2*n:3*n].copy()

        grad = np.zeros(3 * n)

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

        grad[2*n:3*n] -= 1.0
        return grad

    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        def obj(v, b=beta):
            if time.time() > deadline:
                return float('inf')
            return _smooth_objective(v, n, b)

        def grad(v, b=beta):
            if time.time() > deadline:
                return np.zeros(3 * n)
            return _smooth_grad(v, n, b)

        try:
            result = minimize(obj, np.concatenate([current_xy[:, 0], current_xy[:, 1], current_r]), 
                              jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                                       'maxfun': 10000})
        except Exception:
            continue

        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]

        total, valid = _eval_packing(new_xy, new_r)
        if valid and total > best_total:
            best_total = total
            current_xy = new_xy
            current_r = new_r

    return current_xy, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + physics relaxation
    for side in [5, 6, 7]:
        if time.time() > deadline - 20:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _physics_relax(xy, r, deadline - 20, steps=3000)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + simulated annealing
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _simulated_annealing(xy, r, deadline - 30, steps=10000)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline - 10)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

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
