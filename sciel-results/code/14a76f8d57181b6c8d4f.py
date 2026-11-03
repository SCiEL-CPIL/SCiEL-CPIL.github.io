"""Circle packing in unit square - Delaunay-based structural approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
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
    return np.sum(r), True


def _compute_radii_from_centers(xy):
    """
    Given circle centers, compute radii based on Delaunay triangulation.
    Each circle's radius is half the minimum of:
    - distance to nearest neighbor center
    - distance to nearest wall
    Then we can optionally scale radii up while maintaining validity.
    """
    n = len(xy)
    if n < 3:
        # Degenerate case
        r = np.zeros(n)
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = wall_dist * 0.5
        return r
    
    try:
        tri = Delaunay(xy)
    except Exception:
        # Fallback: just use wall distances and nearest neighbors
        r = np.zeros(n)
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            min_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                min_dist = min(min_dist, d)
            r[i] = min(wall_dist, min_dist) * 0.5
        return r
    
    # For each vertex, find the shortest incident edge
    r = np.zeros(n)
    for i in range(n):
        wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        min_edge_dist = np.inf
        
        # Find all edges incident to vertex i
        # In Delaunay, we can find neighbors through simplices
        for simplex in tri.simplices:
            if i in simplex:
                for j in simplex:
                    if i == j:
                        continue
                    d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                    min_edge_dist = min(min_edge_dist, d)
        
        if min_edge_dist == np.inf:
            min_edge_dist = wall_dist * 2
        
        r[i] = min(wall_dist, min_edge_dist * 0.5)
    
    return r


def _maximize_radii_given_centers(xy, max_iter=50):
    """
    Given fixed centers, iteratively increase radii while maintaining validity.
    Uses a simple greedy approach: repeatedly find the circle with most room to grow.
    """
    n = len(xy)
    r = _compute_radii_from_centers(xy)
    
    # Iteratively try to increase radii
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            # Compute max possible radius for circle i given others
            max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                max_r = min(max_r, (d - r[j]))
            
            if max_r > r[i] + 1e-10:
                r[i] = max_r
                improved = True
        
        if not improved:
            break
    
    return r


def _objective_repulsion(pos_params, n, repulsion_strength=10.0, wall_penalty=100.0):
    """
    Objective function: minimize potential energy.
    pos_params: flat array of shape (2*n,) containing [x_0, y_0, x_1, y_1, ...]
    
    The potential energy has:
    1. Repulsion between circles (inverse distance squared)
    2. Penalty for being outside the unit square
    3. Bonus for larger minimum radius (encourages spreading)
    """
    xy = pos_params.reshape(-1, 2)
    
    energy = 0.0
    
    # Wall penalty
    for i in range(n):
        x, y = xy[i, 0], xy[i, 1]
        if x < 0.01:
            energy += wall_penalty * (0.01 - x)**2
        elif x > 0.99:
            energy += wall_penalty * (x - 0.99)**2
        if y < 0.01:
            energy += wall_penalty * (0.01 - y)**2
        elif y > 0.99:
            energy += wall_penalty * (y - 0.99)**2
    
    # Repulsion between circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-10:
                dist_sq = 1e-10
            # Soft repulsion: stronger when closer
            dist = np.sqrt(dist_sq)
            # Ideal distance is related to target radius
            target_dist = 0.12  # Approximate ideal center distance
            if dist < target_dist:
                energy += repulsion_strength * (target_dist - dist)**2
    
    # Bonus: encourage larger minimum radius
    # Compute what radii would be
    r = _compute_radii_from_centers(xy)
    min_r = np.min(r)
    avg_r = np.mean(r)
    
    # We want to maximize sum of radii, so penalize small radii
    energy -= 5.0 * avg_r
    energy -= 10.0 * min_r
    
    return energy


def _gradient_descend_positions(xy, deadline, lr=0.01, max_iter=2000):
    """
    Custom gradient descent on positions using numerical gradients.
    """
    n = len(xy)
    params = xy.ravel().copy()
    best_energy = _objective_repulsion(params, n)
    best_params = params.copy()
    
    step_size = lr
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute numerical gradient
        grad = np.zeros_like(params)
        eps = 1e-5
        for k in range(len(params)):
            params_plus = params.copy()
            params_plus[k] += eps
            e_plus = _objective_repulsion(params_plus, n)
            grad[k] = (e_plus - best_energy) / eps
            params_plus[k] = params[k]
        
        # Gradient step
        params = params - step_size * grad
        
        # Clip to valid range
        for i in range(n):
            params[2*i] = np.clip(params[2*i], 0.01, 0.99)
            params[2*i+1] = np.clip(params[2*i+1], 0.01, 0.99)
        
        new_energy = _objective_repulsion(params, n)
        if new_energy < best_energy:
            best_energy = new_energy
            best_params = params.copy()
            # Reduce step size slowly
            step_size *= 0.999
        else:
            # Reject step, reduce step size
            step_size *= 0.95
            params = best_params.copy()
        
        if step_size < 1e-6:
            break
    
    return best_params.reshape(-1, 2)


def _simulated_annealing_positions(xy, deadline, t0=0.1, cooling=0.999, max_iter=5000):
    """
    Simulated annealing on center positions.
    """
    n = len(xy)
    current_params = xy.ravel().copy()
    current_energy = _objective_repulsion(current_params, n)
    best_params = current_params.copy()
    best_energy = current_energy
    t = t0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Propose random perturbation
        new_params = current_params.copy()
        idx = random.randint(0, len(new_params) - 1)
        delta = random.gauss(0, 0.02)
        new_params[idx] += delta
        new_params[idx] = np.clip(new_params[idx], 0.01, 0.99)
        
        new_energy = _objective_repulsion(new_params, n)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-10)):
            current_params = new_params
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_params = new_params.copy()
        
        t *= cooling
    
    return best_params.reshape(-1, 2)


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _random_init():
    """Generate random positions."""
    positions = []
    for i in range(N):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        positions.append([x, y])
    return np.array(positions[:N])


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
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

    max_sweeps = 25
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


def _gradient_polish_full(xy, r, deadline):
    """
    Full gradient-based polish on all 3n variables using smooth penalty.
    """
    n = len(xy)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    best_total, best_valid = _eval_packing(xy, r)
    best_params = params.copy()
    
    def _smooth_obj(p):
        xy_local = p[0::3].reshape(-1, 2)
        r_local = np.maximum(p[1::3], 0.001)
        
        penalty = 0.0
        # Wall violations
        for i in range(n):
            x, y, ri = xy_local[i, 0], xy_local[i, 1], r_local[i]
            penalty += max(0, ri - x)**2
            penalty += max(0, ri - (1.0 - x))**2
            penalty += max(0, ri - y)**2
            penalty += max(0, ri - (1.0 - y))**2
        # Overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy_local[i, 0] - xy_local[j, 0]
                dy = xy_local[i, 1] - xy_local[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r_local[i] + r_local[j]
                if dist < min_d:
                    penalty += (min_d - dist)**2
        
        return -np.sum(r_local) + 1000.0 * penalty
    
    try:
        result = minimize(
            _smooth_obj,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    return final_xy, final_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Strategy 1: Delaunay-based approach with simulated annealing on centers
    if time.time() < deadline - 30.0:
        # Initialize with grid
        for side in [5, 6]:
            if time.time() > deadline - 30.0:
                break
            xy_init = _grid_init(side)
            
            # Run simulated annealing on centers
            sa_xy = _simulated_annealing_positions(xy_init, deadline - 30.0, t0=0.08, cooling=0.9995, max_iter=3000)
            
            # Compute radii from centers
            r = _maximize_radii_given_centers(sa_xy)
            
            # Validate and evaluate
            total, valid = _eval_packing(sa_xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = sa_xy.copy()
                best_r = r.copy()
    
    # Strategy 2: Gradient descent on positions with repulsion model
    if time.time() < deadline - 20.0:
        for side in [5, 6, 7]:
            if time.time() > deadline - 20.0:
                break
            xy_init = _grid_init(side)
            
            gd_xy = _gradient_descend_positions(xy_init, deadline - 20.0, lr=0.005, max_iter=1500)
            
            # Compute radii
            r = _maximize_radii_given_centers(gd_xy)
            
            total, valid = _eval_packing(gd_xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = gd_xy.copy()
                best_r = r.copy()
    
    # Strategy 3: Best-fit local search (proven effective)
    if time.time() < deadline - 15.0:
        for side in [5, 6, 7]:
            if time.time() > deadline - 15.0:
                break
            xy_init = _grid_init(side)
            r_init = np.full(N, 0.5 / side - 0.002)
            
            out_xy, out_r, total, valid = _best_fit_local_search(xy_init, r_init, deadline - 15.0, grid_size=80)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r
    
    # If no valid result, fall back
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy_init = _grid_init(side)
        r = np.full(N, 0.4 / side)
        total, valid = _eval_packing(xy_init, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy_init
            best_r = r
        else:
            # Last resort
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.35 / side
            return result
    
    # Gradient polish on the best result
    if time.time() < deadline - 8.0:
        g_xy, g_r, g_total, g_valid = _gradient_polish_full(best_xy, best_r, deadline - 8.0)
        if g_valid and g_total > best_total:
            best_total = g_total
            best_valid = True
            best_xy = g_xy
            best_r = g_r
    
    # Final best-fit local search polish
    if time.time() < deadline - 3.0:
        p_xy, p_r, p_total, p_valid = _best_fit_local_search(best_xy, best_r, deadline - 2.0, grid_size=100)
        if p_valid and p_total > best_total:
            best_total = p_total
            best_valid = True
            best_xy = p_xy
            best_r = p_r
    
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
            result[i, 2] = 0.35 / side
    
    return result
# EVOLVE-BLOCK-END
