"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _repulsion_relaxation_init(n_circles=26, n_iters=2000, repulse_strength=0.01):
    """
    Initialize circles via random placement then repulsion relaxation.
    Circles push each other apart and are attracted to the center of available space.
    This produces a well-distributed starting configuration close to optimal packing.
    """
    # Start with random positions, small radii
    r = np.full(n_circles, 0.02)
    xy = np.column_stack([
        random.uniform(0.1, 0.9) for _ in range(n_circles)
    ])
    xy = np.column_stack([
        np.array([random.uniform(0.1, 0.9) for _ in range(n_circles)]),
        np.array([random.uniform(0.1, 0.9) for _ in range(n_circles)])
    ])
    
    for it in range(n_iters):
        # Compute repulsion forces between all pairs
        forces_x = np.zeros(n_circles)
        forces_y = np.zeros(n_circles)
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-6:
                    dx = random.uniform(-0.01, 0.01)
                    dy = random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx * dx + dy * dy)
                
                # Repulsion: stronger when circles are close
                min_sep = r[i] + r[j]
                if dist < min_sep * 2.0:
                    force = repulse_strength * (min_sep * 2.0 - dist) / dist
                    forces_x[i] += force * dx
                    forces_y[i] += force * dy
                    forces_x[j] -= force * dx
                    forces_y[j] -= force * dy
        
        # Wall repulsion: push circles away from walls
        for i in range(n_circles):
            margin = r[i]
            if xy[i, 0] < margin:
                forces_x[i] += repulse_strength * (margin - xy[i, 0])
            if xy[i, 0] > 1.0 - margin:
                forces_x[i] -= repulse_strength * (xy[i, 0] - (1.0 - margin))
            if xy[i, 1] < margin:
                forces_y[i] += repulse_strength * (margin - xy[i, 1])
            if xy[i, 1] > 1.0 - margin:
                forces_y[i] -= repulse_strength * (xy[i, 1] - (1.0 - margin))
        
        # Apply forces with damping
        xy[:, 0] += forces_x
        xy[:, 1] += forces_y
        
        # Clamp to unit square
        for i in range(n_circles):
            xy[i, 0] = np.clip(xy[i, 0], r[i], 1.0 - r[i])
            xy[i, 1] = np.clip(xy[i, 1], r[i], 1.0 - r[i])
        
        # Gradually increase radii toward target
        target_r = 0.04
        r[:] = np.minimum(r + 0.0001, target_r)
    
    # Final radius assignment: each circle gets max possible radius given neighbors
    for i in range(n_circles):
        max_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        for j in range(n_circles):
            if i != j:
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, max(0.005, (dist - r[j]) * 0.95))
        r[i] = max(max_r, 0.005)
    
    return xy, r


def _sa_objective(params, n):
    """
    Simulated annealing objective: minimize negative sum of radii + large penalty for violations.
    params: flat array [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    xy = params[0::3].reshape(-1, 2)
    r = np.maximum(params[2::3], 0.001)
    
    # Wall violations
    wall_pen = 0.0
    for i in range(n):
        wall_pen += max(0.0, r[i] - xy[i, 0]) ** 2
        wall_pen += max(0.0, r[i] - (1.0 - xy[i, 0])) ** 2
        wall_pen += max(0.0, r[i] - xy[i, 1]) ** 2
        wall_pen += max(0.0, r[i] - (1.0 - xy[i, 1])) ** 2
    
    # Overlap violations
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if dist < min_d:
                overlap_pen += (min_d - dist) ** 2
    
    return -np.sum(r) + 100.0 * (wall_pen + overlap_pen)


def _simulated_annealing(xy_init, r_init, deadline, max_iters=50000, T0=0.05, T_end=1e-6, cooling=0.9998):
    """
    Simulated annealing to optimize circle packing.
    Moves: perturb position and/or radius of one circle.
    """
    n = len(xy_init)
    params = np.zeros(3 * n)
    params[0::3] = xy_init[:, 0]
    params[1::3] = xy_init[:, 1]
    params[2::3] = r_init
    
    best_params = params.copy()
    best_val = _sa_objective(params, n)
    
    # Check if current is valid
    cur_xy = params[0::3].reshape(-1, 2)
    cur_r = np.maximum(params[2::3], 0.001)
    total, valid = _eval_packing(cur_xy, cur_r)
    if valid:
        best_val = -total
        best_params = params.copy()
    
    T = T0
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random circle and perturb it
        i = rng.randint(n)
        move_type = rng.random()
        
        if move_type < 0.4:
            # Perturb x position
            params[3*i] += rng.normal(0, 0.01)
        elif move_type < 0.8:
            # Perturb y position
            params[3*i+1] += rng.normal(0, 0.01)
        else:
            # Perturb radius
            params[3*i+2] += rng.normal(0, 0.005)
        
        # Clamp
        params[3*i] = np.clip(params[3*i], 0.001, 0.999)
        params[3*i+1] = np.clip(params[3*i+1], 0.001, 0.999)
        params[3*i+2] = np.clip(params[3*i+2], 0.001, 0.4)
        
        new_val = _sa_objective(params, n)
        delta = new_val - best_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept move
            best_val = new_val
            best_params = params.copy()
        
        T *= cooling
    
    # Extract best
    best_xy = best_params[0::3].reshape(-1, 2)
    best_r = np.maximum(best_params[2::3], 0.001)
    
    total, valid = _eval_packing(best_xy, best_r)
    if not valid:
        # Try to repair
        for i in range(n):
            best_r[i] = min(best_r[i], best_xy[i, 0], 1.0 - best_xy[i, 0], best_xy[i, 1], 1.0 - best_xy[i, 1])
            best_xy[i, 0] = np.clip(best_xy[i, 0], best_r[i], 1.0 - best_r[i])
            best_xy[i, 1] = np.clip(best_xy[i, 1], best_r[i], 1.0 - best_r[i])
        total, valid = _eval_packing(best_xy, best_r)
    
    return best_xy, best_r, total, valid


def _slsqp_polish(xy, r, deadline):
    """
    SLSQP polish with explicit constraints for tight convergence.
    """
    n = len(xy)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    # Objective: maximize sum of radii (minimize negative)
    def obj(p):
        r_arr = np.maximum(p[2::3], 0.001)
        return -np.sum(r_arr)
    
    # Constraints: no overlap, inside square
    constraints = []
    for i in range(n):
        # Wall constraints
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (1.0 - p[3*i]) - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (1.0 - p[3*i+1]) - p[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
            })
    
    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.001, 0.4)] * n
    
    best_params = params.copy()
    best_total, best_valid = _eval_packing(xy, r)
    
    try:
        result = minimize(
            obj, params, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[2::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[2::3], 0.001)
    return final_xy, final_r, best_total, best_valid


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Grid-based best-fit local search for quick improvement.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Repulsion relaxation initialization
    if time.time() < deadline - 60.0:
        xy, r = _repulsion_relaxation_init(N, n_iters=1500)
        
        # Quick local search to fix up
        if time.time() < deadline - 40.0:
            out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 40.0, grid_size=60)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r
    
    # Phase 2: Simulated annealing (multiple restarts if time permits)
    if best_xy is not None and time.time() < deadline - 20.0:
        sa_deadline = deadline - 15.0
        sa_xy, sa_r, sa_total, sa_valid = _simulated_annealing(
            best_xy, best_r, sa_deadline, 
            max_iters=30000, T0=0.03, T_end=1e-7, cooling=0.99985
        )
        if sa_valid and sa_total > best_total:
            best_total = sa_total
            best_valid = True
            best_xy = sa_xy
            best_r = sa_r
    
    # Phase 3: SLSQP polish for tight convergence
    if best_xy is not None and time.time() < deadline - 5.0:
        p_xy, p_r, p_total, p_valid = _slsqp_polish(best_xy, best_r, deadline - 2.0)
        if p_valid and p_total > best_total:
            best_total = p_total
            best_valid = True
            best_xy = p_xy
            best_r = p_r
    
    # Fallback if nothing valid found
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
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
