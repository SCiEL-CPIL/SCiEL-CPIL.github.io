"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def _expand_radius(centers, r, time_limit=5.0):
    """
    Expand radius r by incrementally increasing it and resolving collisions.
    Returns the final centers and radius.
    """
    start_time = time.time()
    current_r = r
    current_centers = centers.copy()
    max_steps = 2000
    step_size = 0.001
    
    # Precompute indices for pairwise constraints
    iu = np.triu_indices(N, 1)
    
    for _ in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Try to increase radius
        new_r = current_r + step_size
        
        # Check if we can move centers to accommodate new_r
        # We use a simple gradient ascent on the minimum distance, 
        # but constrained to stay within the square and not overlap at new_r.
        # Actually, a better approach for "expansion" is to treat it as a 
        # feasibility problem: find centers such that min_dist >= 2*new_r 
        # and within walls.
        
        # Let's define a function that measures how much we violate the constraints
        # for the new radius. We want to minimize this violation.
        
        def violation(x):
            c = x.reshape(N, 2)
            # Wall violation
            wall_viol = np.maximum(0, np.abs(c) - (0.5 - new_r))
            # Pairwise violation
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            pair_viol = np.maximum(0, 2 * new_r - dist)[iu]
            
            # Sum of squares of violations
            return np.sum(wall_viol ** 2) + np.sum(pair_viol ** 2)
        
        # If violation is already 0, we can accept the new radius
        if violation(current_centers.ravel()) < 1e-12:
            current_r = new_r
            continue
            
        # Try to minimize violation using a few steps of gradient descent
        # We need the gradient of the violation function.
        # This is expensive. Let's use a simpler heuristic:
        # Move points away from each other and towards center if hitting walls.
        
        # Calculate forces
        forces = np.zeros_like(current_centers)
        
        # Pairwise repulsion
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0) # avoid div by zero
        
        # Only consider pairs that are too close
        too_close = dist < 2 * new_r
        # Force magnitude proportional to overlap
        overlap = np.maximum(0, 2 * new_r - dist)
        # Normalize diff
        norm_diff = diff / dist[:, :, np.newaxis]
        
        # Sum forces
        for i in range(N):
            for j in range(N):
                if i != j and too_close[i, j]:
                    forces[i] += norm_diff[i, j] * overlap[i, j]
        
        # Wall repulsion
        for i in range(N):
            for k in range(2):
                if current_centers[i, k] > 0.5 - new_r:
                    forces[i, k] -= (current_centers[i, k] - (0.5 - new_r))
                elif current_centers[i, k] < -(0.5 - new_r):
                    forces[i, k] += (-(0.5 - new_r) - current_centers[i, k])
        
        # Move
        current_centers += 0.1 * forces
        
        # Project back to square (hard constraint)
        current_centers = np.clip(current_centers, -0.5 + new_r, 0.5 - new_r)
        
        # If we made progress, accept new radius
        # Check if min distance is now >= 2*new_r
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_d = np.min(dist)
        
        if min_d >= 2 * new_r - 1e-9 and np.all(np.abs(current_centers) <= 0.5 - new_r + 1e-9):
            current_r = new_r
        else:
            # Failed to expand, try smaller step or break?
            # For simplicity, if we fail a few times, we stop expanding.
            # A more robust approach would be to use scipy.optimize to solve 
            # the feasibility problem for the new radius.
            pass

    return current_centers, current_r

def _slsqp_polish(centers, r, time_limit=5.0):
    """
    Use SLSQP to maximize r (or minimize -r) with hard constraints.
    x = [c1, c2, ..., cN, r]
    """
    start_time = time.time()
    
    # Initial x: centers and current radius
    x0 = np.concatenate([centers.ravel(), [r]])
    
    def objective(x):
        return -x[-1] # Maximize r
    
    def constraint(x):
        c = x[:N*2].reshape(N, 2)
        r = x[-1]
        
        # Wall constraints: 0.5 - |c_ik| >= r  => 0.5 - r - |c_ik| >= 0
        # This is non-differentiable at 0. Use smooth approx or split into linear?
        # Linear constraints:
        # 0.5 - c_x - r >= 0
        # 0.5 + c_x - r >= 0
        # 0.5 - c_y - r >= 0
        # 0.5 + c_y - r >= 0
        
        cons = []
        for i in range(N):
            cons.append(0.5 - c[i, 0] - r)
            cons.append(0.5 + c[i, 0] - r)
            cons.append(0.5 - c[i, 1] - r)
            cons.append(0.5 + c[i, 1] - r)
            
        # Pairwise constraints: ||c_i - c_j|| >= 2r
        # (c_ix - c_jx)^2 + (c_iy - c_jy)^2 - 4r^2 >= 0
        # This is non-linear.
        for i in range(N):
            for j in range(i+1, N):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                cons.append(dx*dx + dy*dy - 4*r*r)
        
        return cons
    
    def jac(x):
        # Gradient of objective
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        
        # We don't provide jac for constraints to SLSQP by default if not specified, 
        # it uses finite differences. This is slow. 
        # Given time constraints, we might skip jac or provide it if we can.
        # Let's rely on finite differences but limit iterations.
        return grad

    # Bounds
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.01, 0.5)]
    
    # Constraints for SLSQP format: dict(type='ineq', fun=...)
    # SLSQP expects fun(x) >= 0 for 'ineq'
    
    # Since we have many constraints, it's better to pass a single function 
    # returning an array if using 'ineq' type? No, SLSQP takes a list of dicts.
    # Or we can define a single constraint function that returns an array 
    # and use type='ineq'? 
    # Actually, SLSQP `constraints` is a list of dictionaries.
    # Each dict can have 'type', 'fun', 'jac'.
    # If 'fun' returns an array, it's treated as multiple constraints of that type.
    
    cons_slsqp = ({'type': 'ineq', 'fun': constraint},)
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=cons_slsqp,
            options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
        )
        
        if res.x is not None:
            return res.x[:N*2].reshape(N, 2), res.x[-1]
            
    except Exception:
        pass
        
    return centers, r

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal Grids (best starting point for circle packing)
    for sp in [0.14, 0.15, 0.16, 0.17]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 2. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Refine each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        alloc_time = min(2.0, time_left / max(1, len(inits) - i))
        
        # Step 1: Expand radius from a small initial value
        # Start with a radius that is definitely feasible (e.g., 0.05)
        r0 = 0.05
        # Check if init is feasible for r0
        # If not, scale init down? Or just use r0 as is and let expansion fix it?
        # The expansion function moves points to resolve overlaps.
        
        # Let's initialize centers to the init, and try to expand from r=0.05
        # But if init points are closer than 0.1, it will take time to push them apart.
        # Better: Initialize with a valid configuration for r0.
        # For hex grid, if sp is large enough, it might be valid for larger r.
        
        # Let's just run expansion from a small r.
        refined_centers, refined_r = _expand_radius(init, r0, time_limit=alloc_time * 0.5)
        
        # Step 2: Polish with SLSQP
        if time.time() - start_time < DEADLINE - 5:
            time_left_polish = min(alloc_time * 0.5, DEADLINE - 5 - (time.time() - start_time))
            if time_left_polish > 0:
                polished_centers, polished_r = _slsqp_polish(refined_centers, refined_r, time_limit=time_left_polish)
                refined_centers = polished_centers
                refined_r = polished_r
        
        # Verify and update best
        actual_r = _radius(refined_centers)
        if actual_r > best_r:
            best_r = actual_r
            best_centers = refined_centers.copy()
            
    # If time remains, do more random restarts
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 1: break
        
        alloc_time = min(1.5, time_left * 0.9)
        
        refined_centers, refined_r = _expand_radius(c_rand, 0.05, time_limit=alloc_time * 0.5)
        
        if time.time() - start_time < DEADLINE - 5:
            time_left_polish = min(alloc_time * 0.5, DEADLINE - 5 - (time.time() - start_time))
            if time_left_polish > 0:
                polished_centers, polished_r = _slsqp_polish(refined_centers, refined_r, time_limit=time_left_polish)
                refined_centers = polished_centers
                refined_r = polished_r
        
        actual_r = _radius(refined_centers)
        if actual_r > best_r:
            best_r = actual_r
            best_centers = refined_centers.copy()

    return best_centers
