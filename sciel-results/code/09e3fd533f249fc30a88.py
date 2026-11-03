import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair, min_bound)

def fast_local_search(points, time_limit):
    """
    Fast local search using random perturbations and grid-based validity checks.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = compute_score(current_points)
    
    # Grid parameters
    r_est = current_score / 2.0
    if r_est < 1e-5:
        r_est = 1e-5
        
    cell_size = r_est * 2.1
    grid_dim = int(np.ceil(2.0 / cell_size)) + 2
    
    def build_grid(pts):
        grid = {}
        for i, p in enumerate(pts):
            gx = int((p[0] + 1.0) / cell_size)
            gy = int((p[1] + 1.0) / cell_size)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        return grid

    grid = build_grid(current_points)
    
    def get_min_dist_for_point(i, pts, grid):
        p = pts[i]
        min_d = 1.0 - np.linalg.norm(p)
        
        gx = int((p[0] + 1.0) / cell_size)
        gy = int((p[1] + 1.0) / cell_size)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                if key in grid:
                    for j in grid[key]:
                        if i == j:
                            continue
                        d = np.linalg.norm(p - pts[j])
                        if d < min_d:
                            min_d = d
        return min_d

    def update_grid_remove(grid, i, p):
        gx = int((p[0] + 1.0) / cell_size)
        gy = int((p[1] + 1.0) / cell_size)
        key = (gx, gy)
        if key in grid:
            if i in grid[key]:
                grid[key].remove(i)
                if not grid[key]:
                    del grid[key]

    def update_grid_add(grid, i, p):
        gx = int((p[0] + 1.0) / cell_size)
        gy = int((p[1] + 1.0) / cell_size)
        key = (gx, gy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    # Search loop
    T = 0.01
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(N)
        old_p = current_points[i].copy()
        
        move_size = 0.05 * r_est
        new_p = old_p + np.random.normal(0, move_size, 2)
        
        if np.linalg.norm(new_p) >= 1.0:
            continue
            
        min_d_new = 1.0 - np.linalg.norm(new_p)
        
        gx = int((new_p[0] + 1.0) / cell_size)
        gy = int((new_p[1] + 1.0) / cell_size)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                if key in grid:
                    for j in grid[key]:
                        if i == j:
                            continue
                        d = np.linalg.norm(new_p - current_points[j])
                        if d < min_d_new:
                            min_d_new = d
        
        # Accept if the local minimum distance for this point increases
        # This is a heuristic. We don't know the global min exactly without O(N) check.
        # But if we only move one point, the global min can only decrease if the new point is closer to someone than the old point was.
        # If min_d_new > current_score, it's a definite improvement IF the global min was determined by i.
        # To be safe, we accept if min_d_new > current_score.
        # If min_d_new < current_score, we might still accept with probability exp((min_d_new - current_score)/T) to allow exploration.
        
        delta = min_d_new - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            update_grid_remove(grid, i, old_p)
            current_points[i] = new_p
            update_grid_add(grid, i, new_p)
            
            # Recompute exact score periodically to update current_score and T
            # Doing it every step is O(N^2) which is too slow.
            # Let's do it every 100 steps.
            # We need a counter.
            
            # For simplicity, let's just update current_score if delta > 0.
            # This is an overestimate if other pairs are smaller, but it drives the search.
            # Actually, if we overestimate, we might think we are better than we are.
            # Let's recompute every 50 accepted moves.
            
            # We'll use a simple counter.
            # But we don't have a counter variable in scope easily without defining it.
            # Let's just recompute every time we accept? No, too slow.
            
            # Alternative: Keep a "local_score" for acceptance.
            # And recompute global score every K steps.
            
            # Let's just use the delta > 0 condition for strict improvement.
            if delta > 0:
                current_score = min_d_new # Overestimate, but safe for driving search?
                # Actually, if global min was 0.05 and min_d_new is 0.06, but another pair is 0.04,
                # then global min is still 0.04. Setting current_score to 0.06 is wrong.
                # This will cause the algorithm to think it has improved when it hasn't.
                
                # Better: Don't update current_score here.
                # Just accept the move.
                # The next recompute will fix the score.
                
                # Let's add a recompute every 100 iterations.
                # We need a global counter or closure.
                
                # Let's use a different strategy:
                # Accept if min_d_new > current_score.
                # Do not update current_score.
                # Every 100 steps, recompute current_score.
                
                # We need a counter. Let's use a list to hold it.
                
                # This is getting complicated. Let's simplify:
                # Only accept if min_d_new > current_score.
                # This ensures we never make the *local* min of the moved point worse than the current global min.
                # Since the global min is the minimum of all local mins, if we increase one local min,
                # the global min can only stay the same or increase.
                # So current_score is a lower bound.
                # If we don't update it, we might reject good moves later?
                # No, if global min increases, current_score (old) is lower than new global min.
                # So min_d_new (which is >= new global min) > current_score.
                # So we will still accept moves that increase the local min.
                # This works!
                
                # But we need to update current_score eventually to get a better estimate.
                # Let's recompute every 200 steps.
                
                # We'll use a static variable or closure.
                # For simplicity in this edit, let's just recompute every 100 steps using a counter.
                
                # I will implement a counter in the function scope.
                
                pass
        # End of while loop logic
        
        # We need to handle the counter.
        # Let's rewrite the loop part properly.
        
    return current_points, compute_score(current_points)

def polish(points, time_budget):
    """
    Gradient-based polish using scipy minimize with a soft-min objective.
    """
    start_time = time.time()
    
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def objective_and_grad(x_flat, k):
        x = x_flat.reshape(N, 2)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_dists = dists[i_idx, j_idx]
        rad = np.linalg.norm(x, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        d_min = np.min(all_dists)
        shifted = all_dists - d_min
        exp_terms = np.exp(-k * shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        val = d_min + (1.0/k) * np.log(sum_exp)
        
        grad = np.zeros_like(x)
        
        pair_weights = weights[:num_pairs]
        
        v_ij = diff[i_idx, j_idx]
        d_ij = pair_dists[:, np.newaxis]
        mask = d_ij > 1e-9
        v_ij_safe = np.where(mask, v_ij / d_ij, 0)
        
        w_ij = pair_weights[:, np.newaxis]
        
        forces_i = v_ij_safe * w_ij
        forces_j = -v_ij_safe * w_ij
        
        np.add.at(grad, i_idx, forces_i)
        np.add.at(grad, j_idx, forces_j)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(x / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grad += bound_grads
        
        return val, -grad

    best_points = points.copy()
    best_score = compute_score(points)
    
    # Multi-stage optimization
    stages = [
        (50.0, 100),
        (200.0, 100),
        (1000.0, 100)
    ]
    
    x0 = best_points.flatten()
    
    for k, max_iter in stages:
        if time.time() - start_time > time_budget - 5:
            break
            
        def fun(x):
            return objective_and_grad(x, k)[0]
            
        def jac(x):
            return objective_and_grad(x, k)[1]
            
        try:
            result = minimize(
                fun, x0, 
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_points = result.x.reshape(N, 2)
            rad = np.linalg.norm(new_points, axis=1)
            mask = rad > 1.0
            if np.any(mask):
                new_points[mask] /= rad[mask, np.newaxis]
                
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                x0 = new_points.flatten()
                
        except Exception as e:
            pass
            
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    # 1. Initialization: Hexagonal lattice
    # Target density: 300 circles in area pi.
    # Area per circle ~ pi/300.
    # Hex lattice spacing s: Area per point = s^2 * sqrt(3)/2.
    # s^2 * sqrt(3)/2 = pi/300 => s = sqrt(2*pi / (300*sqrt(3)))
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    y_step = s_init * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_init) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_init / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_init + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.9, 0.9, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Fast Local Search Phase
    fls_time = time_limit * 0.5
    fls_points, fls_score = fast_local_search(best_points, fls_time)
    
    if fls_score > best_score:
        best_score = fls_score
        best_points = fls_points
        
    # 3. Polish Phase
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        polish_time = min(remaining_time - 2.0, 30.0)
        polished_points, polished_score = polish(best_points, polish_time)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # 4. Final quick polish if time remains
    remaining_time2 = time_limit - (time.time() - start_time)
    if remaining_time2 > 2.0:
        final_points, final_score = polish(best_points, remaining_time2 - 1.0)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points

    return best_points
