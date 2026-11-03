import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 50

def _true_score(centers):
    """Compute the true packing radius."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    if boundary <= 0:
        return 0.0
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r = min(boundary, min_dist / 2.0)
    return r

def _init_grid():
    """Grid initialization."""
    best_pts = None
    best_r = -1
    # Try various grid dimensions
    for rows in range(5, 11):
        for cols in range(5, 11):
            if rows * cols < N:
                continue
            xs = (np.arange(cols) + 0.5) / cols - 0.5
            ys = (np.arange(rows) + 0.5) / rows - 0.5
            grid = np.array([(x, y) for y in ys for x in xs])
            if len(grid) > N:
                # Keep points closest to center to maximize min distance to boundary
                dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
                indices = np.argsort(dists_to_center)[:N]
                pts = grid[indices]
            else:
                pts = grid
            r = _true_score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    return best_pts, best_r

def _init_hex():
    """Hexagonal initialization."""
    # Hexagonal packing is dense. We fit a hex grid into the square.
    # Radius of hex grid points roughly 1/sqrt(3*N) * adjustment
    # Let's just generate a dense hex grid and pick best N
    s = 0.05 # spacing
    pts = []
    x = -0.5
    row = 0
    while x < 0.5:
        y = -0.5
        offset = (row % 2) * s * np.sqrt(3) / 2
        while y < 0.5:
            if -0.5 <= x + offset <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x + offset, y])
            y += s
        x += s
        row += 1
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()[0]
    
    # Greedy selection of farthest points
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if len(selected) == 0:
            # Start from center
            dists_to_center = np.sqrt((remaining ** 2).sum(axis=1))
            idx = np.argmin(dists_to_center)
        else:
            sel = np.array(selected)
            dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    
    selected = np.array(selected)
    return selected, _true_score(selected)

def _langevin_relax(centers, target_r, time_budget=5.0, lr_init=0.01, T_init=0.001):
    """
    Relax configuration to satisfy constraints for target_r.
    Returns updated centers and success boolean.
    """
    start_time = time.time()
    centers = centers.copy()
    lr = lr_init
    T = T_init
    
    while time.time() - start_time < time_budget:
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Boundary forces
        # If x > 0.5 - r, push left. If x < -0.5 + r, push right.
        # Linear spring force
        bound = 0.5 - target_r
        for k in range(2):
            mask_r = centers[:, k] > bound
            forces[mask_r, k] -= (centers[mask_r, k] - bound) * 50.0
            mask_l = centers[:, k] < -bound
            forces[mask_l, k] += ((-bound) - centers[mask_l, k]) * 50.0
            
        # Pairwise forces
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion if dist < 2 * target_r
        threshold = 2.0 * target_r
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        # Force magnitude proportional to overlap
        # F = k * (threshold - dist)
        k_rep = 100.0
        force_mag = np.zeros_like(dists)
        force_mag[mask] = k_rep * (threshold - dists[mask])
        
        # Sum forces
        # force on i from j is force_mag[i,j] * (x_i - x_j) / dist[i,j]
        # This is vectorized as: sum_j (force_mag[i,j] / dist[i,j]) * (x_i - x_j)
        # Note: dists is symmetric.
        # Avoid division by zero (mask is False where dist=0 or inf)
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask]
        
        forces += (force_mag * inv_dists)[:, :, np.newaxis] * diffs
        
        # Langevin step
        noise = np.random.randn(N, 2) * np.sqrt(2 * T * lr)
        centers += forces * lr + noise
        
        # Clip to square (hard constraint)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check success: are all constraints satisfied?
        # Boundary
        if (0.5 - np.abs(centers)).min() < target_r - 1e-5:
            continue
        # Pairs
        if dists.min() < 2.0 * target_r - 1e-5:
            continue
            
        # Success
        return centers, True
        
        # Decay
        lr *= 0.999
        T *= 0.999
        if T < 1e-6:
            T = 1e-6

    return centers, False

def _soft_min_radius(centers, beta=100.0):
    """
    Smooth approximation of the radius for optimization.
    Returns negative radius (to be minimized).
    """
    boundary = 0.5 - np.abs(centers)
    b_min = boundary.min(axis=1) # min dist to boundary for each point
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft min of pairwise distances / 2
    # log-sum-exp trick for min
    # min(d_ij) approx -1/beta * log(sum(exp(-beta * d_ij)))
    # Using dists_sq to avoid sqrt in exp if possible? 
    # exp(-beta * sqrt(x)) is fine.
    
    dists = np.sqrt(dists_sq)
    
    # Scale for numerical stability
    # We want min of d_ij/2 and b_min
    # Let's combine them into one set of values
    # Values: d_ij/2 for i<j, and b_min_i for each i
    # This is O(N^2) values.
    
    # Alternative: Just optimize the soft min of pairwise and boundary separately and take min?
    # Better: Construct a matrix of all "constraints"
    # N(N-1)/2 pair constraints + N boundary constraints
    
    # For speed, we can just use the soft min of the distances.
    # Let's use a simpler surrogate: 
    # Objective = - (soft_min(pair_dists/2) * soft_min(boundary)) ? No.
    # Objective = - soft_min( [pair_dists/2, boundary_dists] )
    
    # Flatten pair dists
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    bound_dists = b_min
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    # m = min(all_dists)
    # soft_min = m - 1/beta * log( sum( exp(-beta * (all_dists - m))) )
    m = all_dists.min()
    if not np.isfinite(m):
        return 0.0
    exp_terms = np.exp(-beta * (all_dists - m))
    # Handle underflow/overflow
    # Since we subtracted min, exponents are <= 0
    log_sum = np.log(exp_terms.sum())
    soft_min_val = m - np.log(log_sum) / beta
    
    return -soft_min_val

def _lbfgs_refine(centers, time_budget=10.0):
    """
    Polish the configuration using L-BFGS-B on the soft-min radius.
    """
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 0.0 # Timeout, return arbitrary
        c = x.reshape(N, 2)
        # Hard clip to avoid infinite values in log/exp if it goes wild
        c = np.clip(c, -0.49, 0.49)
        return _soft_min_radius(c, beta=50.0)
        
    # Flatten centers
    x0 = centers.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
    except Exception:
        return centers, _true_score(centers)
        
    new_centers = res.x.reshape(N, 2)
    new_centers = np.clip(new_centers, -0.5, 0.5)
    
    return new_centers, _true_score(new_centers)

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -1.0
    
    # 1. Generate initial candidates
    inits = []
    grid_pts, grid_r = _init_grid()
    inits.append((grid_pts, grid_r))
    
    hex_pts, hex_r = _init_hex()
    inits.append((hex_pts, hex_r))
    
    # Random inits
    for _ in range(3):
        r_pts = (np.random.rand(N, 2) - 0.5) * 0.9
        r_r = _true_score(r_pts)
        inits.append((r_pts, r_r))
        
    # Sort by initial score
    inits.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Process each init
    # We allocate time to the best few inits.
    # Strategy: 
    # - Take the best init.
    # - Run radius expansion.
    # - If time remains, take next best and run shorter expansion.
    
    # We will use a loop over inits, with decreasing time budgets.
    
    for i, (init_pts, init_r) in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
            
        # Determine time budget for this init
        # Give more time to better starts
        if i == 0:
            t_budget = min(remaining * 0.5, 40.0)
        elif i == 1:
            t_budget = min(remaining * 0.3, 20.0)
        else:
            t_budget = min(remaining * 0.1, 10.0)
            
        if t_budget < 5:
            continue
            
        # Start with this init
        current_centers = init_pts.copy()
        current_r = init_r
        
        # Radius Expansion Loop
        # We try to increase r.
        # Step size depends on time remaining.
        # Start with a small step, increase if successful.
        
        # Estimate max possible r: 0.5 / ceil(sqrt(N)) approx
        # For N=50, sqrt(50)~7.07. 0.5/7 ~ 0.0714.
        # Known best is 0.07137.
        
        max_possible = 0.5 / np.ceil(np.sqrt(N))
        
        # If current r is already close to max, we are in refinement mode.
        
        # Langevin Relaxation with increasing r
        # We will run Langevin for a fixed time chunk, then try to bump r.
        
        t_chunk = 2.0 # seconds per Langevin run
        
        while time.time() - start_time < start_time + t_budget:
            elapsed = time.time() - start_time
            if elapsed > start_time + t_budget:
                break
                
            # Try to increase radius
            # Step size: start small, maybe adaptive
            # If we are at 0.07, we need very small steps.
            step_r = 0.0005
            if current_r < 0.05:
                step_r = 0.002
            elif current_r < 0.06:
                step_r = 0.001
            
            new_r = current_r + step_r
            
            # If new_r is too big (unrealistic), cap it
            if new_r > max_possible * 1.05:
                new_r = max_possible * 1.05
                
            # Run Langevin to try to achieve new_r
            # Time for this attempt
            t_langevin = min(t_chunk, (start_time + t_budget) - time.time())
            if t_langevin < 0.5:
                break
                
            new_centers, success = _langevin_relax(current_centers, new_r, time_budget=t_langevin)
            
            if success:
                current_centers = new_centers
                current_r = new_r
                # Update global best
                score = _true_score(current_centers)
                if score > best_score:
                    best_score = score
                    best_centers = current_centers.copy()
                # Keep stepping up
            else:
                # Failed to achieve new_r.
                # Maybe we can still improve position for current_r?
                # Or backtrack?
                # Let's try to polish current_r with Langevin for a bit longer
                # or just accept current state and move to next init or refinement.
                
                # If we've been stuck for a while, break to L-BFGS
                # Simple heuristic: if we failed, try a smaller step next time
                # Or just stop expansion for this init and go to refinement.
                break
                
        # Refinement phase for this init
        elapsed = time.time() - start_time
        remaining_for_refine = (start_time + t_budget) - elapsed
        if remaining_for_refine > 5:
            # Use L-BFGS to polish
            refined_centers, refined_score = _lbfgs_refine(current_centers, time_budget=remaining_for_refine * 0.8)
            if refined_score > best_score:
                best_score = refined_score
                best_centers = refined_centers.copy()
                
    # Final global refinement if time permits
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if best_centers is not None and remaining > 10:
        # One last polish on the global best
        final_centers, final_score = _lbfgs_refine(best_centers, time_budget=remaining * 0.8)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers.copy()
            
    if best_centers is None:
        # Fallback
        best_centers = grid_pts
        best_score = grid_r
        
    return best_centers
