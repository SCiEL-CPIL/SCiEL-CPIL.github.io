import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin for 120s limit

def compute_initial_hex():
    """
    Generates a hexagonal lattice initialization for N circles.
    Attempts to fit them into the square [-0.5, 0.5].
    """
    # Approximate number of rows and cols
    # Hex packing density is higher.
    # Let's try to fit a grid of hex points.
    # y spacing: sqrt(3) * r
    # x spacing: r
    # For a square, we can just generate a large hex lattice and center/crop it.
    
    # Let's estimate r ~ 0.07
    # y_step = 0.07 * 1.732 ~ 0.12
    # x_step = 0.07
    # In 1.0 width, we can fit ~14 circles in x, ~8 in y.
    # 14*8 = 112 points. We need 50.
    
    # Generate points in a hex pattern
    points = []
    r_est = 0.07
    y_step = r_est * np.sqrt(3)
    x_step = r_est
    
    # Generate enough points
    max_pts = 100
    row = 0
    while len(points) < max_pts:
        y = -0.5 + r_est + row * y_step
        if y > 0.5 - r_est:
            break
        col = 0
        offset = (r_est / 2) if row % 2 == 1 else 0
        while True:
            x = -0.5 + r_est + offset + col * x_step
            if x > 0.5 - r_est:
                break
            points.append([x, y])
            col += 1
        row += 1
    
    # If we don't have enough, add random ones or expand
    # For N=50, the above loop with r_est=0.07 should generate plenty.
    # Let's check count.
    # If len(points) < N, we need to generate more or use a different init.
    # Let's just take the first N.
    
    if len(points) < N:
        # Fallback to random
        points = []
        for _ in range(N):
            points.append([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)])
    
    pts = np.array(points[:N])
    # Add small random jitter to break symmetry
    pts += np.random.normal(0, 0.001, size=pts.shape)
    
    # Ensure they are inside
    pts[:, 0] = np.clip(pts[:, 0], -0.499, 0.499)
    pts[:, 1] = np.clip(pts[:, 1], -0.499, 0.499)
    
    return pts

def compute_distance_matrix(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def compute_wall_dists(pts):
    dx = np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5)
    dy = np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    return np.minimum(dx, dy)

def compute_energy(pts, r, d_mat=None, d_walls=None):
    if d_mat is None:
        d_mat = compute_distance_matrix(pts)
    if d_walls is None:
        d_walls = compute_wall_dists(pts)
        
    # Pairwise
    # We sum over i < j
    # Use triu indices for efficiency in full recompute, but here we might pass precomputed
    # For simplicity in this step, we recompute violations
    tri_idx = np.triu_indices(N, 1)
    d_pairs = d_mat[tri_idx]
    v_pairs = np.maximum(0, 2*r - d_pairs)
    e_pairs = np.sum(v_pairs**2)
    
    # Walls
    v_walls = np.maximum(0, r - d_walls)
    e_walls = np.sum(v_walls**2)
    
    return e_pairs + e_walls

def sa_optimize(pts, r, max_iters, t0, t1, step0, step1):
    """
    Runs Simulated Annealing to minimize energy for a fixed radius r.
    Returns optimized pts.
    """
    current_pts = pts.copy()
    current_d_mat = compute_distance_matrix(current_pts)
    current_d_walls = compute_wall_dists(current_pts)
    current_energy = compute_energy(current_pts, r, current_d_mat, current_d_walls)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    # Precompute indices for pairwise distances to update quickly
    # We will update d_mat and d_walls incrementally
    
    for i_iter in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Cooling
        progress = i_iter / max_iters
        t = t0 * (t1 / t0) ** progress
        step_size = step0 * (step1 / step0) ** progress
        
        # Pick random index
        idx = np.random.randint(N)
        
        # Save old state for rollback
        old_x = current_pts[idx].copy()
        old_d_mat_row = current_d_mat[idx].copy()
        old_d_walls_val = current_d_walls[idx]
        old_energy = current_energy
        
        # Perturb
        delta = np.random.normal(0, step_size, size=2)
        current_pts[idx] += delta
        
        # Clip to square to prevent invalid states in intermediate steps?
        # The energy penalty should handle it, but clipping helps stability.
        current_pts[idx, 0] = np.clip(current_pts[idx, 0], -0.5, 0.5)
        current_pts[idx, 1] = np.clip(current_pts[idx, 1], -0.5, 0.5)
        
        # Update distances
        # Distance from idx to all j != idx
        diff = current_pts[idx] - current_pts
        new_d_row = np.sqrt(np.sum(diff**2, axis=1))
        new_d_row[idx] = 0 # Self distance
        
        # Update d_mat
        # d_mat is symmetric
        # We need to update row idx and col idx
        # Note: d_mat[idx, j] and d_mat[j, idx]
        
        # Calculate change in energy
        # Old violations for pairs (idx, j)
        # We need to compare old and new distances for all j != idx
        
        # Get old distances for idx
        old_d_row = old_d_mat_row
        
        # New distances for idx
        new_d_row_clean = new_d_row.copy()
        new_d_row_clean[idx] = np.inf # Ignore self
        
        old_d_row_clean = old_d_row.copy()
        old_d_row_clean[idx] = np.inf
        
        # Calculate violations
        # v = max(0, 2r - d)
        # E_pair = sum(v^2)
        
        # Only j != idx contribute to pair energy involving idx
        # Indices j != idx
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        
        d_old = old_d_row_clean[mask]
        d_new = new_d_row_clean[mask]
        
        v_old = np.maximum(0, 2*r - d_old)
        v_new = np.maximum(0, 2*r - d_new)
        
        delta_e_pairs = np.sum(v_new**2) - np.sum(v_old**2)
        
        # Wall energy
        new_d_walls_val = compute_wall_dists(current_pts[idx:idx+1])[0]
        v_w_old = max(0, r - old_d_walls_val)
        v_w_new = max(0, r - new_d_walls_val)
        
        delta_e_walls = v_w_new**2 - v_w_old**2
        
        delta_e = delta_e_pairs + delta_e_walls
        
        new_energy = old_energy + delta_e
        
        # Metropolis
        if new_energy < old_energy or np.random.rand() < np.exp(-delta_e / t):
            # Accept
            current_energy = new_energy
            # Update d_mat
            current_d_mat[idx, :] = new_d_row
            current_d_mat[:, idx] = new_d_row
            current_d_walls[idx] = new_d_walls_val
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = current_pts.copy()
        else:
            # Reject
            current_pts[idx] = old_x
            
    return best_pts

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    best_r = 0
    best_pts = None
    
    # Initial radius estimate
    # Grid gives 0.0625. Hex is better.
    # Let's start with 0.06 and increase.
    r = 0.06
    
    # Run multiple SA starts
    num_starts = 5
    iters_per_run = 50000
    
    for start in range(num_starts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Generate initial points
        if start == 0:
            pts = compute_initial_hex()
        else:
            # Jittered hex or random
            pts = compute_initial_hex()
            pts += np.random.normal(0, 0.01, size=pts.shape)
            pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
            pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
            
        # SA parameters
        t0 = 0.01
        t1 = 1e-5
        step0 = 0.01
        step1 = 0.001
        
        # Adaptive radius loop
        current_r = r
        max_radius_steps = 50
        no_improve_count = 0
        
        for r_step in range(max_radius_steps):
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            # Run SA for this radius
            optimized_pts = sa_optimize(pts, current_r, iters_per_run, t0, t1, step0, step1)
            
            # Check energy
            d_mat = compute_distance_matrix(optimized_pts)
            d_walls = compute_wall_dists(optimized_pts)
            energy = compute_energy(optimized_pts, current_r, d_mat, d_walls)
            
            # Calculate actual min distance
            # Min pair dist
            tri_idx = np.triu_indices(N, 1)
            min_pair_d = np.min(d_mat[tri_idx])
            min_wall_d = np.min(d_walls)
            actual_r = min(min_pair_d, min_wall_d) / 2
            
            if actual_r > best_r:
                best_r = actual_r
                best_pts = optimized_pts.copy()
                
            # Decide if we can increase r
            # If energy is small, we can increase
            if energy < 1e-4:
                current_r *= 1.01
                no_improve_count = 0
            else:
                # If energy is high, maybe decrease or stop
                # If we can't pack at current_r, we shouldn't go higher
                # But SA might have failed. 
                # Let's try to decrease r slightly to find a valid packing, then increase
                # Or just stop increasing for this start
                if no_improve_count > 5:
                    break
                no_improve_count += 1
                # Keep current_r same, try again? Or break?
                # Let's break to save time, as we likely hit a local minimum or limit
                break
                
            # Update pts for next radius step
            pts = optimized_pts
            
        # After radius loop, try to refine at best_r
        # Run SA one more time at best_r to polish
        if best_pts is not None and best_r > 0:
             # Use best_r for final polish
             final_pts = sa_optimize(best_pts, best_r, 10000, 0.001, 1e-6, 0.001, 0.0001)
             # Check if it improved
             d_mat = compute_distance_matrix(final_pts)
             d_walls = compute_wall_dists(final_pts)
             min_pair_d = np.min(d_mat[np.triu_indices(N, 1)])
             min_wall_d = np.min(d_walls)
             final_r = min(min_pair_d, min_wall_d) / 2
             
             if final_r > best_r:
                 best_r = final_r
                 best_pts = final_pts

    # Fallback if no valid result
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # Ensure validity
    best_pts[:, 0] = np.clip(best_pts[:, 0], -0.4999, 0.4999)
    best_pts[:, 1] = np.clip(best_pts[:, 1], -0.4999, 0.4999)
    
    return best_pts
