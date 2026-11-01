"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # 1. Initialization: Hexagonal Grid
    # Estimate optimal radius to set grid spacing
    # r ~ 0.053, so distance ~ 0.106
    # Hex grid spacing a
    a = 0.11 
    
    # Generate hex grid points
    # Hex grid:
    # Row y = k * a * sqrt(3) / 2
    # x = i * a + (y / (a * sqrt(3) / 2)) % 2 * a / 2
    
    pts_list = []
    # Bounds for grid
    max_k = int(1.0 / (a * np.sqrt(3) / 2)) + 2
    max_i = int(1.0 / a) + 2
    
    for k in range(-max_k, max_k + 1):
        y = k * a * np.sqrt(3) / 2
        # Offset for even/odd rows
        offset = (a / 2) if (k % 2 != 0) else 0
        for i in range(-max_i, max_i + 1):
            x = i * a + offset
            pts_list.append([x, y])
            
    pts = np.array(pts_list)
    
    # Filter to inside unit circle (with some margin to allow movement)
    # We want points strictly inside, but we will optimize.
    # Let's keep points with norm < 0.95 to ensure we have enough "bulk"
    # and then we can move them outwards.
    # Actually, just keep those with norm < 1.0
    norms = np.linalg.norm(pts, axis=1)
    mask = norms < 0.98
    pts = pts[mask]
    
    # If we have more than N points, randomly select N
    if len(pts) > N:
        indices = np.random.choice(len(pts), N, replace=False)
        pts = pts[indices]
    else:
        # If we have fewer, add random points
        while len(pts) < N:
            # Sample random point in unit circle
            r = np.sqrt(np.random.random())
            theta = np.random.uniform(0, 2 * np.pi)
            pts = np.vstack([pts, [r * np.cos(theta), r * np.sin(theta)]])
            
    # Ensure exactly N points
    pts = pts[:N]
    
    # 2. Simulated Annealing
    def calc_score(p):
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Get upper triangle indices for min
        iu = np.triu_indices(N, 1)
        min_pair_dist = np.min(dists[iu])
        
        # Boundary distance
        norms = np.linalg.norm(p, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_score = calc_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # SA Parameters
    T_start = 0.005
    T_end = 1e-7
    cooling_rate = 0.9995
    move_sigma = 0.01
    num_points_to_move = 3 # Move 3 random points at a time
    
    start_time = time.time()
    time_budget = 110.0
    
    T = T_start
    it = 0
    
    while T > T_end:
        if time.time() - start_time > time_budget:
            break
            
        it += 1
        
        # Generate move: perturb num_points_to_move random points
        indices = np.random.choice(N, num_points_to_move, replace=False)
        noise = np.random.normal(0, move_sigma, size=(num_points_to_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += noise
        
        # Optional: Project points back into circle if they go far out?
        # No, let the score penalty handle it. But if they go way out, score is 0.
        # To be safe, if any point has norm > 1.5, revert that point?
        # Or just accept the bad score.
        
        new_score = calc_score(new_pts)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                pts = new_pts
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        
        # Adaptive move size?
        # If we are accepting too many, increase sigma. If too few, decrease.
        # For simplicity, keep constant or slowly decrease.
        # Let's slowly decrease move_sigma to refine.
        move_sigma *= 0.9999
        
    return best_pts
# EVOLVE-BLOCK-END
