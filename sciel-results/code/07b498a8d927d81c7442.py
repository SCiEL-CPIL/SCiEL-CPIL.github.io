import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Better lattice structure
    # Use a 5x5x4 grid but with better spacing and FCC-like offsets
    # The cube is [-0.5, 0.5]^3, so we need 100 points
    
    # Strategy: Use a 5x5x4 grid with alternating layer offsets (FCC-like)
    # Layer spacing in z: 1.0/4 = 0.25 (4 layers)
    # In each layer, use 5x5 grid with spacing 0.25 (5 points per row)
    # But 5*5*4 = 100 exactly!
    
    # Better: Use spacing that maximizes minimum distance
    # If we use 5x5 per layer with 4 layers, spacing in x,y = 1.0/4 = 0.25
    # But that gives points at edges. Let's use 5 points with margin.
    
    # Actually, for 5 points in [-0.5, 0.5], spacing = 1.0/4 = 0.25
    # Points at: -0.5, -0.25, 0, 0.25, 0.5 -> but these are at the boundary
    # We need margin for radius r, so effective spacing should be > 2r
    
    # Let's try: 5 points with spacing s, centered at 0
    # Points: -2s, -s, 0, s, 2s
    # Constraint: 2s + r <= 0.5, and 2r <= s (for adjacent spheres)
    # From 2s <= 0.5 - r, and s >= 2r
    # Maximize r: 2s = 0.5 - r, s = 2r -> 4r = 0.5 - r -> 5r = 0.5 -> r = 0.1
    # That's less than 0.1114. So we need a better structure.
    
    # The optimal packing uses a non-uniform structure. Let's start with a 
    # reasonable grid and let the optimizer find the better arrangement.
    
    # Use a 5x5x4 grid with FCC-like offsets in alternating layers
    # Layer 0: 5x5 grid at z = -0.375
    # Layer 1: 5x5 grid offset by (0.125, 0.125) at z = -0.125  
    # Layer 2: 5x5 grid at z = 0.125
    # Layer 3: 5x5 grid offset by (0.125, 0.125) at z = 0.375
    
    # Spacing in x,y for 5 points: 0.25
    # But with offset, the effective spacing between layers improves
    
    xs_base = np.linspace(-0.375, 0.375, 5)  # spacing 0.1875
    ys_base = np.linspace(-0.375, 0.375, 5)
    
    # Actually, let's use the full extent with margin
    # For 5 points, use spacing = 0.2, centered
    # Points: -0.4, -0.2, 0, 0.2, 0.4
    # Margin = 0.1, so r <= 0.1 from walls
    # Adjacent spacing = 0.2, so r <= 0.1 from neighbors
    # That gives r = 0.1, still below target.
    
    # The key insight: the optimal packing is NOT a simple grid. 
    # We need to start from a reasonable configuration and let the 
    # soft-min optimization find the better arrangement.
    
    # Use a 5x5x4 grid with small random perturbation
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    points = np.array(points)
    
    # Add small noise to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)
    
    # Clip to cube with margin
    points = np.clip(points, -0.45, 0.45)
    
    # Precompute pairs
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    
    # Wall gradient vectors
    grad_vecs = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1]
    ])
    
    # Optimization parameters
    beta = 20.0  # Start higher for sharper soft-min
    step_size = 0.005  # Smaller initial step
    best_radius = 0.0
    best_points = points.copy()
    
    # Helper to compute radius
    def compute_radius(pts):
        d_ij = np.linalg.norm(pts[i_idx] - pts[j_idx], axis=1)
        min_pair_dist = np.min(d_ij)
        wall_dists = np.min(np.minimum(pts + 0.5, 0.5 - pts), axis=1)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    # Initial evaluation
    current_radius = compute_radius(points)
    if current_radius > best_radius:
        best_radius = current_radius
        best_points = points.copy()
    
    # Main optimization loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Compute distances
        d_vecs = points[i_idx] - points[j_idx]
        d_ij = np.linalg.norm(d_vecs, axis=1)
        
        # Avoid division by zero
        d_ij_safe = np.where(d_ij < 1e-10, 1e-10, d_ij)
        
        # Pair gradients (unit vectors)
        unit_vecs = d_vecs / d_ij_safe[:, np.newaxis]
        
        # Wall distances (N, 6)
        wall_dists_2d = np.zeros((N, 6))
        wall_dists_2d[:, 0] = 0.5 - points[:, 0]
        wall_dists_2d[:, 1] = points[:, 0] + 0.5
        wall_dists_2d[:, 2] = 0.5 - points[:, 1]
        wall_dists_2d[:, 3] = points[:, 1] + 0.5
        wall_dists_2d[:, 4] = 0.5 - points[:, 2]
        wall_dists_2d[:, 5] = points[:, 2] + 0.5
        
        S_pairs = d_ij / 2.0
        S_walls = wall_dists_2d.flatten()
        
        # Combine S
        S = np.concatenate([S_pairs, S_walls])
        
        # Soft-min weights
        E = -beta * S
        E_max = np.max(E)
        W = np.exp(E - E_max)
        W_sum = np.sum(W)
        W_norm = W / W_sum
        
        # Gradients
        W_pairs = W_norm[:len(S_pairs)]
        
        grad_points = np.zeros((N, 3))
        
        # Pair gradients
        contrib_i = 0.5 * unit_vecs * W_pairs[:, np.newaxis]
        contrib_j = -0.5 * unit_vecs * W_pairs[:, np.newaxis]
        
        np.add.at(grad_points, i_idx, contrib_i)
        np.add.at(grad_points, j_idx, contrib_j)
        
        # Wall gradients
        W_walls = W_norm[len(S_pairs):]
        W_walls_2d = W_walls.reshape(N, 6)
        grad_wall_points = W_walls_2d @ grad_vecs
        grad_points += grad_wall_points
        
        # Adaptive step size: reduce if radius decreased
        current_radius_before = compute_radius(points)
        
        # Update
        points += step_size * grad_points
        points = np.clip(points, -0.5, 0.5)
        
        current_radius = compute_radius(points)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            # If improved, keep step size or increase slightly
            step_size = min(step_size * 1.02, 0.01)
        else:
            # If not improved, revert and reduce step size
            points = best_points.copy()
            step_size *= 0.95
        
        # Increase beta to sharpen the soft-min
        beta *= 1.03
        if beta > 200:
            beta = 200
        
        # Check time more frequently
        if iteration % 10 == 0:
            if time.time() > deadline - 5:
                break
    
    # Final polish: Use higher precision with smaller steps
    if time.time() < deadline - 2:
        polish_start = time.time()
        polish_deadline = polish_start + 3.0
        
        points = best_points.copy()
        beta_polish = 100.0
        step_polish = 0.001
        
        while time.time() < polish_deadline:
            d_vecs = points[i_idx] - points[j_idx]
            d_ij = np.linalg.norm(d_vecs, axis=1)
            d_ij_safe = np.where(d_ij < 1e-10, 1e-10, d_ij)
            unit_vecs = d_vecs / d_ij_safe[:, np.newaxis]
            
            wall_dists_2d = np.zeros((N, 6))
            wall_dists_2d[:, 0] = 0.5 - points[:, 0]
            wall_dists_2d[:, 1] = points[:, 0] + 0.5
            wall_dists_2d[:, 2] = 0.5 - points[:, 1]
            wall_dists_2d[:, 3] = points[:, 1] + 0.5
            wall_dists_2d[:, 4] = 0.5 - points[:, 2]
            wall_dists_2d[:, 5] = points[:, 2] + 0.5
            
            S_pairs = d_ij / 2.0
            S_walls = wall_dists_2d.flatten()
            S = np.concatenate([S_pairs, S_walls])
            
            E = -beta_polish * S
            E_max = np.max(E)
            W = np.exp(E - E_max)
            W_sum = np.sum(W)
            W_norm = W / W_sum
            
            W_pairs = W_norm[:len(S_pairs)]
            grad_points = np.zeros((N, 3))
            
            contrib_i = 0.5 * unit_vecs * W_pairs[:, np.newaxis]
            contrib_j = -0.5 * unit_vecs * W_pairs[:, np.newaxis]
            np.add.at(grad_points, i_idx, contrib_i)
            np.add.at(grad_points, j_idx, contrib_j)
            
            W_walls = W_norm[len(S_pairs):]
            W_walls_2d = W_walls.reshape(N, 6)
            grad_wall_points = W_walls_2d @ grad_vecs
            grad_points += grad_wall_points
            
            points += step_polish * grad_points
            points = np.clip(points, -0.5, 0.5)
            
            current_radius = compute_radius(points)
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()
            
            beta_polish = min(beta_polish * 1.1, 500)
            step_polish *= 0.98
    
    return best_points
