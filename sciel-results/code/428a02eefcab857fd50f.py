import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initialization: Hexagonal Lattice
    # Target spacing for N=300 in unit circle.
    # Area of unit circle = pi. Area per circle ~ pi/N.
    # Hexagonal packing density ~ 0.9069.
    # s ~ sqrt(pi / (N * 0.9069)) / sqrt(3/4) ... roughly
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    # Generate a slightly denser lattice to have more points, then prune
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Fill with random points if lattice was too sparse (unlikely with 0.95 factor)
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    
    # Compute initial score
    def compute_score(pts):
        if len(pts) != N:
            return -1.0
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Boundary distances
        rad = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.max(rad)
        return min(min_pair, min_bound)

    best_score = compute_score(points)

    # 2. Build Neighbor List (Static Topology)
    # We assume the local topology (which points are neighbors) remains roughly stable.
    # We define neighbors as points within a certain radius in the initial configuration.
    # This avoids O(N^2) checks every step.
    # Let's use a radius slightly larger than the initial spacing to be safe.
    neighbor_radius = s_init * 2.5
    
    diff_init = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_init = np.linalg.norm(diff_init, axis=2)
    np.fill_diagonal(dists_init, np.inf)
    
    # Get indices of neighbors
    i_idx, j_idx = np.where(dists_init < neighbor_radius)
    # Ensure unique pairs i < j
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    # Store unit vectors for these initial neighbor pairs? 
    # No, we should recompute unit vectors every step to be accurate, 
    # but we only compute them for these pairs.
    
    # 3. Optimization: Vectorized Soft-Min Gradient Ascent
    # We use a soft-min of the minimum distance.
    # Objective: Maximize min(d_ij, b_i)
    # Soft Min: -1/k * log(sum(exp(-k * d_ij)))
    # Gradient of soft min w.r.t. positions gives us the direction to move.
    
    # Hyperparameters
    alpha = 0.05
    cooling = 0.999
    k_soft = 50.0  # Start with moderate sharpness
    k_polish = 200.0 # Sharper for final phase
    steps_per_k = 1000 # Increase sharpness every 1000 steps
    
    step_count = 0
    
    while time.time() - start_time < time_limit:
        # Check if we should increase sharpness
        if step_count > 0 and step_count % steps_per_k == 0:
            k_soft = min(k_soft * 1.2, k_polish)
            
        # Compute distances for neighbor pairs only
        # diff[i, j] = p_i - p_j
        diff = points[i_idx] - points[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        safe_dists = np.maximum(dists, 1e-9)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        # Pairwise contribution to soft-min gradient
        # We want to maximize the min distance.
        # Soft min m = -1/k log(sum exp(-k d))
        # grad m = (1/k) * sum(exp(-k d) * grad d) / sum(exp(-k d))
        # Note: grad d_ij w.r.t p_i is -unit_vec (if d = |p_i - p_j|, grad is (p_i-p_j)/|p_i-p_j|)
        # Wait, d = |p_i - p_j|. grad_{p_i} d = (p_i - p_j) / |p_i - p_j| = unit_vec.
        # We want to INCREASE d. So we move p_i in direction +unit_vec.
        # The soft min gradient points in the direction of steepest increase of the soft min.
        # Since soft min approximates min, and we want to maximize min, we follow the gradient.
        
        # Weights for soft min
        # To avoid overflow, subtract max exponent
        exponents = -k_soft * dists
        max_exp = np.max(exponents)
        weights = np.exp(exponents - max_exp)
        sum_weights = np.sum(weights)
        
        # Normalize weights
        if sum_weights < 1e-9:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= sum_weights
            
        # Gradient from pairs
        # grad_{p_i} soft_min += weight * unit_vec
        # grad_{p_j} soft_min -= weight * unit_vec
        grads = np.zeros_like(points)
        
        # Scatter add
        np.add.at(grads, i_idx, unit_vecs * weights[:, np.newaxis])
        np.add.at(grads, j_idx, -unit_vecs * weights[:, np.newaxis])
        
        # Boundary contribution
        # We want to maximize min(1 - |p_i|).
        # Let b_i = 1 - |p_i|. We want to maximize min(b_i).
        # Soft min of b_i.
        # grad_{p_i} b_i = -p_i / |p_i|
        # We want to increase b_i, so move p_i in direction -p_i/|p_i| (towards origin).
        
        rad = np.linalg.norm(points, axis=1)
        safe_rad = np.maximum(rad, 1e-9)
        bound_vecs = -points / safe_rad[:, np.newaxis]
        
        b_vals = 1.0 - rad
        
        # We need to combine pair distances and boundary distances in the soft min.
        # The objective is min( min_pair_dists, min_bound_dists ).
        # This is equivalent to min( all_pair_dists, all_bound_dists ).
        # Let's compute a unified soft min over all constraints.
        
        # Pair constraints values: dists (we want to maximize these)
        # Bound constraints values: b_vals (we want to maximize these)
        
        # Combine values
        all_vals = np.concatenate([dists, b_vals])
        
        # Recompute weights over the combined set
        exponents_all = -k_soft * all_vals
        max_exp_all = np.max(exponents_all)
        weights_all = np.exp(exponents_all - max_exp_all)
        sum_weights_all = np.sum(weights_all)
        
        if sum_weights_all < 1e-9:
            weights_all = np.ones_like(weights_all) / len(weights_all)
        else:
            weights_all /= sum_weights_all
            
        # Split weights back
        pair_weights = weights_all[:len(dists)]
        bound_weights = weights_all[len(dists):]
        
        # Recompute grads with correct weights
        grads = np.zeros_like(points)
        np.add.at(grads, i_idx, unit_vecs * pair_weights[:, np.newaxis])
        np.add.at(grads, j_idx, -unit_vecs * pair_weights[:, np.newaxis])
        
        # Boundary grads
        grads += bound_vecs * bound_weights[:, np.newaxis]
        
        # Update
        points += alpha * grads
        
        # Project back inside circle (hard constraint)
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cooling
        alpha *= cooling
        
        # Random perturbation to escape local minima
        if np.random.rand() < 0.01:
            points += np.random.normal(0, alpha * 5.0, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
                
        step_count += 1
        
        # Evaluate score periodically
        if step_count % 100 == 0:
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

    return best_points
