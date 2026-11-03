import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill

    # 1. Initialization: Hexagonal Lattice
    # Target spacing for N=300 in unit circle.
    # Density of hex lattice: 2 / (sqrt(3) * s^2)
    # N * (sqrt(3) * s^2 / 2) = pi  =>  s = sqrt(2*pi / (sqrt(3)*N))
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
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0

    # Precompute indices for pairs
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    n_pairs = len(i_idx)

    # Helper function to compute score
    def compute_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        min_bound = np.min(1.0 - rad)
        return min(min_pair, min_bound)

    # Phase 1: Broad search with soft-min gradient ascent
    alpha = 0.05
    cooling = 0.999
    k_soft = 50.0  # Softer for broad search
    
    # Initial score
    best_score = compute_score(points)
    best_points = points.copy()
    
    last_improvement = 0.0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit - 20.0:  # Leave 20s for polish
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score + 1e-10:
            best_score = current_min
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute Soft-Min Weights
        shifted_vals = all_vals - current_min
        weights = np.exp(-k_soft * shifted_vals)
        weights /= np.sum(weights)
        
        # Compute Gradients
        grads = np.zeros_like(points)
        
        # Pair contributions
        pair_weights = weights[:n_pairs]
        # Avoid division by zero
        pair_dists = dists[i_idx, j_idx]
        pair_dists_safe = np.where(pair_dists > 1e-10, pair_dists, 1.0)
        vecs = (diff[i_idx, j_idx] / pair_dists_safe[:, np.newaxis]) * pair_weights[:, np.newaxis]
        
        np.add.at(grads, i_idx, vecs)
        np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[n_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Normalize gradients to prevent large steps
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-10:
            grads /= grad_norm
        
        # Update points
        points += alpha * grads
        
        # Project back inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cool down
        alpha *= cooling
        
        # Random perturbation to escape local minima
        if np.random.rand() < 0.03:
            points += np.random.normal(0, alpha * 3.0, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]

    # Phase 2: Polish with very sharp soft-min
    points = best_points.copy()
    alpha = 0.005
    cooling = 0.9995
    k_soft = 200.0  # Very sharp for polishing
    
    while time.time() - start_time < time_limit - 5.0:  # Leave 5s for final check
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score + 1e-12:
            best_score = current_min
            best_points = points.copy()
        
        # Compute Soft-Min Weights
        shifted_vals = all_vals - current_min
        weights = np.exp(-k_soft * shifted_vals)
        weights /= np.sum(weights)
        
        # Compute Gradients
        grads = np.zeros_like(points)
        
        # Pair contributions
        pair_weights = weights[:n_pairs]
        pair_dists = dists[i_idx, j_idx]
        pair_dists_safe = np.where(pair_dists > 1e-10, pair_dists, 1.0)
        vecs = (diff[i_idx, j_idx] / pair_dists_safe[:, np.newaxis]) * pair_weights[:, np.newaxis]
        
        np.add.at(grads, i_idx, vecs)
        np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[n_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Normalize gradients
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-10:
            grads /= grad_norm
        
        # Update points
        points += alpha * grads
        
        # Project back inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cool down
        alpha *= cooling

    # Final validation
    final_score = compute_score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure all points are strictly inside
    rad = np.linalg.norm(best_points, axis=1)
    mask = rad >= 1.0
    if np.any(mask):
        best_points[mask] /= (rad[mask, np.newaxis] + 1e-12)
        best_points[mask] *= 0.999999

    return best_points
