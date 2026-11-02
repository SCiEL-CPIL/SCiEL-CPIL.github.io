"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Parameters
    num_restarts = 30
    max_iter = 30000
    num_candidates = 20
    
    for restart in range(num_restarts):
        np.random.seed(restart * 12345 + 99)
        
        # Initialization: Jittered Grid
        # 5x5 grid = 25 points. Add 1 more.
        # Grid spacing 0.18
        grid_x = np.linspace(0.1, 0.9, 5)
        grid_y = np.linspace(0.1, 0.9, 5)
        gx, gy = np.meshgrid(grid_x, grid_y)
        centers = np.column_stack((gx.ravel(), gy.ravel()))[:N]
        
        # If N > 25, add random points
        if N > len(centers):
            extra = np.random.uniform(0.1, 0.9, size=(N - len(centers), 2))
            centers = np.vstack([centers, extra])
            
        # Add jitter to avoid symmetry
        centers += np.random.normal(0, 0.01, size=(N, 2))
        centers = np.clip(centers, 0.05, 0.95)
        
        radii = np.zeros(N)
        
        # Greedy Initialization
        order = np.random.permutation(N)
        for i in order:
            # Calculate max radius
            dists = np.linalg.norm(centers - centers[i], axis=1)
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            constraints = dists[mask] - radii[mask]
            min_constraint = np.min(constraints) if np.any(mask) else np.inf
            r_bound = min(centers[i][0], 1.0 - centers[i][0], centers[i][1], 1.0 - centers[i][1])
            radii[i] = max(0.0, min(min_constraint, r_bound))
            
        current_score = np.sum(radii)
        if current_score > best_score:
            best_score = current_score
            best_result = np.column_stack((centers, radii)).copy()
            
        # Local Search
        for _ in range(max_iter):
            i_min = np.argmin(radii)
            
            # Generate candidates
            # 10 random global
            cands = np.random.uniform(0, 1, size=(num_candidates // 2, 2))
            # 10 local perturbations
            steps = np.random.uniform(0.01, 0.1, size=(num_candidates - num_candidates // 2, 1))
            local_cands = centers[i_min] + np.random.normal(0, 1, size=(num_candidates - num_candidates // 2, 2)) * steps
            local_cands = np.clip(local_cands, 0, 1)
            
            all_cands = np.vstack([cands, local_cands])
            
            # Evaluate all candidates vectorized
            # dists: (num_cands, N)
            dists = np.linalg.norm(centers[None, :, :] - all_cands[:, None, :], axis=2)
            
            # Constraints: (num_cands, N)
            # r_i <= d_ij - r_j
            constraints = dists - radii[None, :]
            
            # Mask out self (idx i_min)
            mask = np.ones(N, dtype=bool)
            mask[i_min] = False
            constraints_masked = constraints[:, mask]
            
            min_constraints = np.min(constraints_masked, axis=1)
            
            # Boundary constraints for each candidate
            r_bounds = np.min(np.stack([
                all_cands[:, 0],
                1.0 - all_cands[:, 0],
                all_cands[:, 1],
                1.0 - all_cands[:, 1]
            ], axis=1), axis=1)
            
            new_radii = np.min(np.stack([min_constraints, r_bounds], axis=1), axis=1)
            new_radii = np.maximum(0.0, new_radii)
            
            best_idx = np.argmax(new_radii)
            best_new_r = new_radii[best_idx]
            best_new_pos = all_cands[best_idx]
            
            if best_new_r > radii[i_min]:
                centers[i_min] = best_new_pos
                radii[i_min] = best_new_r
                current_score += (best_new_r - radii[i_min]) # Wait, radii[i_min] was updated? No, we use old value for delta?
                # Let's just recalc sum for safety or track delta correctly.
                # delta = best_new_r - old_radii[i_min]
                # current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_result = np.column_stack((centers, radii)).copy()
            else:
                # No improvement found for smallest circle.
                # Try a random "kick" to escape local optimum
                # Pick a random circle and move it randomly, accepting even if radius decreases (with prob)
                # Or just perturb the smallest one significantly.
                if _ % 10 == 0:
                    # Perturb smallest circle
                    new_pos = centers[i_min] + np.random.normal(0, 0.05, size=2)
                    new_pos = np.clip(new_pos, 0, 1)
                    
                    # Recalc radius
                    dists = np.linalg.norm(centers - new_pos, axis=1)
                    mask = np.ones(N, dtype=bool)
                    mask[i_min] = False
                    constraints = dists[mask] - radii[mask]
                    min_constraint = np.min(constraints) if np.any(mask) else np.inf
                    r_bound = min(new_pos[0], 1.0 - new_pos[0], new_pos[1], 1.0 - new_pos[1])
                    new_r = max(0.0, min(min_constraint, r_bound))
                    
                    # Accept if not too bad
                    if new_r > radii[i_min] * 0.5:
                        current_score += (new_r - radii[i_min])
                        centers[i_min] = new_pos
                        radii[i_min] = new_r
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_result = np.column_stack((centers, radii)).copy()

    return best_result
# EVOLVE-BLOCK-END
