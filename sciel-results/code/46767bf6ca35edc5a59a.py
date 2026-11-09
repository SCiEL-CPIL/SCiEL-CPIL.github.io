import numpy as np
import time

N = 100

def _compute_score_fast(centers):
    """
    Fast score computation using squared distances to avoid sqrt.
    Returns (radius, min_pair_sq_dist, min_wall_dist).
    """
    # Pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(sq_dists, np.inf)
    
    min_pair_sq_dist = np.min(sq_dists)
    min_pair_dist = np.sqrt(min_pair_sq_dist)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius, min_pair_sq_dist, min_wall_dist

def _get_active_points(centers, radius, tol=1e-5):
    """
    Identify points that are close to the limiting radius.
    Uses squared distances for speed.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Points involved in pairs with distance < 2*radius + tol
    threshold_sq = (2.0 * radius + tol) ** 2
    active_pairs = sq_dists < threshold_sq
    
    # Points close to walls
    wall_dists = 0.5 - np.abs(centers)
    active_walls = (wall_dists < (radius + tol)).any(axis=1)
    
    active_indices = np.unique(np.concatenate([
        np.where(active_pairs)[0],
        np.where(active_pairs)[1],
        np.where(active_walls)[0]
    ]))
    
    return active_indices

def _compute_score(centers):
    """
    Compute the maximum radius for the given centers.
    Returns just radius for compatibility.
    """
    radius, _, _ = _compute_score_fast(centers)
    return radius

def _soft_min_score(centers, beta=50.0):
    """
    Compute a smooth surrogate for the minimum radius using log-sum-exp.
    Returns a scalar value to be maximized.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Collect all constraints: pair distances / 2 and wall distances
    # For pairs, we want to maximize min(pair_dist / 2)
    # For walls, we want to maximize min(wall_dist)
    
    # Use log-sum-exp for soft-min
    # soft_min(x) = -log(sum(exp(-beta*x))) / beta
    # We want to maximize soft_min, so we minimize -soft_min
    
    # Pair terms: x_i = pair_dist_i / 2
    pair_dists = np.sqrt(sq_dists)
    pair_terms = pair_dists / 2.0
    
    # Wall terms: x_j = wall_dist_j
    wall_terms = wall_dists.flatten()
    
    # Combine all terms
    all_terms = np.concatenate([pair_terms.flatten(), wall_terms])
    
    # Filter out inf values (diagonal)
    all_terms = all_terms[np.isfinite(all_terms)]
    
    if len(all_terms) == 0:
        return 0.0
    
    # Soft-min: -log(sum(exp(-beta*x))) / beta
    # To avoid overflow, use max trick
    max_val = np.max(all_terms)
    exp_terms = np.exp(-beta * (all_terms - max_val))
    sum_exp = np.sum(exp_terms)
    
    soft_min = -(np.log(sum_exp) / beta) + max_val
    
    return soft_min

def _soft_min_grad(centers, beta=50.0):
    """
    Compute gradient of the soft-min score with respect to centers.
    Returns gradient array of shape (N, 3).
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pair distances
    pair_dists = np.sqrt(sq_dists)
    pair_terms = pair_dists / 2.0
    
    # Wall terms
    wall_terms = wall_dists.flatten()
    
    # Combine all terms
    all_terms = np.concatenate([pair_terms.flatten(), wall_terms])
    
    # Filter out inf values (diagonal)
    finite_mask = np.isfinite(all_terms)
    all_terms_finite = all_terms[finite_mask]
    
    if len(all_terms_finite) == 0:
        return np.zeros((N, 3))
    
    # Soft-min: -log(sum(exp(-beta*x))) / beta
    max_val = np.max(all_terms_finite)
    exp_terms = np.exp(-beta * (all_terms_finite - max_val))
    sum_exp = np.sum(exp_terms)
    
    # Weights for each term: w_i = exp(-beta*x_i) / sum(exp(-beta*x_j))
    weights = exp_terms / sum_exp
    
    # Gradient of soft_min w.r.t. x_i is -w_i
    # But we need gradient w.r.t. centers
    
    grad = np.zeros((N, 3))
    
    # Process pair terms
    n_pairs = len(pair_terms.flatten())
    pair_weights = weights[:n_pairs]
    wall_weights = weights[n_pairs:]
    
    # For each pair (i, j), x_ij = |c_i - c_j| / 2
    # dx_ij/dc_i = -(c_i - c_j) / |c_i - c_j|
    # dx_ij/dc_j = (c_i - c_j) / |c_i - c_j|
    
    # Unflatten pair terms
    pair_terms_2d = pair_terms
    pair_weights_2d = pair_weights.reshape(pair_terms.shape)
    
    # Gradient contribution from pairs
    for i in range(N):
        for j in range(i+1, N):
            if not np.isfinite(sq_dists[i, j]):
                continue
            d = pair_dists[i, j]
            if d < 1e-10:
                continue
            w = pair_weights_2d[i, j]
            # x_ij = d/2, dx/dc_i = -(c_i - c_j)/d, dx/dc_j = (c_i - c_j)/d
            # d(soft_min)/dc_i = -w * dx/dc_i = -w * (-(c_i-c_j)/d) = w*(c_i-c_j)/d
            # d(soft_min)/dc_j = -w * dx/dc_j = -w * ((c_i-c_j)/d) = -w*(c_i-c_j)/d
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec
            grad[j] -= w * vec
    
    # For wall terms: x_ik = 0.5 - |c_ik|
    # dx/dc_ik = -sign(c_ik)
    # d(soft_min)/dc_ik = -w * dx/dc_ik = w * sign(c_ik)
    
    wall_weights_2d = wall_weights.reshape(N, 3)
    signs = np.sign(centers)
    # Handle zero: sign(0) = 0, but derivative of |x| at 0 is undefined
    # Use subgradient: 0
    grad += wall_weights_2d * signs
    
    return grad

def construct():
    # Initialize with FCC lattice
    # FCC lattice points in a unit cube
    # Generate FCC points: (0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0) plus translations
    
    # Create FCC lattice
    a = 1.0  # lattice parameter
    fcc_points = []
    
    # Generate points in a grid that covers the cube
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    # Generate lattice vectors
    n_cells = 3  # 3x3x3 cells
    cell_size = 1.0 / n_cells
    
    for ix in range(n_cells + 1):
        for iy in range(n_cells + 1):
            for iz in range(n_cells + 1):
                origin = np.array([ix * cell_size, iy * cell_size, iz * cell_size])
                for b in basis:
                    point = origin + b * cell_size
                    # Check if point is within the cube [-0.5, 0.5]^3
                    if np.all(point >= -0.5 - 1e-10) and np.all(point <= 0.5 + 1e-10):
                        # Shift to center at origin
                        point_centered = point - 0.5
                        if np.all(point_centered >= -0.5 - 1e-10) and np.all(point_centered <= 0.5 + 1e-10):
                            fcc_points.append(point_centered)
    
    fcc_points = np.array(fcc_points)
    
    # Remove duplicates
    if len(fcc_points) > 0:
        # Use rounding to remove near-duplicates
        rounded = np.round(fcc_points, decimals=8)
        unique_idx = np.unique(rounded, axis=0)
        # Map back to original points
        seen = set()
        unique_points = []
        for p in fcc_points:
            key = tuple(np.round(p, decimals=8))
            if key not in seen:
                seen.add(key)
                unique_points.append(p)
        fcc_points = np.array(unique_points)
    
    # Select 100 points with largest minimum distance to boundary
    if len(fcc_points) >= N:
        # Compute distance to boundary for each point
        wall_dists = 0.5 - np.abs(fcc_points)
        min_wall_dist = np.min(wall_dists, axis=1)
        
        # Also consider pairwise distances
        # For now, just select by wall distance
        # Sort by min wall distance descending
        sorted_idx = np.argsort(-min_wall_dist)
        centers = fcc_points[sorted_idx[:N]].copy()
    else:
        # If not enough FCC points, fill with grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = np.vstack([fcc_points, grid])
        centers = centers[:N].copy()
    
    # Ensure centers are strictly inside the cube
    centers = np.clip(centers, -0.49, 0.49)
    
    best_radius, _, _ = _compute_score_fast(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    time_limit = 100.0  # Safety margin
    
    # Phase 1: Gradient-based optimization using soft-min surrogate
    # Use L-BFGS-B from scipy
    from scipy.optimize import minimize
    
    # Start with moderate beta and increase
    beta_schedule = [20, 50, 100, 200, 400]
    
    current_centers = centers.copy()
    current_radius, _, _ = _compute_score_fast(current_centers)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit - 30:
            break
        
        # Define objective and gradient for scipy
        def obj(x):
            c = x.reshape(-1, 3)
            c = np.clip(c, -0.49, 0.49)
            return -_soft_min_score(c, beta=beta)
        
        def grad(x):
            c = x.reshape(-1, 3)
            c = np.clip(c, -0.49, 0.49)
            g = -_soft_min_grad(c, beta=beta)
            return g.flatten()
        
        # Run L-BFGS-B
        try:
            result = minimize(
                obj, 
                current_centers.flatten(), 
                jac=grad, 
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 100, 'maxfun': 200}
            )
            
            new_centers = result.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.49, 0.049)
            
            new_radius, _, _ = _compute_score_fast(new_centers)
            
            if new_radius > current_radius:
                current_centers = new_centers
                current_radius = new_radius
        except Exception:
            pass
    
    # Update best if improved
    if current_radius > best_radius:
        best_radius = current_radius
        best_centers = current_centers.copy()
    
    # Phase 2: Local search refinement
    if time.time() - start_time > time_limit - 20:
        # Not enough time for local search
        final_radius, _, _ = _compute_score_fast(best_centers)
        if final_radius > best_radius:
            best_radius = final_radius
        return best_centers
    
    # Continue with local search from the gradient-optimized position
    centers = current_centers.copy()
    current_radius = current_radius if current_radius > best_radius else best_radius
    if current_radius < best_radius:
        centers = best_centers.copy()
        current_radius = best_radius
    
    # Local Search Parameters
    step_size = 0.01
    kick_interval = 300
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Check time more frequently to avoid overshooting
        if (iteration % 10 == 0) and (time.time() - start_time > time_limit - 5.0):
            break
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck for a while, kick with larger perturbation
        if no_improve_count > kick_interval:
            # Gradually increase kick size based on how long we've been stuck
            kick_size = 0.03 + 0.03 * min(no_improve_count / 3000.0, 1.0)
            centers += np.random.normal(0, kick_size, size=(N, 3))
            # Clip to cube to ensure validity
            centers = np.clip(centers, -0.49, 0.49)
            no_improve_count = 0
            current_radius, _, _ = _compute_score_fast(centers)
            continue
            
        # Find active points
        active_indices = _get_active_points(centers, current_radius)
        
        if len(active_indices) == 0:
            # Should not happen if radius is correct, but handle just in case
            active_indices = np.random.choice(N, size=min(5, N), replace=False)
            
        # Try to move one active point
        idx = active_indices[np.random.randint(len(active_indices))]
        
        # Generate random direction
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        # Try multiple step sizes for better exploration
        step_sizes = [step_size, step_size * 0.5, step_size * 2.0]
        improved = False
        
        for ss in step_sizes:
            for sign in [1, -1]:
                new_centers = centers.copy()
                new_centers[idx] += sign * ss * direction
                
                # Ensure points stay inside the cube
                new_centers = np.clip(new_centers, -0.49, 0.49)
                
                new_radius, _, _ = _compute_score_fast(new_centers)
                
                if new_radius > current_radius:
                    centers = new_centers
                    current_radius = new_radius
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a small random kick on the point
            small_kick = 0.003
            for _ in range(3):
                new_centers = centers.copy()
                new_centers[idx] += np.random.normal(0, small_kick, size=3)
                new_centers = np.clip(new_centers, -0.49, 0.49)
                new_radius, _, _ = _compute_score_fast(new_centers)
                if new_radius > current_radius:
                    centers = new_centers
                    current_radius = new_radius
                    improved = True
                    break
        
        iteration += 1
        
        # Adaptive step size
        if improved:
            # Keep or slightly increase step size
            step_size = min(step_size * 1.03, 0.03)
        else:
            # Reduce step size
            step_size *= 0.97
            if step_size < 0.003:
                step_size = 0.008  # Reset if too small

    # Final check
    final_radius, _, _ = _compute_score_fast(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return best_centers

# EVOLVE-BLOCK-END
