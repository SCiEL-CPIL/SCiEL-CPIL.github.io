import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

N = 100

def _compute_score(centers):
    """Compute the hard score: max radius."""
    # Centers must be in [-0.5, 0.5]^3
    # Distance to nearest wall
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use efficient computation
    d = pdist(centers)
    min_pair = np.min(d)
    
    r = min(min_pair / 2.0, min_wall)
    return r

def _find_bottlenecks(centers):
    """
    Identify the constraints that define the current radius.
    Returns:
    - bottleneck_pairs: list of (i, j) indices for pairs with dist <= 2*r + eps
    - bottleneck_walls: list of (i, axis) for walls with dist <= r + eps
    - current_r: the current radius
    """
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    d = pdist(centers)
    min_pair = np.min(d)
    
    r = min(min_pair / 2.0, min_wall)
    
    eps = 1e-8
    bottleneck_pairs = []
    bottleneck_walls = []
    
    # Find pairs
    # We need to map pdist indices to (i, j)
    # For N=100, this is fast enough to do in a loop or vectorized trick
    # Vectorized approach for index mapping:
    # We can just loop over the close ones since usually few are close.
    # But to be safe and fast, let's just get all pairs < 2*r + eps.
    # Actually, pdist doesn't give indices directly.
    # Let's compute full distance matrix for index finding, or use a trick.
    # N=100, full matrix is 100x100. Very fast.
    dist_mat = squareform(d)
    
    # Pairs where dist <= 2*r + eps
    # Diagonal is 0, ignore
    np.fill_diagonal(dist_mat, np.inf)
    
    pair_mask = dist_mat <= (2.0 * r + eps)
    # Get indices
    i_idx, j_idx = np.where(pair_mask)
    # Ensure i < j for consistency if needed, but for repulsion direction it doesn't matter
    # We just need the pairs
    bottleneck_pairs = list(zip(i_idx, j_idx))
    
    # Walls
    # wall_dists is (N, 3)
    wall_mask = wall_dists <= (r + eps)
    # Get indices
    w_i_idx, w_axis_idx = np.where(wall_mask)
    bottleneck_walls = list(zip(w_i_idx, w_axis_idx))
    
    return bottleneck_pairs, bottleneck_walls, r

def _repulsion_step(centers, step_size=0.001):
    """
    Apply repulsion forces based on hard bottlenecks.
    This is a subgradient ascent on the hard min function.
    """
    pairs, walls, r = _find_bottlenecks(centers)
    
    new_centers = centers.copy()
    
    # 1. Pair repulsion
    if pairs:
        for i, j in pairs:
            # Vector from i to j
            vec = centers[j] - centers[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-9:
                # Random nudge
                vec = np.random.normal(size=3)
                dist = np.linalg.norm(vec)
            unit_vec = vec / dist
            
            # Push i away from j, j away from i
            # We want to increase the distance between them
            # Gradient of distance wrt i is -unit_vec, wrt j is unit_vec
            # We want to ascend distance, so move i in direction of -unit_vec?
            # Wait. d(|c_j - c_i|)/d c_i = (c_i - c_j)/|c_i - c_j| = -unit_vec.
            # So to increase distance, we should move i in direction of -unit_vec?
            # Let's check: c_i_new = c_i - step * unit_vec.
            # c_j - c_i_new = c_j - c_i + step * unit_vec.
            # Norm increases. Correct.
            
            new_centers[i] -= step_size * unit_vec
            new_centers[j] += step_size * unit_vec
    
    # 2. Wall repulsion
    if walls:
        for i, axis in walls:
            # Wall constraint: 0.5 - |c_ik| >= r
            # We want to increase 0.5 - |c_ik|, which means decrease |c_ik|.
            # So move center towards 0.
            # Gradient of wall_dist wrt c_ik is -sign(c_ik).
            # To ascend wall_dist, move in direction of -sign(c_ik).
            # i.e., towards 0.
            
            # Add a small random component if stuck at 0?
            # sign(0) is 0. If center is at 0, gradient is 0.
            # But if it's a bottleneck, it's at r distance from wall.
            # So c_ik is either r or -r.
            # If c_ik > 0, move left. If c_ik < 0, move right.
            
            val = centers[i, axis]
            if val > 1e-9:
                new_centers[i, axis] -= step_size
            elif val < -1e-9:
                new_centers[i, axis] += step_size
            else:
                # At origin? Unlikely to be a bottleneck unless r=0.
                # Nudge randomly
                new_centers[i, axis] += np.random.choice([-1, 1]) * step_size * 0.5

    # Project back to cube to ensure validity
    new_centers = np.clip(new_centers, -0.5, 0.5)
    
    return new_centers

def _jitter_and_repair(centers, rng, jitter_scale=0.05):
    """
    Apply random jitter to break symmetries and escape local minima.
    Then repair any overlaps/wall violations by pushing spheres apart/in.
    """
    # Jitter
    jittered = centers + rng.normal(0, jitter_scale, size=centers.shape)
    jittered = np.clip(jittered, -0.5, 0.5)
    
    # Repair: Simple iterative separation
    # This is a basic heuristic repair
    for _ in range(50):
        # Check for overlaps
        d = pdist(jittered)
        min_d = np.min(d)
        if min_d > 2 * 0.11: # Assume we want to maintain at least some radius
            break
        
        # Find closest pair
        dist_mat = squareform(d)
        np.fill_diagonal(dist_mat, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        i, j = min_idx
        
        vec = jittered[j] - jittered[i]
        dist = np.linalg.norm(vec)
        if dist < 1e-9:
            vec = rng.normal(size=3)
            dist = np.linalg.norm(vec)
        
        unit_vec = vec / dist
        # Push apart
        jittered[i] -= unit_vec * 0.01
        jittered[j] += unit_vec * 0.01
        
        # Check walls
        wall_dists = 0.5 - np.abs(jittered)
        min_wall = np.min(wall_dists)
        if min_wall < 0.05:
            # Push in
            w_idx, w_axis = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            if jittered[w_idx, w_axis] > 0:
                jittered[w_idx, w_axis] -= 0.01
            else:
                jittered[w_idx, w_axis] += 0.01
        
        jittered = np.clip(jittered, -0.5, 0.5)
        
    return jittered

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    rng = np.random.default_rng(42)
    
    # 1. Initialization
    # Start with FCC lattice if possible, or dense grid
    # For N=100, a 4x5x5 grid is 100 points.
    # 4x5x5 = 100.
    # Spacing: 1/4=0.25, 1/5=0.2, 1/5=0.2.
    # Min dist is 0.2. Radius 0.1.
    # Better: Use a perturbed FCC or just a good grid.
    # Let's use a grid of 5x5x4 = 100.
    m_x, m_y, m_z = 5, 5, 4
    ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
    ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
    ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    best_centers = grid[:N].copy()
    best_score = _compute_score(best_centers)
    
    # 2. Main Loop: Alternating Repulsion and Jitter
    # We will run a loop that:
    # a) Applies repulsion steps to improve local packing
    # b) If stuck, applies jitter to escape
    # c) Evaluates score
    
    current_centers = best_centers.copy()
    current_score = best_score
    
    # Adaptive step size
    base_step = 0.005
    
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() < deadline - 5:
        # a) Repulsion Phase
        # Perform several small steps
        for step in range(100):
            if time.time() > deadline - 5:
                break
            
            # Apply repulsion
            new_centers = _repulsion_step(current_centers, step_size=base_step)
            
            # Check if score improved
            new_score = _compute_score(new_centers)
            
            if new_score > current_score:
                current_centers = new_centers
                current_score = new_score
                stuck_counter = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
            else:
                # If no improvement, maybe reduce step size or try jitter
                # But for now, just accept the move if it's not worse?
                # Actually, for hard constraints, any move that maintains validity is good.
                # But we want to maximize.
                # If score didn't improve, we might be oscillating.
                # Let's accept the move anyway to explore, but track "stuck" if score drops significantly?
                # No, _repulsion_step is a gradient ascent on the min distance.
                # It should generally improve or maintain.
                # If it doesn't improve, we might be at a local max or plateau.
                current_centers = new_centers
                stuck_counter += 1
        
        # b) Jitter Phase
        if stuck_counter > max_stuck:
            # Try to escape
            jittered = _jitter_and_repair(current_centers, rng, jitter_scale=0.02)
            j_score = _compute_score(jittered)
            
            if j_score > current_score * 0.9: # Allow some degradation to escape deep local minima
                current_centers = jittered
                current_score = j_score
                stuck_counter = 0
            else:
                # If jitter failed badly, try smaller jitter
                jittered = _jitter_and_repair(current_centers, rng, jitter_scale=0.005)
                j_score = _compute_score(jittered)
                if j_score > current_score:
                    current_centers = jittered
                    current_score = j_score
                    stuck_counter = 0
                else:
                    # Give up on this local spot?
                    # Reset to best known
                    current_centers = best_centers.copy()
                    current_score = best_score
                    stuck_counter = 0
                    
    # 3. Final Polish with BFGS on Soft-Min (using the best centers found)
    # This helps to squeeze out the last bit of radius by smoothing the landscape
    if time.time() < deadline - 10:
        try:
            def _soft_min_score(c, alpha=100.0):
                d_pairs = pdist(c) / 2.0
                wall_dists = 0.5 - np.abs(c)
                w_flat = wall_dists.flatten()
                constraints = np.concatenate([d_pairs, w_flat])
                min_c = np.min(constraints)
                shifted = -alpha * (constraints - min_c)
                log_sum = np.log(np.sum(np.exp(shifted)))
                return min_c - log_sum / alpha

            def _grad_soft_min_score(c, alpha=100.0):
                d_pairs = pdist(c) / 2.0
                wall_dists = 0.5 - np.abs(c)
                w_flat = wall_dists.flatten()
                constraints = np.concatenate([d_pairs, w_flat])
                min_c = np.min(constraints)
                shifted = -alpha * (constraints - min_c)
                exp_shifted = np.exp(shifted)
                sum_exp = np.sum(exp_shifted)
                weights = exp_shifted / sum_exp
                
                grad = np.zeros_like(c)
                
                # Pair gradients
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                dists_safe = np.where(dists < 1e-8, 1e-8, dists)
                unit_vecs = diff / dists_safe[:, :, np.newaxis]
                
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                i_idx, j_idx = np.where(mask)
                pair_weights = weights[:len(i_idx)]
                u_vecs = unit_vecs[i_idx, j_idx]
                
                np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * u_vecs)
                np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * u_vecs)
                
                # Wall gradients
                wall_weights = weights[len(i_idx):]
                signs = np.sign(c)
                ww = wall_weights.reshape(N, 3)
                grad += ww * signs
                
                return grad

            def neg_score(x):
                c = x.reshape(N, 3)
                return -_soft_min_score(c, alpha=50.0)
                
            def neg_grad(x):
                c = x.reshape(N, 3)
                return -_grad_soft_min_score(c, alpha=50.0)

            res = minimize(
                neg_score, 
                best_centers.flatten(), 
                jac=neg_grad, 
                method='BFGS', 
                options={'maxiter': 100, 'gtol': 1e-8, 'disp': False}
            )
            
            opt_centers = res.x.reshape(N, 3)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            hard_score = _compute_score(opt_centers)
            
            if hard_score > best_score:
                best_score = hard_score
                best_centers = opt_centers.copy()
        except Exception:
            pass

    # Safety check
    if not np.all(np.isfinite(best_centers)):
        best_centers = grid[:N].copy()
        
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_centers
