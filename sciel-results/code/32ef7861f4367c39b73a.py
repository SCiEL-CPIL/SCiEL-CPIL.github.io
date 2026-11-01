import time
import numpy as np
from scipy.optimize import minimize

N = 50
# 150s limit, keep 10s safety margin
_DEADLINE = time.time() + 140.0 

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check if inside unit circle
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
        
    # Calculate min distance to boundary
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    # Calculate min pairwise distance
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    # x_vector contains boundary_dists and pair_dists
    # Note: The score is min(boundary_dists, pair_dists/2).
    # So the "critical distances" for the min are:
    # x_k = boundary_dists[i]
    # x_k = pair_dists[j] / 2.0
    # We want to maximize min(x_k).
    
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation
    # y = -beta * all_dists
    y = -beta * all_dists
    max_y = np.max(y)
    # Prevent overflow
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    # d/dx_j [ 1/beta ln(sum_k exp(-beta x_k)) ]
    # = - sum_k [ exp(-beta x_k) / sum(exp(-beta x_k)) ] * d/dx_j x_k
    # Let w_k = exp(-beta x_k) / sum(exp(-beta x_k)).
    # Gradient w.r.t x_k is -w_k.
    
    weights = np.exp(y_shifted) / sum_exp  # These are w_k
    
    # Now we need d/d(pts) x_k for each k.
    # x_k are either boundary_dist[i] or pair_dist[j]/2.
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    # x_b_i = 1 - |p_i|
    # d/dp_i x_b_i = - p_i / |p_i|
    # Contribution: - w_b_i * (- p_i / |p_i|) = w_b_i * p_i / |p_i|
    # Note: weights for boundary terms are weights[0:N]
    w_boundary = weights[0:N]
    
    # Avoid division by zero for points at origin (probable measure 0, but safe)
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    # x_p_j = 0.5 * |p_i - p_m| for pair (i, m)
    # d/dp_i x_p_j = 0.5 * (p_i - p_m) / |p_i - p_m|
    # d/dp_m x_p_j = 0.5 * (p_m - p_i) / |p_i - p_m| = - d/dp_i x_p_j
    # Contribution to grad_pts[i] from pair j: - w_p_j * (0.5 * (p_i - p_m) / d_ij)
    # Contribution to grad_pts[m] from pair j: - w_p_j * (0.5 * (p_m - p_i) / d_ij)
    
    w_pairs = weights[N:]
    
    # We have iu indices
    i_idx, m_idx = iu
    # dist_mat[i_idx, m_idx] are the distances
    d_pairs = dist_mat[i_idx, m_idx]
    
    # Safe division
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    
    # Vector (p_i - p_m)
    vecs = pts[i_idx] - pts[m_idx]
    
    # Unit vectors
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    # Gradient for point i: - w_j * 0.5 * unit_vec
    # Gradient for point m: - w_j * 0.5 * (-unit_vec) = w_j * 0.5 * unit_vec
    
    # Accumulate gradients
    # We need to add to grad_pts[i] and grad_pts[m]
    # Use np.add.at for repeated indices
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    # Flatten gradient
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _ring_init(ring_counts):
    """
    Generate a ring-based initial configuration.
    ring_counts: list of ints, e.g., [1, 6, 12, 14] means 1 center, 6 in ring 1, etc.
    We place points uniformly on each ring, with radii chosen to be roughly optimal.
    """
    pts = []
    
    # 1. Center point
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
        
    num_rings = len(ring_counts) - 1
    if num_rings == 0:
        # If only center, just return it (but N=50, so this won't happen)
        pass
        
    # Estimate radii for rings.
    # For a hexagonal packing, ring radii are roughly proportional to sqrt(k) or k.
    # Let's try to fit them within the unit circle.
    # A good heuristic for ring radii in circle packing is r_k = k * r_1.
    # We need to scale so the outermost ring fits.
    
    if num_rings > 0:
        # Determine scale factor
        # Assume ideal spacing d.
        # Radius of ring k is k*d.
        # Boundary constraint: r_outer + r_circle <= 1.
        # Pairwise constraint: d >= 2*r_circle.
        
        # Let's just pick radii that are evenly spaced from a small value to ~0.8
        # and let the optimizer fix them.
        # Better: Use the known structure.
        # For N=50, typical rings are 1, 6, 12, 14, ... or 1, 6, 12, 15, ...
        # Let's assume linear growth in radius for initialization.
        
        # Max radius should be < 1.
        # Let's map ring indices 1..num_rings to radii [0.2, 0.8]
        radii = np.linspace(0.2, 0.8, num_rings) if num_rings > 1 else [0.5]
        
        for k in range(num_rings):
            count = ring_counts[k+1]
            if count <= 0:
                continue
            r = radii[k]
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            # Add a small random jitter to break symmetry and help optimization
            angles += np.random.uniform(-0.1, 0.1, count)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.extend(np.column_stack([x, y]))
            
    pts = np.array(pts)
    
    # If we have more points than N, take the first N (center + inner rings)
    # If we have fewer, pad with random points? 
    # The ring_counts provided should sum to N.
    if len(pts) != N:
        # Fallback: if counts don't sum to N, adjust
        # This shouldn't happen if we construct valid counts
        pass
        
    return pts[:N]

def _generate_ring_configs():
    """
    Generate several plausible ring configurations for N=50.
    """
    configs = []
    
    # Common structures for 50 circles
    # 1 center + 6 + 12 + 14 + ...?
    # 1 + 6 + 12 + 15 + 16 = 50
    configs.append([1, 6, 12, 15, 16])
    
    # 1 + 6 + 12 + 12 + 19 = 50
    configs.append([1, 6, 12, 12, 19])
    
    # 1 + 6 + 12 + 10 + 21 = 50
    configs.append([1, 6, 12, 10, 21])
    
    # 1 + 6 + 10 + 14 + 19 = 50
    configs.append([1, 6, 10, 14, 19])
    
    # 1 + 6 + 12 + 18 + 13 = 50
    configs.append([1, 6, 12, 18, 13])
    
    # 1 + 6 + 14 + 14 + 15 = 50
    configs.append([1, 6, 14, 14, 15])
    
    # 1 + 6 + 12 + 16 + 15 = 50
    configs.append([1, 6, 12, 16, 15])
    
    return configs

def _optimize_with_lbfgsb(pts0, beta_schedule, max_time=10.0):
    """
    Run L-BFGS-B optimization with a schedule of beta values.
    """
    start_time = time.time()
    current_pts = pts0.copy()
    
    for beta in beta_schedule:
        if time.time() - start_time > max_time:
            break
            
        res = minimize(
            _soft_min_obj_and_grad,
            current_pts.flatten(),
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        current_pts = res.x.reshape(N, 2)
        
        # Project to unit disk if necessary (L-BFGS-B doesn't enforce constraints directly unless specified, 
        # but our objective penalizes going out. However, numerical errors might push slightly out.
        # We should add a hard projection or use bounds? 
        # The objective function returns a value that goes to infinity if pts go far out? 
        # No, soft-min of negative distances.
        # Let's project manually after each step to ensure validity.
        radii = np.linalg.norm(current_pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / radii[mask, np.newaxis]
            
    return current_pts

def construct():
    start_time = time.time()
    
    # 1. Generate initial configurations
    configs = _generate_ring_configs()
    inits = []
    for cfg in configs:
        # Add some randomization to angles/radii to avoid symmetric traps
        # We'll just generate the base ring config and rely on L-BFGS to move them
        pts = _ring_init(cfg)
        # Add small random noise
        pts = pts + np.random.normal(0, 0.01, (N, 2))
        # Project to disk
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis]
        inits.append(pts)
        
    # Also add a random initialization as a fallback
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    random_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    inits.append(random_pts)
    
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # 2. Optimization Loop
    # We want to use the time budget effectively.
    # Strategy: Run a few initializations with a moderate beta schedule, then polish the best one.
    
    # Beta schedule: start low (smooth), end high (sharp)
    beta_schedule = [10, 50, 200, 500]
    
    time_per_run = 10.0 # seconds per initialization optimization
    
    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 10.0: # Leave 10s for final polish
            break
            
        # Run optimization
        try:
            opt_pts = _optimize_with_lbfgsb(init_pts, beta_schedule, max_time=time_per_run)
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
            
    # 3. Final Polish
    # Use the remaining time to polish the best solution with very high beta
    if time.time() < _DEADLINE - 2.0:
        remaining_time = _DEADLINE - time.time() - 2.0
        # Use high beta for precision
        polish_beta = [1000, 2000, 5000]
        try:
            polished_pts = _optimize_with_lbfgsb(best_pts, polish_beta, max_time=remaining_time)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
            
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        # Return the first init if all failed
        return inits[0]
        
    return best_pts

# EVOLVE-BLOCK-END
