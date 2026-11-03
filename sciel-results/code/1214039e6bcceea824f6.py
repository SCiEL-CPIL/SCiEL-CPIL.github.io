# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_init(n_points, seed=0):
    """Generate hexagonal lattice points, select n_points closest to origin."""
    # Estimate spacing for hex packing in unit circle
    # Area of circle = pi. Area of N circles = N * pi * r^2.
    # Hex density ~ 0.9069.
    # N * (2r)^2 * (sqrt(3)/2) * 0.9069 ~ pi (rough area match)
    # Actually, just use r ~ 0.053.
    # Spacing d = 2r.
    
    best_pts = None
    best_r = 0.0
    
    # Try a few different spacings to ensure we get enough points and good coverage
    for attempt in range(5):
        # Adjust spacing slightly
        base_spacing = 2.0 / (np.sqrt(n_points) * 1.1)
        spacing = base_spacing * (0.95 + 0.05 * attempt)
        
        # Generate points in a grid that covers the unit circle
        # Max radius is 1. So we need points up to radius 1.
        # k is the number of steps in x and y.
        # x range: -1 to 1. Steps of 'spacing'.
        # k ~ 1/spacing
        k = int(1.0 / spacing) + 5
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1.0 - spacing*0.5)**2:
                    points.append([x, y])
        
        if len(points) < n_points:
            continue
            
        points = np.array(points)
        
        # Sort by distance from center to get the innermost N points
        # This creates a "disk" of points
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n_points]
        pts = points[idx]
        
        # Calculate the radius of this configuration
        # Min pairwise distance
        # To speed up check, we can just take min of nearest neighbors in lattice?
        # No, just compute properly for small N or use a fast check.
        # For init, we just want a valid starting point.
        # We'll compute the actual radius to pick the best init.
        
        # Fast radius calculation for init comparison
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # This is O(N^2). 300^2 = 90k. Fast enough.
        
        # Only do full check if it looks promising or for the first few attempts
        if attempt < 2 or best_r < 0.05:
             d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
             np.fill_diagonal(d, np.inf)
             min_pair = np.min(d)
             min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
             r = min(min_pair / 2.0, min_bound)
             
             if r > best_r:
                 best_r = r
                 best_pts = pts

    if best_pts is None:
        # Fallback: Random points
        best_pts = np.random.rand(n_points, 2) * 1.8 - 0.9
        # Scale to fit
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts / norms[:, np.newaxis] * 0.9

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for a given set of points."""
    if len(pts) == 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
    return min(min_pair, min_bound)

def _soft_min_logsumexp(values, lam):
    """
    Compute -1/lam * log(sum(exp(-lam * values)))
    This approximates min(values) for large lam.
    """
    # Shift for numerical stability
    min_val = np.min(values)
    shifted = values - min_val
    # exp(-lam * shifted)
    exp_terms = np.exp(-lam * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - log_sum / lam

def _objective_and_grad(x, lam):
    """
    Objective: Minimize -soft_min(distances)
    This is equivalent to Maximize soft_min(distances)
    Distances: all pairwise distances and (1 - |c_i|)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # 1. Compute pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # We only need the upper triangle for the sum, but for gradient we need all contributions
    # Let's use the full matrix for simplicity in gradient derivation, 
    # but divide the sum by 2 to avoid double counting in the objective?
    # No, log-sum-exp is not linear. 
    # Sum over i < j of exp(-lam * d_ij).
    # If we sum over all i != j, we are summing 2 * exp(-lam * d_ij).
    # log(2 * S) = log(2) + log(S).
    # The constant log(2) disappears in the gradient?
    # d/dx log(2S) = (1/(2S)) * 2 * dS/dx = (1/S) dS/dx.
    # So the gradient is the same if we use full matrix or upper triangle, 
    # provided we are consistent.
    # However, the value of the objective will differ by a constant factor inside the log.
    # -1/lam * log(2 * S) = -1/lam * (log(2) + log(S)) = -1/lam log(S) - log(2)/lam.
    # The constant offset doesn't affect the minimizer.
    # So we can use the full matrix for easier vectorization.
    
    # 2. Compute boundary distances
    radii = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - radii
    
    # 3. Combine all distances into a single vector for log-sum-exp
    # Flatten upper triangle of dist
    iu = np.triu_indices(n, 1)
    d_pairs = dist[iu]
    
    # d_pairs shape: (N*(N-1)/2,)
    # bound_dist shape: (N,)
    # Combined shape: (N*(N-1)/2 + N,)
    all_dists = np.concatenate([d_pairs, bound_dist])
    
    # 4. Objective Value
    # We want to MAXIMIZE min(all_dists).
    # Minimize -soft_min(all_dists).
    m_soft = _soft_min_logsumexp(all_dists, lam)
    obj = -m_soft
    
    # 5. Gradient
    # d/dx (-soft_min) = d/dx (1/lam * log(sum(exp(-lam * d))))
    # = 1/lam * 1/S * sum ( -lam * exp(-lam * d_k) * d d_k/dx )
    # = -1/S * sum ( exp(-lam * d_k) * d d_k/dx )
    # Let P_k = exp(-lam * d_k) / S. Sum P_k = 1.
    # Grad = - sum P_k * d d_k/dx
    
    # Calculate weights P_k
    exp_vals = np.exp(-lam * (all_dists - np.min(all_dists))) # Stable exp
    S = np.sum(exp_vals)
    P = exp_vals / S
    
    # Split P back into pair weights and bound weights
    n_pairs = len(d_pairs)
    P_pairs = P[:n_pairs]
    P_bonds = P[n_pairs:]
    
    # Gradient accumulation
    grad = np.zeros_like(pts)
    
    # A. Boundary Gradient
    # d (1 - |c_i|) / d c_i = - c_i / |c_i|
    # Contribution to grad[i]: - P_bonds[i] * (- c_i / |c_i|) = P_bonds[i] * c_i / |c_i|
    # Wait, objective is -soft_min.
    # d(-soft_min)/dx = - ( - sum P_k d d_k/dx ) = sum P_k d d_k/dx.
    # So Grad = sum P_k * d d_k/dx.
    
    # For boundary term k=i:
    # d d_i/d c_i = - c_i / |c_i|
    # Grad_bound[i] = P_bonds[i] * (- c_i / |c_i|)
    
    radii_safe = np.maximum(radii, 1e-8)
    unit_vecs = pts / radii_safe[:, np.newaxis]
    grad_bound = -P_bonds[:, np.newaxis] * unit_vecs
    
    # B. Pairwise Gradient
    # For pair (i, j) with weight P_pair:
    # d d_ij / d c_i = (c_i - c_j) / d_ij
    # d d_ij / d c_j = (c_j - c_i) / d_ij = - (c_i - c_j) / d_ij
    
    # We need to sum P_pair * (c_i - c_j) / d_ij for all j != i.
    
    # Create a matrix of weights W[i, j] = P_pair(i, j) / d_ij
    # Note: P_pairs is a flat array corresponding to upper triangle.
    # We need to map it back to a matrix.
    
    W = np.zeros((n, n))
    # Fill W with P_pairs / d_pairs
    # iu gives indices for upper triangle
    # We need to assign W[i, j] = P_pairs[k] / d_pairs[k]
    # And W[j, i] = P_pairs[k] / d_pairs[k] (since distance is symmetric and gradient is antisymmetric in direction but we handle signs in sum)
    # Actually, let's just use the full distance matrix for the division.
    
    # Map flat indices to matrix indices
    # This is a bit slow if done element by element. 
    # Vectorized approach:
    # We have iu[0], iu[1] which are arrays of indices.
    # W[iu[0], iu[1]] = P_pairs / d_pairs
    # W[iu[1], iu[0]] = P_pairs / d_pairs
    
    W[iu[0], iu[1]] = P_pairs / d_pairs
    W[iu[1], iu[0]] = P_pairs / d_pairs
    
    # Now Grad_pair[i] = sum_j W[i, j] * (c_i - c_j)
    # = sum_j W[i, j] * c_i - sum_j W[i, j] * c_j
    # = c_i * sum_j W[i, j] - sum_j (W[i, j] * c_j)
    
    row_sums = np.sum(W, axis=1)
    W_pts = W @ pts
    
    grad_pair = pts * row_sums[:, np.newaxis] - W_pts
    
    total_grad = grad_bound + grad_pair
    
    return obj, total_grad.ravel()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization
    points = _hex_init(N, seed=42)
    x0 = points.ravel()
    
    # 2. Optimization
    # Start with a smaller lambda (smoother) and increase? 
    # Or just use a fixed reasonable lambda.
    # N=300, min dist ~ 0.1.
    # lambda * d ~ 1-5 is good range.
    # lambda ~ 10-50.
    
    lam = 20.0
    
    # Bounds: keep points inside [-1, 1] to help optimizer
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Use L-BFGS-B
    # maxiter: limit to ensure time safety
    # We can run multiple stages with increasing lambda
    
    best_x = x0.copy()
    best_r = _compute_radius(points)
    
    # Stage 1: Coarse optimization
    lam1 = 5.0
    result = minimize(
        _objective_and_grad, 
        x0, 
        args=(lam1,),
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() > deadline - 30:
         # If running out of time, return best so far
         return best_x.reshape(-1, 2)
        
    if result.fun < 0: # It's minimizing -soft_min, so lower is better (more negative means larger min)
         # Wait, obj = -m_soft. m_soft is positive. So obj is negative.
         # We want to minimize obj.
         pass
         
    x1 = result.x
    r1 = _compute_radius(x1.reshape(-1, 2))
    if r1 > best_r:
        best_r = r1
        best_x = x1.copy()
        
    # Stage 2: Fine optimization with higher lambda
    lam2 = 20.0
    result2 = minimize(
        _objective_and_grad, 
        x1, 
        args=(lam2,),
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    if time.time() > deadline - 10:
         return best_x.reshape(-1, 2)
         
    x2 = result2.x
    r2 = _compute_radius(x2.reshape(-1, 2))
    if r2 > best_r:
        best_r = r2
        best_x = x2.copy()
        
    # Stage 3: Local Refinement
    # Identify points that are "critical" (close to min distance or boundary)
    pts = best_x.reshape(-1, 2)
    
    # Find min pairwise distance and min boundary distance
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    radii = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - np.max(radii)
    
    r_current = min(min_pair / 2.0, min_bound)
    
    # Find indices of points involved in the minimum distance
    # Or just take points close to the boundary or each other
    # Let's take points where boundary dist < r_current + 0.01
    # Or points involved in pairs with dist < r_current * 2 + 0.01
    
    # Simpler: Just run a few more iterations on the whole set if time permits
    # Or a very high lambda step
    
    if time.time() < deadline - 5:
         lam3 = 50.0
         result3 = minimize(
             _objective_and_grad, 
             best_x, 
             args=(lam3,),
             method='L-BFGS-B', 
             jac=True, 
             bounds=bounds,
             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
         )
         x3 = result3.x
         r3 = _compute_radius(x3.reshape(-1, 2))
         if r3 > best_r:
             best_r = r3
             best_x = x3.copy()
             
    # Final validity check and clipping
    final_pts = best_x.reshape(-1, 2)
    radii = np.linalg.norm(final_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        final_pts[mask] = final_pts[mask] / radii[mask, np.newaxis] * 0.999999
        
    return final_pts
# EVOLVE-BLOCK-END
