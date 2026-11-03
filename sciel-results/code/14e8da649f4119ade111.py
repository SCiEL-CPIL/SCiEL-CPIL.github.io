# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _compute_radius_fast(pts):
    """Compute packing radius using chunked distance to save memory."""
    n = len(pts)
    if n == 0:
        return 0.0
    
    # Compute pairwise distances in chunks to avoid O(n^2) memory
    min_dist_sq = np.inf
    chunk_size = 50
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        diff = pts[i:j, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Zero out diagonal for this chunk
        for k in range(j - i):
            dist_sq[k, i + k] = np.inf
        min_dist_sq = min(min_dist_sq, np.min(dist_sq))
    
    min_pair = np.sqrt(min_dist_sq) / 2.0
    radii = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - np.max(radii)
    
    return min(min_pair, min_bound)

def _compute_radius(pts):
    """Compute the packing radius for a given set of points."""
    return _compute_radius_fast(pts)

def _hex_init(n_points, seed=0):
    """
    Generate multiple diverse hexagonal lattice initializations 
    and return the best one based on initial packing radius.
    """
    best_pts = None
    best_r = -1.0
    
    # Try different rotations, spacings, and offsets
    for attempt in range(20):
        # Base spacing estimation
        base_spacing = 2.0 / (np.sqrt(n_points) * 1.1)
        
        # Vary spacing
        spacing = base_spacing * (0.90 + 0.015 * attempt)
        
        # Generate points in a grid that covers the unit circle
        k = int(1.0 / spacing) + 8
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                # Standard hex lattice with offset
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                
                # Apply rotation and translation
                angle = attempt * np.pi / 7.0
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                
                # Small random offset for diversity
                offset = 0.01 * attempt
                x_rot += offset * np.cos(attempt * 2.0)
                y_rot += offset * np.sin(attempt * 3.0)
                
                if x_rot**2 + y_rot**2 <= (1.0 - spacing*0.5)**2:
                    points.append([x_rot, y_rot])
        
        if len(points) < n_points:
            continue
            
        points = np.array(points)
        
        # Sort by distance from center to get the innermost N points
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n_points]
        pts = points[idx]
        
        # Calculate radius
        r = _compute_radius(pts)
        
        if r > best_r:
            best_r = r
            best_pts = pts

    if best_pts is None:
        # Fallback: Random points scaled to fit
        np.random.seed(42)
        best_pts = np.random.rand(n_points, 2) * 1.8 - 0.9
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts / norms[:, np.newaxis] * 0.9
        best_r = _compute_radius(best_pts)

    return best_pts

def _objective_and_grad(x, lam):
    """
    Objective: Minimize -soft_min(distances)
    This is equivalent to Maximize soft_min(distances)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # 1. Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # 2. Compute boundary distances
    radii = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - radii
    
    # 3. Combine all distances into a single vector for log-sum-exp
    iu = np.triu_indices(n, 1)
    d_pairs = dist[iu]
    all_dists = np.concatenate([d_pairs, bound_dist])
    
    # 4. Objective Value
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_terms = np.exp(-lam * shifted)
    log_sum = np.log(np.sum(exp_terms))
    m_soft = min_val - log_sum / lam
    obj = -m_soft
    
    # 5. Gradient
    S = np.sum(exp_terms)
    P = exp_terms / S
    
    n_pairs = len(d_pairs)
    P_pairs = P[:n_pairs]
    P_bonds = P[n_pairs:]
    
    # Gradient accumulation
    grad = np.zeros_like(pts)
    
    # A. Boundary Gradient
    radii_safe = np.maximum(radii, 1e-8)
    unit_vecs = pts / radii_safe[:, np.newaxis]
    grad_bound = -P_bonds[:, np.newaxis] * unit_vecs
    
    # B. Pairwise Gradient
    W = np.zeros((n, n))
    W[iu[0], iu[1]] = P_pairs / d_pairs
    W[iu[1], iu[0]] = P_pairs / d_pairs
    
    row_sums = np.sum(W, axis=1)
    W_pts = W @ pts
    grad_pair = pts * row_sums[:, np.newaxis] - W_pts
    
    total_grad = grad_bound + grad_pair
    
    return obj, total_grad.ravel()

def _local_refinement(pts, max_iter=50, step_size=0.01):
    """Simple local refinement to improve packing."""
    current = pts.copy()
    current_r = _compute_radius(current)
    
    for it in range(max_iter):
        improved = False
        # Try small random perturbations
        for _ in range(10):
            trial = current.copy()
            mask = np.random.rand(N) < 0.1  # Perturb 10% of points
            trial[mask] += np.random.randn(np.sum(mask), 2) * step_size
            # Keep inside unit circle
            radii = np.linalg.norm(trial, axis=1)
            mask_out = radii > 1.0
            trial[mask_out] = trial[mask_out] / radii[mask_out, np.newaxis] * 0.9999
            
            trial_r = _compute_radius(trial)
            if trial_r > current_r + 1e-12:
                current = trial
                current_r = trial_r
                improved = True
        
        if not improved:
            break
    
    return current, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit is 150s)
    
    # 1. Initialization
    points = _hex_init(N)
    x0 = points.ravel()
    best_x = x0.copy()
    best_r = _compute_radius(points)
    
    # 2. Optimization with Annealing
    # Improved lambda schedule: more stages, smoother increase
    lambdas = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    current_x = x0.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time()
        if time_left < 15:
            break
        
        # Smaller lambda needs more iterations
        if lam < 2.0:
            maxiter = 800
        elif lam < 10.0:
            maxiter = 500
        elif lam < 50.0:
            maxiter = 300
        else:
            maxiter = 200
            
        try:
            result = minimize(
                _objective_and_grad, 
                current_x, 
                args=(lam,),
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 150}
            )
            
            if result.x is not None:
                new_r = _compute_radius(result.x.reshape(-1, 2))
                if new_r > best_r:
                    best_r = new_r
                    best_x = result.x.copy()
                current_x = result.x.copy()
                
        except Exception as e:
            break
            
    # 3. Local refinement pass
    if time.time() < deadline - 10:
        pts = best_x.reshape(-1, 2)
        refined_pts, refined_r = _local_refinement(pts, max_iter=30, step_size=0.005)
        if refined_r > best_r:
            best_r = refined_r
            best_x = refined_pts.ravel()
    
    # 4. Final high-precision polish
    if time.time() < deadline - 3:
        lam_final = 1000.0
        try:
            result_final = minimize(
                _objective_and_grad, 
                best_x, 
                args=(lam_final,),
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200}
            )
            
            if result_final.x is not None:
                final_r = _compute_radius(result_final.x.reshape(-1, 2))
                if final_r > best_r:
                    best_r = final_r
                    best_x = result_final.x.copy()
        except Exception:
            pass
    
    # Final validity check and clipping
    final_pts = best_x.reshape(-1, 2)
    radii = np.linalg.norm(final_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        final_pts[mask] = final_pts[mask] / radii[mask, np.newaxis] * 0.999999
    
    return final_pts
# EVOLVE-BLOCK-END
