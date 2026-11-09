import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

def _initial_hex_lattice(n):
    """Generate a hexagonal lattice of points inside the unit disk."""
    # Estimate spacing
    # Area per point in hex lattice is s^2 * sqrt(3)/2
    # Total area pi. n * s^2 * sqrt(3)/2 ~ pi
    # s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    # Scale down slightly to ensure they fit and allow expansion
    s *= 0.98
    
    points = []
    # Generate grid
    # Range of i, j such that (x,y) is in [-1,1]
    # x = i*s + j*s*0.5
    # y = j*s*sqrt(3)/2
    max_j = int(2.0 / (s * np.sqrt(3.0) / 2.0)) + 2
    max_i = int(2.0 / s) + 2
    
    for j in range(-max_j, max_j + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-max_i, max_i + 1):
            x = i * s + j * s * 0.5
            r = np.hypot(x, y)
            if r <= 1.0 - s/2.0:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) == 0:
        # Fallback
        return np.random.rand(n, 2) * 0.8 - 0.4
        
    if len(points) < n:
        # If we don't have enough, we need to add more or scale up
        # But for N=300, this shouldn't happen with s*0.98
        # If it does, just return what we have and pad? No, must return N.
        # Let's just return random if this fails, but it shouldn't.
        pass
        
    # Sort by radius to take the most central ones
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii, kind='stable')
    points = points[idx[:n]]
    
    return points

def _compute_score(points):
    """Compute the packing score."""
    n = len(points)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1])
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    return min(min_pair / 2.0, min_boundary)

def _local_refine(points, max_iters=10):
    """
    Refine points using coordinate descent.
    For each point, optimize its position to maximize min distance to neighbors and boundary.
    """
    n = len(points)
    points = points.copy()
    
    # Build initial tree
    tree = cKDTree(points)
    
    for it in range(max_iters):
        improved = False
        # Random order to avoid bias
        order = np.random.permutation(n)
        
        for i in order:
            # Find neighbors within a reasonable distance (e.g., 4 * current estimated radius)
            # Estimate current radius
            # Using a slightly larger radius for neighbor search to be safe
            r_est = _compute_score(points) * 1.5 + 0.01
            neighbors = tree.query_ball_point(points[i], r_est)
            neighbors = [j for j in neighbors if j != i]
            
            if not neighbors:
                continue
                
            # Define objective for point i
            # We want to maximize min( min_j |p_i - p_j|/2, 1 - |p_i| )
            # Equivalent to minimizing -min(...)
            # Use soft-min or just Nelder-Mead on the exact min?
            # Nelder-Mead on exact min is non-smooth but works well in 2D.
            
            def obj_2d(x):
                p_new = x
                # Distance to boundary
                d_bound = 1.0 - np.linalg.norm(p_new)
                # Distance to neighbors
                if neighbors:
                    # Vectorized distance calculation
                    diffs = points[neighbors] - p_new
                    dists = np.linalg.norm(diffs, axis=1)
                    d_min = np.min(dists) / 2.0
                else:
                    d_min = 1.0 # Large value if no neighbors
                
                val = min(d_bound, d_min)
                return -val
            
            x0 = points[i]
            try:
                res = minimize(obj_2d, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < -obj_2d(x0): # If it improved
                    points[i] = res.x
                    # Update tree? 
                    # cKDTree doesn't support efficient updates. 
                    # For N=300, rebuilding is fast enough.
                    tree = cKDTree(points)
                    improved = True
            except Exception:
                pass
                
        if not improved:
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Initial hexagonal lattice
    points = _initial_hex_lattice(N)
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Step 2: Global Optimization with L-BFGS-B
    # We will use a soft-min objective to make it differentiable.
    # Objective: -soft_min( min_pair/2, min_boundary )
    
    def soft_min_obj(x, beta):
        pts = x.reshape(N, 2)
        # Boundary
        radii = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - radii
        
        # Pairwise
        # Use KDTree to get nearest neighbors? 
        # Soft-min over ALL pairs is expensive.
        # But for N=300, 45k pairs. 
        # Let's stick to the full calculation for correctness, but optimize if possible.
        # Actually, the previous code did full calculation.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        pair_vals = dist / 2.0
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_vals_flat = pair_vals[i_idx, j_idx]
        
        all_vals = np.concatenate([pair_vals_flat, boundary_dists])
        
        m = np.min(all_vals)
        shifted = all_vals - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            sum_exp = 1e-300
        log_sum_exp = np.log(sum_exp)
        soft_min_val = m - log_sum_exp / beta
        return -soft_min_val

    def soft_min_grad(x, beta):
        pts = x.reshape(N, 2)
        radii = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - radii
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        pair_vals = dist / 2.0
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_vals_flat = pair_vals[i_idx, j_idx]
        
        all_vals = np.concatenate([pair_vals_flat, boundary_dists])
        
        m = np.min(all_vals)
        shifted = all_vals - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            sum_exp = 1e-300
        
        weights = exp_terms / sum_exp
        
        grad_pts = np.zeros_like(pts)
        
        # Boundary grad
        b_weights = weights[len(pair_vals_flat):]
        safe_radii = np.where(radii > 1e-12, radii, 1.0)
        grad_boundary = (b_weights / safe_radii)[:, np.newaxis] * pts
        grad_pts += grad_boundary
        
        # Pair grad
        p_weights = weights[:len(pair_vals_flat)]
        diff_vec = pts[i_idx] - pts[j_idx]
        dist_vec = np.linalg.norm(diff_vec, axis=1)
        safe_dist = np.where(dist_vec > 1e-12, dist_vec, 1.0)
        
        grad_pair_vec = (diff_vec / (2.0 * safe_dist))[:, np.newaxis] * p_weights[:, np.newaxis]
        
        np.add.at(grad_pts, i_idx, -grad_pair_vec)
        np.add.at(grad_pts, j_idx, grad_pair_vec)
        
        return grad_pts.flatten()

    x0 = points.flatten()
    
    # Try a few betas
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 30.0:
            break
            
        try:
            result = minimize(
                soft_min_obj,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=soft_min_grad,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                x0 = new_points.flatten()
        except Exception:
            pass
            
    # Step 3: Local Refinement
    # This is crucial for improving the score.
    if time.time() < deadline - 10.0:
        refined_points = _local_refine(best_points, max_iters=5)
        refined_score = _compute_score(refined_points)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
            
        # Another round of global optimization?
        # Maybe just one more local refinement pass
        if time.time() < deadline - 5.0:
            refined_points2 = _local_refine(best_points, max_iters=5)
            refined_score2 = _compute_score(refined_points2)
            if refined_score2 > best_score:
                best_score = refined_score2
                best_points = refined_points2.copy()

    # Final validation
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0):
        mask = radii > 1.0
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
        
    return best_points
