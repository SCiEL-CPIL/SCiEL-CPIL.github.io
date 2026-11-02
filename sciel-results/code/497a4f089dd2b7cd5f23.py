"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization with local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    max_k = int(1.0 / (s * 0.5)) + 3
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 4
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    # Use a slightly larger perturbation to help escape local minima
    points += np.random.normal(0, 1e-3, points.shape)
    
    return points

def _local_polish(points, max_iter=1000, deadline=None):
    """
    Perform a local polish by directly pushing overlapping circles apart.
    This helps fix small violations that smooth optimization might miss.
    """
    points = points.copy()
    n = len(points)
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    
    # Estimate current radius
    r = _compute_score(points)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        
        # 1. Handle boundary violations
        norms = np.linalg.norm(points, axis=1)
        # If a point is outside the "safe" zone (radius 1-r), pull it in
        # We use r as the target radius. If |p| > 1-r, it violates the constraint.
        # Actually, the constraint is |p| + r <= 1 => |p| <= 1-r.
        # We want to maximize r, so we treat current r as a lower bound.
        # If |p| > 1-r, we must move it in.
        limit = 1.0 - r
        outside_mask = norms > limit
        if np.any(outside_mask):
            # Pull points towards center
            # Move them to the boundary of the safe zone
            # Or just reduce their norm to limit
            points[outside_mask] = points[outside_mask] / norms[outside_mask, np.newaxis] * limit
            improved = True
        
        # 2. Handle pair violations
        # We want |p_i - p_j| >= 2r
        vecs = points[i_idx] - points[j_idx]
        dists = np.sqrt(np.sum(vecs**2, axis=1))
        
        # Find pairs that are too close
        # dist < 2r
        close_mask = dists < 2.0 * r
        
        if np.any(close_mask):
            # Get the indices of close pairs
            close_i = i_idx[close_mask]
            close_j = j_idx[close_mask]
            close_vecs = vecs[close_mask]
            close_dists = dists[close_mask]
            
            # Avoid division by zero
            valid = close_dists > 1e-12
            if np.any(valid):
                close_i = close_i[valid]
                close_j = close_j[valid]
                close_vecs = close_vecs[valid]
                close_dists = close_dists[valid]
                
                # We need to push them apart.
                # Target distance is 2r.
                # Current distance is d.
                # We need to move them by (2r - d) / 2 in opposite directions.
                # Direction is unit vector v = vec / d.
                # p_i += (2r - d)/2 * v
                # p_j -= (2r - d)/2 * v
                
                # Calculate displacement
                # Note: vec = p_i - p_j. So p_i is in direction of vec from p_j.
                # To increase distance, we move p_i in direction of vec, and p_j in direction of -vec.
                # Displacement magnitude for each: (2r - d) / 2.
                
                # However, we must be careful not to cause new violations.
                # A simple gradient step is safer.
                # grad for p_i: v / d * (something)
                # Let's just do a small step in the direction of separation.
                # step_size = 0.5 * (2r - d)
                
                # Accumulate displacements to avoid conflicting updates
                disp = np.zeros_like(points)
                # For each close pair, add displacement to i and j
                # disp_i += (2r - d) * v / (2 * d) ? No.
                # Let's use a simple push:
                # push_vec = (2r - d) * (vec / d)
                # p_i += push_vec / 2
                # p_j -= push_vec / 2
                
                # To handle multiple constraints, we can sum the gradients.
                # Gradient of distance d wrt p_i is vec/d.
                # We want to increase d. So move p_i in direction vec/d.
                # Move p_j in direction -vec/d.
                
                # Normalize vec to get unit vector
                unit_vecs = close_vecs / close_dists[:, np.newaxis]
                
                # Magnitude of push: proportional to violation (2r - d)
                # Let's use a damping factor to ensure stability
                damping = 0.5
                push_mags = (2.0 * r - close_dists) * damping
                
                # Accumulate
                np.add.at(disp, close_i, unit_vecs * push_mags[:, np.newaxis])
                np.add.at(disp, close_j, -unit_vecs * push_mags[:, np.newaxis])
                
                points += disp
                improved = True
        
        # Re-evaluate score
        new_r = _compute_score(points)
        
        if new_r > r:
            r = new_r
        else:
            # If we didn't improve, we might be oscillating or stuck.
            # Reduce the target radius slightly to allow movement?
            # Or just break if no improvement for a while.
            pass
            
        if not improved:
            break
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialization
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min calculation
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        log_sum_exp = max_z + np.log(sum_exp)
        
        val = log_sum_exp / beta
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    # Optimization loop with increasing beta
    # Schedule: start low to allow global rearrangement, end high for precision
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        try:
            # Adjust tolerances based on beta
            if beta >= 10000:
                maxiter = 500
                gtol = 1e-13
                ftol = 1e-16
                maxls = 80
            elif beta >= 1000:
                maxiter = 500
                gtol = 1e-11
                ftol = 1e-14
                maxls = 50
            elif beta >= 100:
                maxiter = 400
                gtol = 1e-10
                ftol = 1e-13
                maxls = 40
            else:
                maxiter = 300
                gtol = 1e-9
                ftol = 1e-12
                maxls = 30
            
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'gtol': gtol,
                    'ftol': ftol,
                    'maxls': maxls,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 5:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                current_points = best_points
                
        except Exception:
            pass

    # Final polish with local direct push
    if time.time() < deadline - 2:
        polished_points = _local_polish(best_points, max_iter=2000, deadline=deadline - 1)
        polished_score = _compute_score(polished_points)
        
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
