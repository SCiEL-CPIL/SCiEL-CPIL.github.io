"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Initialization: Random points inside a slightly smaller disk
    # We start with a radius that ensures no initial overlaps to avoid singularities in force calculation
    # Area = pi, N=300. Area per circle ~ pi/300. r ~ sqrt(pi/300/pi) = 1/sqrt(300) ~ 0.057.
    # Container radius 1. Let's place centers within 0.9 to be safe.
    rng = np.random.default_rng(42)
    
    # Initial random placement
    # We want to avoid overlaps initially. 
    # If we just randomize in unit disk, they will overlap.
    # Better: Randomize in a grid or spiral? 
    # Let's use a simple repulsion simulation to spread them out.
    
    # Start with points on a spiral (Fermat spiral) which is a decent dense packing heuristic
    i = np.arange(N)
    # Golden angle
    phi = np.pi * (3. - np.sqrt(5.))
    # Radius scaling: r = sqrt(i/N) * R_max
    # R_max should be < 1 to leave room for radius
    R_max = 0.95
    r = np.sqrt(i / N) * R_max
    theta = i * phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack((x, y))
    
    # 2. Optimization: Maximize the minimum distance
    # We want to maximize r = min(min_{i<j} dist_ij / 2, min_i (1 - dist_i))
    # This is equivalent to minimizing -r.
    # Since the function is non-smooth (min), we can use a smooth approximation or just gradient descent on the min term.
    # A common approach: Minimize the sum of inverse distances (soft repulsion) to get close, then refine the min term.
    # Or, directly use a penalty method.
    
    # Let's use a two-stage approach:
    # Stage 1: Minimize potential energy U = sum(1/d_ij^2) + sum(1/d_i_boundary^2) to spread points.
    # Stage 2: Steepest ascent on the minimum distance.

    def energy(pts):
        # pts shape (N, 2)
        # Distance matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        # Avoid division by zero
        dist = np.maximum(dist, 1e-6)
        
        # Pairwise repulsion
        # We only need upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dist[i_idx, j_idx]
        # Soft repulsion: 1/d^2
        pair_energy = np.sum(1.0 / (pair_dists**2))
        
        # Boundary repulsion: Push points away from boundary
        # dist_to_boundary = 1 - norm(p)
        norms = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - norms
        # If a point is outside, penalty is huge. If inside, small penalty for being close.
        # We want to keep them inside.
        # Penalty: 1 / (1 - norm)^2 ? No, if norm > 1, this is negative.
        # Let's use: if norm > 1, penalty = (norm-1)^2 * 1000. Else 0.
        # But we want to push them *in* from the edge? No, we want to maximize the radius, so we want them as far from edge as possible?
        # Actually, maximizing min(1-norm) means we want points to be as close to center as possible? 
        # NO. min(1-|c_i|) is the distance to the boundary. We want this to be LARGE.
        # So we want points to be AWAY from the boundary.
        # So we want to MINIMIZE the max distance to boundary? No.
        # We want to maximize the minimum distance to boundary.
        # So in the energy minimization, we should penalize points that are CLOSE to the boundary?
        # Wait. If I move a point towards the center, its distance to boundary increases.
        # So yes, we want to push points towards the center?
        # But if I push ALL points to the center, they overlap.
        # The balance is between pair repulsion and boundary attraction?
        # Actually, for packing, we usually don't use boundary attraction. We just constrain them to be inside.
        # But here the "score" depends on the distance to the boundary.
        # So we DO want to maximize the minimum distance to the boundary.
        # This implies we want the "empty space" at the boundary to be maximized?
        # No, it implies the "worst-case" point (closest to boundary) should be as far from boundary as possible.
        
        # Let's stick to the definition:
        # r = min( min_pair_dist/2, min_boundary_dist )
        
        # For Stage 1 (Spreading), we just want to resolve overlaps.
        # We can ignore boundary distance in the energy, as long as we project back to unit disk.
        
        # Let's use a simpler energy: Sum of 1/d^2 for pairs.
        return pair_energy

    def energy_grad(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # Gradient of sum 1/d^2
        # d(1/d^2)/d(pts_i) = -2 * (pts_i - pts_j) / d^4
        # Sum over j
        # grad_i = sum_j -2 * (pts_i - pts_j) / dist_ij^4
        
        inv_d4 = 1.0 / (dist_sq**2)
        # Zero out diagonal
        np.fill_diagonal(inv_d4, 0.0)
        
        # grad = -2 * sum_j (pts_i - pts_j) * inv_d4_ij
        # diff is pts_i - pts_j
        # grad = -2 * (diff * inv_d4[..., np.newaxis]).sum(axis=1)
        
        grad = -2.0 * (diff * inv_d4[..., np.newaxis]).sum(axis=1)
        
        return grad

    # Simple Gradient Descent for Stage 1
    # Learning rate needs to be small.
    # Initial distances are ~ 0.05. 1/d^2 is ~ 400. Grad ~ 400 * 0.05 ~ 20.
    # Step size 0.001?
    
    alpha = 0.005
    for step in range(2000):
        if time.time() > deadline - 30: break # Reserve 30s for stage 2
        
        grad = energy_grad(points)
        points = points - alpha * grad
        
        # Project to unit disk (with a small margin to avoid numerical issues on boundary)
        norms = np.sqrt(np.sum(points**2, axis=1))
        mask = norms > 0.99
        if np.any(mask):
            points[mask] *= (0.99 / norms[mask])[:, np.newaxis]
        
        # Annealing: reduce step size
        alpha *= 0.995

    # Stage 2: Steepest Ascent on the Minimum Distance
    # Objective: Maximize f(x) = min( min_pair_dist/2, min_boundary_dist )
    # Gradient of min is the gradient of the active term.
    
    def get_min_dist_info(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dist[i_idx, j_idx]
        
        norms = np.sqrt(np.sum(pts**2, axis=1))
        boundary_dists = 1.0 - norms
        
        min_pair_dist = np.min(pair_dists)
        min_boundary_dist = np.min(boundary_dists)
        
        r_pair = min_pair_dist / 2.0
        r_bound = min_boundary_dist
        
        if r_pair < r_bound:
            # Active constraint is pair
            min_val = r_pair
            # Find the pair
            idx = np.argmin(pair_dists)
            i, j = i_idx[idx], j_idx[idx]
            # Gradient of dist_ij wrt pts_i and pts_j
            d_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(d_ij)
            if dist_ij < 1e-9:
                grad = np.zeros_like(pts)
            else:
                grad = np.zeros_like(pts)
                # d(dist)/d(pts_i) = d_ij / dist_ij
                # d(dist/2)/d(pts_i) = 0.5 * d_ij / dist_ij
                # We want to ASCEND, so move in direction of gradient
                grad[i] += 0.5 * d_ij / dist_ij
                grad[j] -= 0.5 * d_ij / dist_ij
        else:
            # Active constraint is boundary
            min_val = r_bound
            idx = np.argmin(boundary_dists)
            i = idx
            p = pts[i]
            norm_p = np.linalg.norm(p)
            if norm_p < 1e-9:
                grad = np.zeros_like(pts)
            else:
                grad = np.zeros_like(pts)
                # d(1 - norm)/d(p) = - p / norm
                # We want to ASCEND (increase 1-norm), so move in direction of -p/norm
                # Wait. 1-norm. If p is (1,0), 1-norm is 0. To increase, move to (0,0).
                # Gradient of (1 - |p|) is -p/|p|.
                # So we move in direction of -p/|p|.
                grad[i] -= p / norm_p
                
        return min_val, grad

    # Steepest Ascent
    # We want to maximize, so points = points + step * grad
    # But we must stay inside the disk.
    
    # Initial step size
    step_size = 0.005
    best_r = get_min_dist_info(points)[0]
    
    for step in range(5000):
        if time.time() > deadline: break
        
        current_r, grad = get_min_dist_info(points)
        
        # Normalize gradient to have unit norm for consistent step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad = grad / grad_norm
        
        # Propose new point
        new_points = points + step_size * grad
        
        # Project to unit disk
        norms = np.sqrt(np.sum(new_points**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
        # Check if valid (no overlaps? The score function handles it, but we want to improve)
        new_r = get_min_dist_info(new_points)[0]
        
        if new_r > current_r:
            points = new_points
            if new_r > best_r:
                best_r = new_r
            # Increase step size slightly if we made progress
            step_size *= 1.01
        else:
            # Decrease step size
            step_size *= 0.95
            # If step size is too small, stop
            if step_size < 1e-6:
                break
        
        # Cap step size
        step_size = min(step_size, 0.05)

    return points

# EVOLVE-BLOCK-END
