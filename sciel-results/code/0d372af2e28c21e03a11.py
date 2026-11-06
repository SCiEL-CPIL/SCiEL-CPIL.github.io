"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def construct():
    # New Approach: Gradient-based optimization (BFGS) on a smooth potential
    # that penalizes small inter-circle distances and small boundary distances.
    # This is distinct from the Simulated Annealing approach.
    
    # Objective: Minimize V = sum_{i<j} exp(-k * d_ij^2) + sum_i exp(-k * (1-|c_i|)^2)
    # This encourages all distances to be large.
    
    k = 15.0 # Stiffness parameter
    
    def potential_and_grad(x):
        pts = x.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Only consider lower triangle to avoid double counting and self
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_sq = dist_sq[i_idx, j_idx]
        
        # Potential term for pairs
        # V_pair = sum(exp(-k * d_sq))
        # grad_wrt_pts_i for pair (i,j): 
        # d/dx_i exp(-k * ||x_i - x_j||^2) = exp(...) * (-k * 2 * (x_i - x_j))
        
        V = 0.0
        grad = np.zeros_like(pts)
        
        # Compute exponentials
        exp_vals = np.exp(-k * d_sq)
        V += np.sum(exp_vals)
        
        # Gradients for pairs
        # For each pair (i, j), force on i is -2 * k * exp_val * (pts[i] - pts[j])
        # Force on j is -2 * k * exp_val * (pts[j] - pts[i]) = 2 * k * exp_val * (pts[i] - pts[j])
        
        diffs = pts[i_idx] - pts[j_idx]
        forces = 2.0 * k * exp_vals[:, np.newaxis] * diffs
        
        # Accumulate gradients
        # np.add.at is efficient enough for N=50
        np.add.at(grad, i_idx, -forces)
        np.add.at(grad, j_idx, forces)
        
        # Boundary term
        # V_bound = sum(exp(-k * (1 - |c_i|)^2))
        # Let r_i = |c_i|. d/dr_i exp(-k(1-r_i)^2) = exp(...) * (-k * 2 * (1-r_i) * (-1)) = 2*k*(1-r_i)*exp(...)
        # d/dx_i r_i = x_i / r_i
        # So grad_i = 2*k*(1-r_i)*exp(...) * (x_i / r_i)
        
        r_norms = np.linalg.norm(pts, axis=1)
        # Avoid division by zero
        r_safe = np.where(r_norms < 1e-6, 1e-6, r_norms)
        dist_to_bound = 1.0 - r_norms
        # Clip dist_to_bound to avoid large positive gradients if outside? 
        # If outside, 1-r is negative. (1-r)^2 is positive. 
        # The potential is symmetric around r=1? No, exp(-k(1-r)^2) is max at r=1.
        # We want to minimize V, so we want to maximize (1-r)^2? 
        # Wait. If we are inside, r < 1, dist_to_bound > 0.
        # We want dist_to_bound to be large.
        # exp(-k * dist^2) is small when dist is large.
        # So minimizing V pushes dist to be large. Correct.
        
        exp_bound = np.exp(-k * dist_to_bound**2)
        V += np.sum(exp_bound)
        
        grad_bound_coeffs = 2.0 * k * dist_to_bound * exp_bound / r_safe
        # Note: if r_i is very small, gradient direction is undefined, but magnitude is small if exp is small?
        # Actually if r_i=0, dist=1, exp(-k) is small.
        # The term is grad_bound_coeffs * pts[i]
        
        grad += grad_bound_coeffs[:, np.newaxis] * pts
        
        return V, grad

    def run_optimizer(initial_pts):
        x0 = initial_pts.flatten()
        # Use BFGS, it's robust for small N
        res = minimize(potential_and_grad, x0, jac=True, method='BFGS', 
                       options={'maxiter': 1000, 'gtol': 1e-8})
        return res.x.reshape(N, 2)

    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(np.min(dist_sq))
        
        r_norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = 1.0 - np.max(r_norms)
        
        return min(min_dist / 2.0, min_boundary_dist)

    best_r = 0.0
    best_pts = None
    
    # Multi-start strategy
    np.random.seed(42)
    
    # Start 1: Hexagonal Grid
    # Approximate hex packing
    hex_pts = []
    # Generate points in a hexagonal pattern
    # Radius of hexagon cell ~ 2*r. We don't know r yet.
    # Just generate a dense grid and filter?
    # Or use a known good structure: 1 center, 6 ring, 12 ring, etc.
    # For 50 circles:
    # 1 + 6 + 12 + 18 = 37. Need 13 more.
    # Let's just use random starts with projection to be safe and simple, 
    # but BFGS is sensitive to initialization.
    # A better initialization: Random points, then a few steps of repulsion relaxation.
    
    for run in range(5):
        # Random initialization
        pts = np.random.rand(N, 2) * 1.8 - 0.9
        # Project inside
        r_norms = np.linalg.norm(pts, axis=1)
        mask = r_norms > 0.9
        pts[mask] = pts[mask] / r_norms[mask, None] * 0.9
        
        # Quick relaxation to remove obvious overlaps before BFGS
        # This helps BFGS converge to a better local minimum
        for _ in range(50):
            # Simple repulsion
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            # Force
            forces = np.zeros_like(pts)
            # Only for close pairs to save time? N=50 is small.
            # F = 1/d^2 * diff
            # Clip dist to avoid blow up
            dist_clipped = np.maximum(dist, 0.1)
            F_mag = 1.0 / (dist_clipped**2)
            F_mag[dist_clipped < 0.1] = 10.0 # Strong repulsion
            
            # Sum forces
            # F_ij = F_mag * (p_i - p_j) / dist
            # This is O(N^3) if done naively in python, but vectorized it's fine.
            # Actually, let's just do a simple gradient step on the same potential
            V, grad = potential_and_grad(pts.flatten())
            pts -= 0.01 * grad.reshape(N, 2)
            
            # Project inside
            r_norms = np.linalg.norm(pts, axis=1)
            mask = r_norms > 0.99
            pts[mask] = pts[mask] / r_norms[mask, None] * 0.99

        # Run BFGS
        opt_pts = run_optimizer(pts)
        
        # Post-process: Ensure validity
        r_norms = np.linalg.norm(opt_pts, axis=1)
        mask = r_norms > 0.999
        if np.any(mask):
            opt_pts[mask] = opt_pts[mask] / r_norms[mask, None] * 0.999
            
        r_val = calc_radius(opt_pts)
        if r_val > best_r:
            best_r = r_val
            best_pts = opt_pts.copy()
            
    # Final check
    if best_pts is None:
        # Fallback to random if something went wrong
        best_pts = np.random.rand(N, 2) * 1.8 - 0.9
        
    return best_pts

# EVOLVE-BLOCK-END
