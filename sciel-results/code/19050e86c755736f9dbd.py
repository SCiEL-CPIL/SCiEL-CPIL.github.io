"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize

def construct():
    # 1. Initialization: Fermat Spiral
    # This provides a very uniform distribution in the unit disk.
    i = np.arange(N)
    # Golden angle
    theta = i * 2.0 * np.pi * (1.0 + np.sqrt(5.0)) / 2.0
    # Radius proportional to sqrt(i) to ensure uniform density
    r = np.sqrt(i / N)
    
    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    
    # 2. Objective Function and Gradient
    # We want to maximize the minimum distance.
    # Approximation: Minimize the sum of repulsive potentials 1/d^2.
    # This pushes all points apart.
    # We also add a penalty for going outside the unit circle.
    
    def objective_and_gradient(p):
        # p is a flat array of shape (2*N,)
        pts = p.reshape(N, 2)
        
        # Compute pairwise differences
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, 1.0)
        
        # Repulsive energy: sum_{i<j} 1/d_ij^2
        # We sum over all i != j, so we divide by 2 at the end to match the sum over i<j
        # Actually, minimizing sum_{i!=j} 1/d^2 is equivalent to minimizing sum_{i<j} 2/d^2
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        E_rep = np.sum(inv_dist_sq) / 2.0
        
        # Gradient of E_rep
        # d/dx_i (1/d_ij^2) = -2 * (x_i - x_j) / d_ij^4
        # We can compute this vectorially.
        # grad_rep[i] = sum_{j != i} -2 * (pts[i] - pts[j]) / dist_sq[i,j]^2
        # Note: dist_sq[i,j] is squared distance.
        # term = diff / (dist_sq**2)[:,:,np.newaxis]
        # But we need to sum over j for each i.
        
        # Let's compute the gradient term by term
        # grad_rep = -2 * sum_j (pts[i] - pts[j]) / dist_sq[i,j]^2
        # diff is pts[i] - pts[j]
        # We need dist_sq^2.
        dist_sq_safe = np.where(dist_sq > 0, dist_sq, 1.0)
        inv_dist_sq_2 = 1.0 / (dist_sq_safe**2)
        np.fill_diagonal(inv_dist_sq_2, 0.0)
        
        # grad_rep[i] = -2 * sum_j diff[i,j] * inv_dist_sq_2[i,j]
        # diff shape: N x N x 2
        # inv_dist_sq_2 shape: N x N
        # Multiply diff by inv_dist_sq_2[:, :, np.newaxis]
        weighted_diff = diff * inv_dist_sq_2[:, :, np.newaxis]
        # Sum over j (axis 1)
        grad_rep = -2.0 * np.sum(weighted_diff, axis=1)
        
        # Boundary Penalty
        # P = sum_i max(0, |p_i| - 1)^2
        # If |p_i| > 1, P_i = (|p_i| - 1)^2
        # dP/dp_i = 2 (|p_i| - 1) * (p_i / |p_i|)
        
        norms = np.linalg.norm(pts, axis=1)
        # Mask for points outside
        outside_mask = norms > 1.0
        
        P = 0.0
        grad_P = np.zeros_like(pts)
        
        if np.any(outside_mask):
            # Only compute for outside points
            # Let r_i = norms[outside_mask]
            # P = sum (r_i - 1)^2
            # grad = 2 (r_i - 1) * p_i / r_i
            r_out = norms[outside_mask]
            p_out = pts[outside_mask]
            
            P_val = np.sum((r_out - 1.0)**2)
            # Gradient factor: 2 * (r - 1) / r
            factor = 2.0 * (r_out - 1.0) / r_out
            grad_P[outside_mask] = p_out * factor[:, np.newaxis]
            P = P_val
        
        # Total Objective
        # We need to balance repulsion and boundary.
        # A large lambda ensures points stay inside.
        lambda_pen = 100.0
        total_obj = E_rep + lambda_pen * P
        
        # Total Gradient
        total_grad = grad_rep + lambda_pen * grad_P
        
        return total_obj, total_grad.flatten()

    # 3. Optimization
    # Use L-BFGS-B
    # We might need to run this for a few seconds.
    # Time limit is 120s.
    
    # Initial point
    p0 = points.flatten()
    
    # Run minimization
    try:
        res = scipy.optimize.minimize(
            objective_and_gradient, 
            p0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        points_opt = res.x.reshape(N, 2)
    except Exception:
        points_opt = points

    # 4. Final Check and Adjustment
    # Ensure all points are strictly inside.
    # If any point is outside, project it back to the boundary?
    # The penalty should have handled this, but let's be safe.
    norms = np.linalg.norm(points_opt, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            points_opt[i] = points_opt[i] / norms[i] * 0.999999

    return points_opt
# EVOLVE-BLOCK-END
