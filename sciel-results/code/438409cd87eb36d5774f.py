"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # 1. Initialize with a deterministic Halton sequence (low-discrepancy lattice)
    # This provides a much better starting distribution than the previous Fibonacci line.
    def halton_sequence(n, dims):
        points = np.zeros((n, dims))
        for d in range(dims):
            base = d + 2  # Bases: 2, 3, 5
            for i in range(n):
                num = i + 1
                res = 0.0
                f = 1.0 / base
                while num > 0:
                    f *= base
                    num, r = divmod(num, base)
                    res += r / f
                points[i, d] = res
        return points

    # Generate points in [0, 1]^3
    init_points = halton_sequence(N, 3)
    # Map to [-0.5, 0.5]^3
    centers_init = init_points - 0.5
    
    # 2. Define smooth objective function for L-BFGS-B
    # We maximize the soft-minimum of pairwise distances and boundary clearances.
    # Soft-min: -1/alpha * log(sum(exp(-alpha * values)))
    alpha = 50.0
    
    def soft_min_objective(params):
        centers = params.reshape(-1, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Get upper triangle indices to avoid double counting
        i_upper, j_upper = np.triu_indices(N, k=1)
        pair_dists = dists[i_upper, j_upper]
        
        # Boundary clearances
        # Distance to nearest face for each point
        # min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
        boundary_clearances = np.min(0.5 - np.abs(centers), axis=1)
        
        # Combine all values into a single array for soft-min
        # Note: We want to maximize the minimum, so we minimize -soft_min
        # values = [pair_dists / 2, boundary_clearances]
        # The radius is limited by min(pair_dist/2, boundary_clearance)
        
        vals = np.concatenate([pair_dists / 2.0, boundary_clearances])
        
        # Soft min calculation
        # To avoid overflow in exp, subtract max
        max_val = np.max(vals)
        exp_vals = np.exp(-alpha * (vals - max_val))
        log_sum = np.log(np.sum(exp_vals))
        soft_min_val = max_val - log_sum / alpha
        
        return -soft_min_val # Minimize negative

    def soft_min_gradient(params):
        centers = params.reshape(-1, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        i_upper, j_upper = np.triu_indices(N, k=1)
        pair_dists = dists[i_upper, j_upper]
        
        # Boundary clearances
        boundary_clearances = np.min(0.5 - np.abs(centers), axis=1)
        
        vals = np.concatenate([pair_dists / 2.0, boundary_clearances])
        
        max_val = np.max(vals)
        exp_vals = np.exp(-alpha * (vals - max_val))
        sum_exp = np.sum(exp_vals)
        
        # Gradient of soft_min w.r.t vals
        # d(soft_min)/d(vals_k) = (alpha * exp(-alpha * (vals_k - max_val))) / sum_exp
        # Note: The -max_val term makes it constant w.r.t vals in the derivative of the log-sum-exp part?
        # Let L = -1/alpha * log(sum(exp(-alpha(v_k - M))))
        # dL/dv_k = -1/alpha * (1/sum) * d(sum)/dv_k
        # d(sum)/dv_k = sum(exp(-alpha(v_i - M)) * (-alpha * delta_ik))
        # = -alpha * exp(-alpha(v_k - M))
        # So dL/dv_k = -1/alpha * (1/sum) * (-alpha * exp(...)) = exp(...)/sum
        
        weights = exp_vals / sum_exp
        
        grad_centers = np.zeros_like(centers)
        
        # Gradient from pairwise distances
        # d(pair_dists[k])/d(centers[i]) = (centers[i] - centers[j]) / dists[i,j]
        # Note: pair_dists[k] corresponds to (i_upper[k], j_upper[k])
        for k in range(len(i_upper)):
            i = i_upper[k]
            j = j_upper[k]
            d = dists[i, j]
            if d < 1e-10:
                continue
            # d(val_k)/d(centers[i]) where val_k = d/2
            # d(d/2)/d(centers[i]) = 0.5 * (centers[i] - centers[j]) / d
            grad_vec = 0.5 * (centers[i] - centers[j]) / d
            grad_centers[i] += weights[k] * grad_vec
            grad_centers[j] -= weights[k] * grad_vec # Gradient w.r.t j is negative
            
        # Gradient from boundary clearances
        # val_k = min(0.5 - |c_x|, 0.5 - |c_y|, 0.5 - |c_z|)
        # This is non-smooth. We use a subgradient or the gradient of the active constraint.
        # For a smooth approximation, we can just use the gradient of the minimum component.
        # If we treat boundary_clearances as the input to the soft-min, we need d(boundary_k)/d(centers[i])
        for i in range(N):
            k = len(pair_dists) + i # Index in vals
            # Find which axis is the minimum for this point
            c = centers[i]
            dists_to_faces = 0.5 - np.abs(c)
            min_idx = np.argmin(dists_to_faces)
            
            # Gradient of (0.5 - |c_min_idx|) w.r.t c
            # d(0.5 - |c_a|)/dc_a = -sign(c_a)
            grad_b = np.zeros(3)
            if c[min_idx] >= 0:
                grad_b[min_idx] = -1.0
            else:
                grad_b[min_idx] = 1.0
                
            grad_centers[i] += weights[k] * grad_b
            
        return grad_centers.flatten()

    # 3. Optimize using L-BFGS-B
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        soft_min_objective, 
        centers_init.flatten(), 
        jac=soft_min_gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    centers = result.x.reshape(-1, 3)
    
    # 4. Polish with Nelder-Mead on the exact (non-smooth) objective
    def exact_objective(params):
        centers = params.reshape(-1, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        boundary_clearance = np.min(0.5 - np.abs(centers))
        
        r = min(min_pair_dist / 2.0, boundary_clearance)
        return -r

    result2 = minimize(
        exact_objective, 
        centers, 
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    centers_final = result2.x.reshape(-1, 3)
    
    # Ensure centers are strictly inside or on the boundary
    centers_final = np.clip(centers_final, -0.5, 0.5)
    
    return centers_final
# EVOLVE-BLOCK-END
