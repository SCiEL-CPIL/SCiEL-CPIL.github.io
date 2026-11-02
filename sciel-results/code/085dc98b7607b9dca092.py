import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization
    # 5x5x4 grid
    m_x, m_y, m_z = 5, 5, 4
    x_coords = (np.arange(m_x) + 0.5) / m_x - 0.5
    y_coords = (np.arange(m_y) + 0.5) / m_y - 0.5
    z_coords = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    centers = grid[:N].copy()
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Ensure inside cube
    centers = np.clip(centers, -0.5, 0.5)
    
    def compute_radius(centers):
        # Min distance between centers
        # Use broadcasting for vectorization
        # dists shape (N, N)
        dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to walls
        # centers shape (N, 3)
        # dist to wall x: 0.5 - |cx|
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)
    
    # Initial radius
    best_radius = compute_radius(centers)
    best_centers = centers.copy()
    
    # Optimization setup
    # We will use a sequence of alphas
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    def soft_min_obj_and_grad(x_flat, alpha):
        centers = x_flat.reshape(N, 3)
        
        # Calculate all pair distances
        # To save memory/time, we can compute only i<j? 
        # But for gradient, we need the sum over all.
        # N=100, N^2=10000. This is small.
        
        # Pairwise distances
        diff = centers[:, np.newaxis] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        # We only care about i<j for the objective sum to avoid double counting? 
        # Actually, sum over i<j is half the sum over i!=j.
        # Let's use the full matrix and divide by 2, or just sum upper triangle.
        # For gradient, it's easier to think in terms of forces.
        
        # Wall distances
        # d_ik = 0.5 - |c_ik|
        # This is non-differentiable at c_ik=0. 
        # Smooth approximation: d_ik = 0.5 - sqrt(c_ik^2 + eps)
        # Or just use the absolute value and handle gradient carefully.
        # grad of 0.5 - |x| is -sign(x).
        # Let's use the smooth version for stability: 0.5 - np.sqrt(c**2 + 1e-8)
        
        eps = 1e-8
        wall_dists = 0.5 - np.sqrt(centers**2 + eps) # shape (N, 3)
        
        # Combine all distances into a single array for log-sum-exp
        # Pair dists: use upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Wall dists: flatten
        wall_dists_flat = wall_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        
        # Log-sum-exp
        m = np.min(all_dists)
        # If m is very small, exp(-alpha*(u-m)) is fine.
        exps = np.exp(-alpha * (all_dists - m))
        sum_exps = np.sum(exps)
        log_sum = np.log(sum_exps)
        
        obj = -(m + log_sum / alpha)
        
        # Gradient
        # dS/du_k = - exp(-alpha*(u_k - m)) / sum_exps
        weights = exps / sum_exps
        
        # Gradient of pair distances
        # d/dx_i |x_i - x_j| = (x_i - x_j) / |x_i - x_j|
        # grad_obj = -sum_k weights_k * grad_u_k
        # We want to minimize obj = -S. So grad_min = -grad_S.
        # grad_S = - sum weights * grad_u
        # grad_min = sum weights * grad_u
        
        grad_flat = np.zeros_like(x_flat)
        
        # Pair contributions
        # For each pair (i, j), weight w_ij
        # grad_u_ij wrt x_i = (x_i - x_j) / d_ij
        # grad_u_ij wrt x_j = (x_j - x_i) / d_ij = - grad_u_ij wrt x_i
        
        # Vectorized update
        # diff shape (N, N, 3)
        # We need to accumulate forces.
        # Force on i from j: w_ij * (x_i - x_j) / d_ij
        
        # Create a force matrix
        forces = np.zeros((N, N, 3))
        
        # Only upper triangle weights
        w_pair = weights[:len(pair_dists)]
        
        # dists is (N, N)
        # diff is (N, N, 3)
        # We need d_ij in denominator
        # dists[i, j]
        
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Unit vectors
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Weights matrix (N, N)
        w_mat = np.zeros((N, N))
        w_mat[i_idx, j_idx] = w_pair
        w_mat[j_idx, i_idx] = w_pair
        
        # Force on i is sum_j w_ij * unit_vec_ij
        # Note: unit_vec_ij = (x_i - x_j)/d_ij
        # So force pushes i away from j.
        # This is the gradient of the distance.
        # We want to maximize distance, so we move in direction of gradient.
        # We are minimizing -S, so we move in direction of grad_min = sum w * grad_u.
        # So grad_flat += forces.
        
        # Sum over j for each i
        pair_grad = np.sum(w_mat[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall contributions
        # d_ik = 0.5 - sqrt(c_ik^2 + eps)
        # grad d_ik / dc_ik = - c_ik / sqrt(c_ik^2 + eps)
        # We want to maximize d_ik, so we move in direction of -c_ik / sqrt(...)
        # This pushes c_ik towards 0? No.
        # If c_ik > 0, grad is negative. We want to increase d.
        # d increases as c_ik decreases (moves towards 0 from positive side? No, 0.5 - |c|).
        # If c is positive, 0.5 - c. To increase this, decrease c.
        # Grad of (0.5 - c) is -1.
        # So we move in direction -1 * c?
        # Let's check: f(c) = 0.5 - |c|.
        # If c > 0, f = 0.5 - c, f' = -1.
        # We want to maximize f. Gradient ascent: c <- c + eta * f' = c - eta.
        # This moves c towards 0. Correct.
        # If c < 0, f = 0.5 + c, f' = 1.
        # c <- c + eta. Moves c towards 0. Correct.
        
        # So the "force" from wall is -sign(c_ik).
        # Using smooth approx: -c_ik / sqrt(c_ik^2 + eps).
        
        w_wall = weights[len(pair_dists):]
        
        # w_wall shape (N*3)
        # centers shape (N, 3)
        # grad_wall = - centers / sqrt(centers^2 + eps)
        # Weighted sum
        wall_grad = np.zeros((N, 3))
        w_wall_mat = w_wall.reshape(N, 3)
        
        smooth_abs = np.sqrt(centers**2 + eps)
        wall_forces = -centers / smooth_abs
        
        wall_grad = w_wall_mat * wall_forces
        
        total_grad = pair_grad + wall_grad
        
        return obj, total_grad.flatten()

    # Optimization loop
    current_x = centers.flatten()
    
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, alpha)[0],
                current_x,
                jac=lambda x: soft_min_obj_and_grad(x, alpha)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 50, 'maxls': 50}
            )
            current_x = res.x
            centers = current_x.reshape(N, 3)
            
            # Check validity
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
        except Exception as e:
            # If optimization fails, continue with next alpha
            pass

    # Final check
    final_r = compute_radius(best_centers)
    if final_r > best_radius:
        best_radius = final_r

    return best_centers
