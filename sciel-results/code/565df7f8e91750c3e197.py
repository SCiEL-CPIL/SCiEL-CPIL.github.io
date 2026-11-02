import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # 1. Initialization: Hexagonal Lattice
    # We want a dense hexagonal lattice.
    # The optimal spacing is roughly 2*r_opt.
    # r_opt ~ 0.053, so spacing ~ 0.106.
    # Let's generate a grid and pick the best N points.

    def generate_hex_grid(spacing):
        points = []
        max_j = int(2.0 / (spacing * np.sqrt(3.0))) + 2
        max_i = int(2.0 / spacing) + 2

        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        return np.array(points)

    # Try a few spacings to find the best initial configuration
    best_init = None
    best_init_score = -np.inf

    # Spacing around 0.106
    spacings = [0.10, 0.105, 0.106, 0.107, 0.11, 0.115]

    for sp in spacings:
        pts = generate_hex_grid(sp)
        if len(pts) < N:
            continue

        # Select points closest to the center to create a compact cluster
        dists_to_center = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        idx = np.argsort(dists_to_center)[:N]
        init_pts = pts[idx].copy()

        # Calculate initial score
        d = pdist(init_pts)
        min_pair = np.min(d)
        min_bound = np.min(1.0 - dists_to_center[idx])
        score = min(min_pair / 2.0, min_bound)

        if score > best_init_score:
            best_init_score = score
            best_init = init_pts

    if best_init is None:
        # Fallback: Random
        best_init = np.random.randn(N, 2)
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_init *= 0.9

    current_points = best_init

    # 2. Optimization using Soft-Min and L-BFGS-B
    # We precompute the upper triangle indices for efficient pairwise distance calculation
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])

    def objective_and_grad(flat_pts, beta):
        pts = flat_pts.reshape(N, 2)

        # Calculate pairwise distances using precomputed indices
        # This is faster than full matrix or pdist in some contexts, but pdist is C-optimized.
        # However, we need gradients, so we need the vectors.
        # Let's stick to the vectorized difference approach which is clean for gradients.
        
        # Efficient pairwise distance calculation
        # diff[i, j, :] = pts[i] - pts[j] for i < j
        vecs = pts[iu[0]] - pts[iu[1]]
        pair_dists = np.sqrt(np.sum(vecs ** 2, axis=1) + 1e-12)
        
        # Boundary distances
        dists_to_center = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        d_bound = 1.0 - dists_to_center

        # Effective radii: min(pair_dist/2, bound_dist)
        # We want to maximize min(pair_dist/2, bound_dist)
        # So we minimize -soft_min([pair_dist/2, bound_dist], beta)
        
        eff_rad_pairs = pair_dists / 2.0
        eff_rad_bounds = d_bound
        
        # Combine for soft min
        # z = -beta * eff_rad
        z_pairs = -beta * eff_rad_pairs
        z_bounds = -beta * eff_rad_bounds
        
        # Log-sum-exp trick for stability
        max_z = max(np.max(z_pairs), np.max(z_bounds))
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # Objective value: -soft_min = (1/beta) * log(sum(exp(-beta * eff_rad)))
        # soft_min = -1/beta * log(sum(exp(-beta * eff_rad)))
        # -soft_min = 1/beta * log(sum(exp(-beta * eff_rad)))
        # log(sum(exp(z - max_z))) + max_z = log_sum_exp
        # log_sum_exp = max_z + log(sum_exp)
        # obj = (max_z + log(sum_exp)) / beta
        obj = (max_z + np.log(sum_exp)) / beta
        
        # Gradients
        # d(obj)/d(eff_rad_i) = (1/beta) * d/d(eff_rad_i) log(sum(exp(-beta*eff_rad)))
        # = (1/beta) * sum_j exp(-beta*eff_rad_j) * (-beta) * delta_ij / sum_j exp(-beta*eff_rad_j)
        # = - exp(-beta*eff_rad_i) / sum_j exp(-beta*eff_rad_j)
        # Note: The gradient of the soft min w.r.t input x_i is exp(-beta*x_i)/sum(exp(-beta*x_j)).
        # Since we are minimizing -soft_min, the gradient is - (exp(-beta*x_i)/sum(exp(-beta*x_j))).
        
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        # d(obj)/d(pair_dist_k) = w_pairs[k] * d(eff_rad_pair_k)/d(pair_dist_k) = w_pairs[k] * 0.5
        grad_pair_dist = 0.5 * w_pairs
        
        # d(obj)/d(bound_dist_i) = w_bounds[i] * 1.0
        grad_bound_dist = w_bounds
        
        # Backpropagate to points
        # grad_pts[i] = sum_{j: i<j} grad_pair_dist[k] * (p_i - p_j)/d_ij 
        #             + sum_{j: j<i} grad_pair_dist[k] * (p_i - p_j)/d_ij  (where k is the index for pair j,i)
        #             + grad_bound_dist[i] * (-p_i / ||p_i||)
        
        # We can compute the pairwise gradient contribution using the vectors.
        # For each pair (i,j), the contribution to grad p_i is grad_pair_dist[k] * (p_i - p_j)/d_ij
        # and to grad p_j is grad_pair_dist[k] * (p_j - p_i)/d_ij
        
        grad_pts = np.zeros_like(pts)
        
        # Vectorized accumulation
        # vecs is (M, 2), pair_dists is (M,)
        # unit_vecs = vecs / pair_dists[:, np.newaxis]
        # contrib = grad_pair_dist[:, np.newaxis] * unit_vecs
        # For each pair k with indices (i, j):
        # grad_pts[i] += contrib[k]
        # grad_pts[j] -= contrib[k]
        
        unit_vecs = vecs / pair_dists[:, np.newaxis]
        contrib = grad_pair_dist[:, np.newaxis] * unit_vecs
        
        # Accumulate into grad_pts
        # np.add.at is slow for large N, but N=300 is small enough?
        # M ~ 45000. np.add.at might be okay.
        # Alternative: loop over pairs? 45000 iterations in Python is slow.
        # Let's use a sparse matrix or just the loop if it's fast enough?
        # Actually, we can use the matrix formulation if we construct the full W matrix, but that's N^2 memory.
        # N=300 -> 90k entries. That's fine.
        
        # Let's try the matrix approach for speed.
        # W[i, j] = grad_pair_dist[k] / d_ij if i < j, else 0.
        # Then grad_pts[i] = sum_j W[i, j] (p_i - p_j) + sum_j W[j, i] (p_i - p_j) ? No.
        # The force on i from j is proportional to (p_i - p_j).
        # Let W[i, j] be the weight for the interaction.
        # grad p_i = sum_j W[i, j] (p_i - p_j)
        # If we set W[i, j] = grad_pair_dist[k]/d_ij for i<j and W[j, i] = grad_pair_dist[k]/d_ij for i<j,
        # then W is symmetric.
        # grad p_i = sum_j W[i, j] p_i - sum_j W[i, j] p_j
        #          = p_i (sum_j W[i, j]) - (W @ p)_i
        
        W = np.zeros((N, N))
        # Fill upper triangle
        W[iu[0], iu[1]] = grad_pair_dist / pair_dists
        W[iu[1], iu[0]] = grad_pair_dist / pair_dists
        
        row_sums = np.sum(W, axis=1)
        grad_pts += row_sums[:, np.newaxis] * pts - (W @ pts)
        
        # Boundary Gradient
        # d(bound_dist_i)/d(p_i) = -p_i / ||p_i||
        # grad_boundary_i = grad_bound_dist[i] * (-p_i / ||p_i||)
        
        dists_safe = np.where(dists_to_center < 1e-9, 1e-9, dists_to_center)
        grad_boundary = -grad_bound_dist[:, np.newaxis] * pts / dists_safe[:, np.newaxis]
        
        total_grad = grad_pts + grad_boundary
        flat_grad = total_grad.flatten()
        
        return obj, flat_grad

    # Optimization Loop with Annealing
    # Start with low beta, increase to high beta.
    # Using a schedule that spends more time at higher betas for refinement.
    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Define objective and gradient for current beta
        def obj_func(x):
            return objective_and_grad(x, beta)[0]
        
        def jac_func(x):
            return objective_and_grad(x, beta)[1]
        
        # Use L-BFGS-B
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                obj_func,
                current_points.flatten(),
                jac=jac_func,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.success or res.fun < objective_and_grad(current_points.flatten(), beta)[0]:
                current_points = res.x.reshape(N, 2)
            else:
                # If it didn't improve, keep old points
                pass
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Final Score Calculation
    # Ensure points are strictly inside
    dists_to_center = np.sqrt(current_points[:, 0] ** 2 + current_points[:, 1] ** 2)
    mask_out = dists_to_center > 1.0
    if np.any(mask_out):
        current_points[mask_out] /= dists_to_center[mask_out, np.newaxis]
        current_points[mask_out] *= 0.9999

    d = pdist(current_points)
    min_pair = np.min(d)
    min_bound = np.min(1.0 - dists_to_center)
    final_r = min(min_pair / 2.0, min_bound)
    
    return current_points

if __name__ == "__main__":
    construct()
