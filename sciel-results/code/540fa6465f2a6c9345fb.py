import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Smooth objective function: minimize -logsumexp(-beta * distances)
    # This approximates maximizing the minimum distance.
    beta = 100.0
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        
        # Pairwise distances
        # diff[i, j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero in gradient, add small epsilon
        dists = np.sqrt(dists_sq + 1e-12)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dists[mask]
        
        # Wall distances
        # Distance to nearest wall for each point
        # x in [-0.5, 0.5]
        # dist to wall = min(x - (-0.5), 0.5 - x) for each coord
        # min_k (0.5 - |x_k|)
        wall_dists = 0.5 - np.abs(x)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, min_wall_dists])
        
        # Log-sum-exp
        vals = -beta * all_dists
        max_val = np.max(vals)
        log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        
        obj = -log_sum_exp / beta
        
        # Gradient
        exp_vals = np.exp(vals - max_val)
        S = np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        # w_ij = exp(-beta * d_ij) / S
        # grad_i += w_ij * (x_i - x_j) / d_ij
        # grad_j += w_ij * (x_j - x_i) / d_ij
        
        # We need to map the flat exp_vals back to the (N, N) matrix structure for pairs
        # The mask was row-major flattening of the upper triangle? No, mask is full N x N with diagonal False.
        # So pair_exp is length N*(N-1).
        
        # It is more efficient to compute the weighted sum directly using the matrix forms.
        # W_ij = exp(-beta * d_ij) / S for i != j, 0 otherwise.
        
        # Reconstruct W from pair_exp?
        # pair_exp corresponds to dists[mask].
        # Let's create a matrix W
        W = np.zeros((N, N))
        W[mask] = exp_vals[:N*(N-1)] / S
        
        # unit vectors
        unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-12)
        
        # grad_pair_i = sum_j W_ij * unit_vecs_ij
        grad_pair = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall gradient
        wall_exp = exp_vals[N*(N-1):]
        weights_wall = wall_exp / S
        
        grad_wall = np.zeros_like(x)
        # For each point i, the min wall distance is in dimension min_dims[i]
        min_dims = np.argmin(wall_dists, axis=1)
        
        for i in range(N):
            k = min_dims[i]
            # d(0.5 - |x_ik|)/dx_ik = -sign(x_ik)
            # If x_ik is 0, subgradient is 0 or +/-1. We use 0 to be safe, or random.
            # Using sign is standard.
            grad_wall[i, k] = weights_wall[i] * (-np.sign(x[i, k]))
            
        grad = grad_pair + grad_wall
        
        return obj, grad

    def calc_true_radius(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair / 2.0, min_wall)

    def hilbert_curve_3d(n, side_length=1.0, offset=-0.5):
        """
        Generate n points on a 3D Hilbert curve within a cube.
        This is a simple recursive implementation.
        """
        # For simplicity and speed, we can use a space-filling curve approximation
        # or just a jittered grid which is often sufficient.
        # Let's use a jittered grid for robustness and speed.
        pass

    def generate_initial_points(seed):
        np.random.seed(seed)
        # Jittered grid initialization
        # 5x5x5 = 125 points. We need 50.
        # We can pick a subset or just use random points with min distance.
        # Let's try random points with a min distance constraint (Poisson-like)
        # Or simply a scaled jittered grid of 4x4x4 = 64 points, pick 50.
        
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        # Add jitter
        jitter = np.random.normal(0, 0.05, grid.shape)
        points = grid + jitter
        
        # Clamp to cube
        points = np.clip(points, -0.45, 0.45)
        
        # If we have more than N, shuffle and take first N
        if len(points) > N:
            indices = np.random.choice(len(points), N, replace=False)
            points = points[indices]
        else:
            # If fewer, add random points
            while len(points) < N:
                p = np.random.uniform(-0.45, 0.45, 3)
                points = np.vstack([points, p])
        
        return points

    best_x = None
    best_score = -np.inf
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Run multiple restarts
    for seed in range(20):
        if time.time() > deadline:
            break
            
        x_init = generate_initial_points(seed)
        
        # Optimize
        try:
            res = minimize(
                objective_and_grad,
                x_init.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            x_opt = res.x.reshape(N, 3)
        except Exception:
            continue
            
        score = calc_true_radius(x_opt)
        
        if score > best_score:
            best_score = score
            best_x = x_opt
            
    if best_x is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
