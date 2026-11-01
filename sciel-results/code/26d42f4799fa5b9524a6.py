"""Rewritten program using differentiable sorting of pairwise distances with soft-min penalty."""
import numpy as np
from scipy.optimize import minimize, Bounds

N = 50


def construct():
    """
    Construct centers of 50 equal spheres inside the unit cube [-1/2, 1/2]^3
    maximizing the minimum clearance (radius).
    
    Uses a differentiable optimization approach where we minimize a smooth
    approximation of the minimum pairwise distance and wall distance.
    """
    
    def compute_objective(x, temperature=0.01):
        """
        x: flattened array of shape (3*N,) containing sphere centers.
        Returns negative of the soft-minimum clearance (to maximize).
        
        The soft-min is computed as:
        softmin(d_i) ≈ -1/T * log(sum(exp(-T * d_i)))
        which approximates min(d_i) as T -> 0.
        """
        centers = x.reshape(-1, 3)
        
        # Pairwise distances
        # Compute all pairwise distances
        # For N=50, this is 1225 pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = centers[i_idx] - centers[j_idx]
        pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        # Wall distances: distance from each center to nearest wall
        # Cube is [-0.5, 0.5]^3, so distance to wall for coordinate k is 0.5 - |c_k|
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft-min approximation
        # softmin = -1/T * log(sum(exp(-T * d)))
        # To avoid overflow, use log-sum-exp trick
        min_d = np.min(all_dists)
        scaled = -temperature * (all_dists - min_d)
        log_sum = np.log(np.sum(np.exp(scaled))) + (-temperature * min_d)
        softmin = -log_sum / temperature
        
        # We want to maximize softmin, so minimize -softmin
        return -softmin
    
    def compute_gradient(x, temperature=0.01):
        """
        Compute gradient of the objective function.
        """
        centers = x.reshape(-1, 3)
        
        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = centers[i_idx] - centers[j_idx]
        pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1) + 1e-12)
        
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(centers), axis=1)
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft-min weights: w_i = exp(-T * d_i) / sum(exp(-T * d_j))
        min_d = np.min(all_dists)
        scaled = -temperature * (all_dists - min_d)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / np.sum(exp_scaled)
        
        # Gradient of softmin w.r.t. each distance
        # d(softmin)/d(d_i) = -T * w_i
        # So d(-softmin)/d(d_i) = T * w_i
        
        # Gradient w.r.t. centers for pairwise distances
        grad_centers = np.zeros_like(centers)
        
        # For each pair (i, j), the contribution to gradient
        # d(pair_dist)/d(centers[i]) = -(centers[i] - centers[j]) / pair_dist
        # d(pair_dist)/d(centers[j]) = (centers[i] - centers[j]) / pair_dist
        
        for idx in range(len(i_idx)):
            i = i_idx[idx]
            j = j_idx[idx]
            d = pair_dists[idx]
            if d < 1e-10:
                continue
            weight = temperature * weights[idx]
            diff_vec = diffs[idx] / d
            grad_centers[i] -= weight * diff_vec
            grad_centers[j] += weight * diff_vec
        
        # Gradient w.r.t. centers for wall distances
        for i in range(N):
            weight = temperature * weights[N * (N - 1) // 2 + i]
            # Wall distance = min(0.5 - |c_k|) over k
            # Find which coordinate gives the minimum
            coords = 0.5 - np.abs(centers[i])
            min_k = np.argmin(coords)
            # d(0.5 - |c_k|)/d(c_k) = -sign(c_k)
            sign_val = np.sign(centers[i, min_k])
            if sign_val == 0:
                sign_val = 1.0
            grad_centers[i, min_k] -= weight * sign_val
        
        return grad_centers.flatten()
    
    # Initialize with a good starting point
    # Use a simple grid as initial guess, then optimize
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # Place points in a grid-like pattern with some randomness
    np.random.seed(42)
    
    # Create a better initial layout: use a face-centered cubic-like arrangement
    # Or start with a perturbed grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    # Add small random perturbation to break symmetry
    x0 = x0 + 0.01 * np.random.randn(3 * N)
    
    # Clamp to valid range
    x0 = np.clip(x0, -0.45, 0.45)
    
    # Optimize
    result = minimize(
        compute_objective,
        x0,
        method='L-BFGS-B',
        jac=compute_gradient,
        bounds=[(-0.5, 0.5)] * (3 * N),
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    centers = result.x.reshape(-1, 3)
    
    # Ensure centers are strictly inside the cube
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers


if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    
    # Compute the actual radius
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = centers[i_idx] - centers[j_idx]
    pair_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    min_pair = np.min(pair_dists) / 2
    min_wall = np.min(wall_dists)
    radius = min(min_pair, min_wall)
    print(f"Radius: {radius}")
