import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initialize with hexagonal lattice
    def init_hex():
        # Approximate radius for hex packing of 50 circles in unit circle
        # r ~ 0.12
        # Spacing ~ 2r
        # Let's generate a large hex grid and cut it
        spacing = 0.25 # Initial guess, will be adjusted by optimizer
        points = []
        # Generate points in a box large enough
        range_x = np.arange(-1.5, 1.5, spacing)
        range_y = np.arange(-1.5, 1.5, spacing * np.sqrt(3) / 2)
        
        for y in range_y:
            offset = 0 if int(y / (spacing * np.sqrt(3) / 2)) % 2 == 0 else spacing / 2
            for x in range_x:
                px = x + offset
                py = y
                if px**2 + py**2 <= 1.0:
                    points.append([px, py])
        
        points = np.array(points)
        if len(points) < N:
            # If not enough, reduce spacing
            # This simple loop might not be robust, but for N=50 it should be fine with small spacing
            # Let's just use a fixed dense grid if this fails
            pass
            
        # Select N points closest to center? Or random?
        # Closest to center tends to be more symmetric, which is good for packing.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        return points[indices[:N]]
    
    X = init_hex().flatten()
    
    # Objective function and gradient
    def objective_and_grad(X_flat, beta):
        X = X_flat.reshape(-1, 2)
        N = X.shape[0]
        
        # Compute all pairwise distances
        # d_ij = ||X_i - X_j||
        # We need sum of exp(-beta * d_ij) for i<j
        # and sum of exp(-beta * (1 - ||X_i||)) for i
        
        # Vectorized calculation for pairs is tricky to do without O(N^2) memory, but N=50 is small.
        # We can compute the full distance matrix.
        
        # Pairwise distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq + 1e-12) # Avoid div by zero
        
        # Mask upper triangle for pairs
        mask = np.triu(np.ones((N, N)), k=1).astype(bool)
        pair_dists = dist[mask]
        
        # Boundary distances
        radii = np.linalg.norm(X, axis=1)
        bound_dists = 1.0 - radii
        
        # Calculate A = sum exp(-beta * dist)
        # To avoid underflow/overflow, we can subtract the min, but for optimization we just need relative values.
        # With beta up to 100 and dist ~ 0.1, exp(-10) is small but not zero.
        
        pair_terms = np.exp(-beta * pair_dists)
        bound_terms = np.exp(-beta * bound_dists)
        
        A = np.sum(pair_terms) + np.sum(bound_terms)
        
        # Objective: (1/beta) * log(A)
        val = (1.0 / beta) * np.log(A + 1e-12)
        
        # Gradient
        # dA/dX_k = beta * ( sum_{j!=k} exp(-beta d_jk) * (X_k - X_j)/d_jk + exp(-beta (1-d_k)) * X_k / d_k )
        # dL/dX_k = (1/A) * ( sum ... )
        
        grad = np.zeros_like(X)
        
        # Pairwise part
        # For each k, sum over j != k
        # We can vectorize this.
        # term_jk = exp(-beta * d_jk) * (X_k - X_j) / d_jk
        # Note: d_jk is symmetric.
        # We computed dist matrix.
        # We need to sum over j for each k.
        
        # Create a matrix of vectors (X_k - X_j) / d_jk
        # diff is (X_i - X_j). So (X_k - X_j) is diff[k, j].
        # dist is dist[k, j].
        
        # Avoid division by zero on diagonal
        inv_dist = np.zeros_like(dist)
        np.fill_diagonal(inv_dist, 0.0)
        inv_dist[~np.isfinite(inv_dist)] = 0.0
        # Actually, just divide where dist > 0
        mask_non_zero = dist > 1e-6
        inv_dist[mask_non_zero] = 1.0 / dist[mask_non_zero]
        
        # Weight matrix W_jk = exp(-beta * d_jk) * inv_dist_jk
        W = np.exp(-beta * dist) * inv_dist
        np.fill_diagonal(W, 0.0)
        
        # grad_pair_k = sum_j W_kj * diff[k, j]
        # diff[k, j] = X_k - X_j
        # This is a matrix multiplication.
        # grad_pair = W @ diff ?
        # (W @ diff)[k] = sum_j W[k, j] * diff[k, j]
        # Yes.
        grad_pair = W @ diff
        
        # Boundary part
        # grad_bound_k = exp(-beta * (1 - r_k)) * X_k / r_k
        inv_radii = np.zeros_like(radii)
        mask_r = radii > 1e-6
        inv_radii[mask_r] = 1.0 / radii[mask_r]
        
        bound_weights = np.exp(-beta * bound_dists) * inv_radii
        grad_bound = bound_weights[:, np.newaxis] * X
        
        total_grad = grad_pair + grad_bound
        
        # Scale by 1/A
        grad = total_grad / A
        
        return val, grad.flatten()

    # Optimization loop
    best_val = np.inf
    best_X = X.copy()
    
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                X,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_X = res.x
                X = res.x
        except Exception:
            pass

    # Final polish: Calculate the actual score and return
    # The score is min( min_pair_dist/2, min_bound_dist )
    X_final = best_X.reshape(-1, 2)
    
    # One more small local step to ensure we are at a local max of the true objective
    # We can do a few iterations of: find min clearance, move that point slightly towards center or away from neighbors.
    # But the smooth optimization with high beta is usually very close.
    
    # Ensure validity
    # Check if any point is outside
    radii = np.linalg.norm(X_final, axis=1)
    # If any radius > 1, pull it back
    out_mask = radii > 1.0
    if np.any(out_mask):
        X_final[out_mask] *= (1.0 / radii[out_mask])[:, np.newaxis] * 0.9999
        
    return X_final

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def init_hex():
        # Generate a dense hex grid
        spacing = 0.2
        points = []
        # Range needs to cover unit circle
        range_x = np.arange(-1.5, 1.5, spacing)
        range_y = np.arange(-1.5, 1.5, spacing * np.sqrt(3) / 2)
        
        for y in range_y:
            # Determine offset
            row_idx = int((y + 1.5) / (spacing * np.sqrt(3) / 2))
            offset = 0 if row_idx % 2 == 0 else spacing / 2
            for x in range_x:
                px = x + offset
                py = y
                if px**2 + py**2 <= 1.0:
                    points.append([px, py])
        
        points = np.array(points)
        if len(points) < N:
             # Fallback: random
             return np.random.rand(N, 2) * 1.4 - 0.7
             
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        return points[indices[:N]]
    
    X = init_hex().flatten()
    
    def objective_and_grad(X_flat, beta):
        X = X_flat.reshape(-1, 2)
        N = X.shape[0]
        
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq + 1e-12)
        
        mask = np.triu(np.ones((N, N)), k=1).astype(bool)
        pair_dists = dist[mask]
        
        radii = np.linalg.norm(X, axis=1)
        bound_dists = 1.0 - radii
        
        pair_terms = np.exp(-beta * pair_dists)
        bound_terms = np.exp(-beta * bound_dists)
        
        A = np.sum(pair_terms) + np.sum(bound_terms)
        
        val = (1.0 / beta) * np.log(A + 1e-12)
        
        inv_dist = np.zeros_like(dist)
        mask_non_zero = dist > 1e-6
        inv_dist[mask_non_zero] = 1.0 / dist[mask_non_zero]
        
        W = np.exp(-beta * dist) * inv_dist
        np.fill_diagonal(W, 0.0)
        
        grad_pair = W @ diff
        
        inv_radii = np.zeros_like(radii)
        mask_r = radii > 1e-6
        inv_radii[mask_r] = 1.0 / radii[mask_r]
        
        bound_weights = np.exp(-beta * bound_dists) * inv_radii
        grad_bound = bound_weights[:, np.newaxis] * X
        
        total_grad = grad_pair + grad_bound
        grad = total_grad / A
        
        return val, grad.flatten()

    best_val = np.inf
    best_X = X.copy()
    
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                X,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_X = res.x
                X = res.x
        except Exception:
            pass

    X_final = best_X.reshape(-1, 2)
    
    # Final check and correction
    radii = np.linalg.norm(X_final, axis=1)
    out_mask = radii > 1.0
    if np.any(out_mask):
        X_final[out_mask] *= (1.0 / radii[out_mask])[:, np.newaxis] * 0.99999
        
    return X_final
# EVOLVE-BLOCK-END
