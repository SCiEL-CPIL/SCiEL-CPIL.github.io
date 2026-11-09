"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers), axis=(0, 1))
    
    # Minimum pairwise distance / 2
    # Using cKDTree for efficiency, but for N=50 a direct loop is also fast and safer for exactness
    min_pair_dist = np.inf
    # Vectorized pairwise distance calculation for N=50 is very fast
    # d[i, j] = ||c_i - c_j||
    # We need min over i < j
    # Let's use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_dist_to_wall, min_pair_dist / 2.0)

def _fibonacci_lattice(n):
    """
    Generate n points on a unit sphere using the Fibonacci lattice algorithm,
    then scale to fit inside the cube [-0.5, 0.5]^3.
    This provides a highly uniform distribution on the sphere surface, 
    which, when mapped to a volume or used as a starting point for volume packing,
    offers good initial separation.
    """
    # Fibonacci sphere points
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    z = np.linspace(-1, 1, n, endpoint=False) + 0.5 / n
    r = np.sqrt(1.0 - z * z)
    theta = phi * np.arange(n)
    
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    
    # These are on unit sphere. 
    # To use as initial guess for a cube packing, we can just scale them down 
    # to fit inside the cube with some margin, or map them into the cube volume.
    # A common trick is to just use the normalized coordinates scaled to fit the cube.
    # Since we are maximizing radius, starting points near the center but spread out is good.
    # Let's map them to the cube [-0.5, 0.5]^3.
    # The points (x,y,z) are in [-1,1]. So x/2, y/2, z/2 are in [-0.5, 0.5].
    points = np.column_stack((x, y, z)) * 0.5
    
    # To ensure they are strictly inside and not on the boundary (which might cause issues with gradients),
    # we can scale slightly or just trust the optimizer. 
    # However, for a "lattice" in a cube, a simple grid is often better for interior points.
    # But the directive specifically asks for "Fibonacci lattice". 
    # Standard Fibonacci lattice is on a sphere. 
    # Perhaps the intent is to use the 3D Fibonacci lattice (spiral in a box)?
    # Or simply use the spherical distribution scaled to the box.
    # Let's stick to the spherical distribution scaled to the box, as it's the most standard "Fibonacci lattice" interpretation.
    
    return points

def _objective_and_grad(x):
    """
    Objective function to minimize: a soft measure of the minimum distance.
    We want to MAXIMIZE the minimum radius.
    Radius r = min( min_pair_dist/2, min_wall_dist )
    
    Instead of maximizing r, we minimize a penalty function that is large when r is small.
    A common approach is to minimize the sum of squared violations or a log-sum-exp of distances.
    
    Let d_ij be pairwise distances. Let w_i be wall distances.
    We want to maximize min( min_ij d_ij / 2, min_i w_i ).
    
    Equivalent to minimizing:
    f(x) = -log( sum_ij exp(-beta * d_ij) + sum_i exp(-beta * w_i) ) / beta  ?
    Or simpler: minimize the sum of squared "shortfalls" from a target radius?
    
    Let's use a smooth approximation of the minimum.
    min(a, b) approx -1/beta * log( exp(-beta a) + exp(-beta b) )
    
    Here a = min_pair_dist/2 and b = min_wall_dist.
    This is complex because of the nested mins.
    
    Alternative: Minimize the sum of squared inverse distances?
    
    Let's try: 
    Objective = sum_{i<j} 1/(d_ij^2 + eps) + sum_i 1/(w_i^2 + eps)
    This penalizes small distances heavily. Minimizing this encourages all distances to be large.
    It's a repulsive force model, but differentiable and smooth.
    
    Gradient:
    d/dx_i ( 1/(||x_i - x_j||^2 + eps) ) = -1/(d^2+eps)^2 * 2 (x_i - x_j)
    d/dx_i ( 1/(w_i^2 + eps) ) ... w_i = 0.5 - |x_ik|.
    This has non-differentiable points at x_ik=0. 
    To avoid this, we can use squared wall distances or just assume x_ik > 0 for the gradient 
    or use a smooth approximation for the absolute value.
    
    Let's use a simpler, robust objective:
    Minimize J = - ( min_pair_dist / 2 + min_wall_dist ) is non-smooth.
    
    Let's use the "soft-min" approach on the set of all relevant distances.
    Distances D = [ d_ij/2 for i<j ] U [ w_i for all i,k ]
    We want to maximize min(D).
    Approximate min(D) by -1/beta * log( sum_{d in D} exp(-beta * d) ).
    Let M = sum exp(-beta * d).
    f = -1/beta * log(M).
    grad f = -1/beta * (1/M) * sum ( d/dx exp(-beta * d) )
    d/dx exp(-beta * d) = exp(-beta * d) * (-beta) * d(d)/dx
    So grad f = 1/M * sum ( exp(-beta * d) * d(d)/dx )
    
    This is a weighted average of the gradients of the individual distances, weighted by their "closeness" to the minimum.
    This is a very standard and effective method for max-min problems.
    
    We need d(d)/dx for:
    1. d_ij = ||x_i - x_j||. d/dx_i d_ij = (x_i - x_j) / d_ij.
    2. w_i = 0.5 - |x_ik|. d/dx_ik w_i = -sign(x_ik).
       To make it smooth, we can use sign(x) approx x/sqrt(x^2+eps) or just handle the sign carefully.
       Since we start from a Fibonacci lattice, points are unlikely to be exactly 0.
    """
    centers = x.reshape(N, 3)
    beta = 50.0  # Sharpness of the soft-min. Higher means closer to true min.
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers) # shape (N, 3)
    
    # Combine all distances into a single list for the soft-min
    # We need to be careful with memory and efficiency.
    # Number of pairwise distances: N*(N-1)/2 = 50*49/2 = 1225
    # Number of wall distances: N*3 = 150
    # Total ~1375 terms. This is small.
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Flatten wall distances
    wall_dists_flat = wall_dists.flatten()
    
    # All distances
    all_dists = np.concatenate([pair_dists, wall_dists_flat])
    
    # Avoid log(0) or exp(overflow)
    # Clip distances to a small positive value
    all_dists = np.maximum(all_dists, 1e-6)
    
    # Compute weights for soft-min
    # w_k = exp(-beta * d_k)
    weights = np.exp(-beta * all_dists)
    sum_weights = np.sum(weights)
    
    if sum_weights < 1e-300:
        # Fallback if underflow
        min_dist = np.min(all_dists)
        return -min_dist, np.zeros_like(x)
        
    # Objective: -1/beta * log(sum_weights)
    obj = -1.0 / beta * np.log(sum_weights)
    
    # Gradient
    # grad_obj = 1/sum_weights * sum_k ( weights[k] * grad_d_k )
    # Note: The formula derived was:
    # grad f = 1/M * sum ( exp(-beta d) * grad d )
    # Wait, let's re-verify.
    # f = -1/beta * log( sum e^{-beta d_k} )
    # df/dx = -1/beta * (1/S) * sum( e^{-beta d_k} * (-beta) * dd_k/dx )
    # df/dx = (1/S) * sum( e^{-beta d_k} * dd_k/dx )
    # Yes, correct.
    
    grad = np.zeros_like(x)
    
    # Pairwise gradients
    # For each pair (i, j), dist = d_ij.
    # d/dx_i d_ij = (x_i - x_j) / d_ij
    # d/dx_j d_ij = (x_j - x_i) / d_ij = - (x_i - x_j) / d_ij
    
    # We need to accumulate:
    # grad[x_i] += weights_pair[k] * (x_i - x_j) / d_ij
    # grad[x_j] -= weights_pair[k] * (x_i - x_j) / d_ij
    
    # weights for pairs are the first len(pair_dists) elements of 'weights'
    w_pairs = weights[:len(pair_dists)]
    
    # Vectorized accumulation is tricky with indices, but N is small.
    # Let's use a loop for clarity and correctness, it's fast enough for N=50.
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        d = pair_dists[k]
        w = w_pairs[k]
        
        if d < 1e-10:
            continue
            
        diff_vec = centers[i] - centers[j]
        grad_vec = w * diff_vec / d
        
        grad[i] += grad_vec
        grad[j] -= grad_vec
        
    # Wall gradients
    # w_i = 0.5 - |x_ik|
    # d/dx_ik w_i = -sign(x_ik)
    # weights for walls are the last N*3 elements
    w_walls = weights[len(pair_dists):].reshape(N, 3)
    
    # sign(x) is 1 if x>0, -1 if x<0.
    # To be smooth, we can use x / sqrt(x^2 + eps)
    # But simple sign is usually fine for BFGS if we avoid 0.
    signs = np.sign(centers)
    # Handle zeros
    signs[centers == 0] = 1.0
    
    grad_walls = -w_walls * signs
    grad += grad_walls.reshape(N, 3)
    
    # Scale gradient by 1/sum_weights
    grad /= sum_weights
    
    return obj, grad.flatten()

def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Initialize with Fibonacci lattice
    centers = _fibonacci_lattice(N)
    
    # Ensure they are inside the cube
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Optimize using BFGS
    # We minimize the soft-min objective.
    x0 = best_centers.flatten()
    
    # We might need multiple restarts or different beta values.
    # Let's try a single run first, then refine if time permits.
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='BFGS',
            jac=True, # We provide the gradient
            options={'gtol': 1e-6, 'maxiter': 500}
        )
        
        if result.success or result.fun < _objective_and_grad(x0)[0]:
            new_centers = result.x.reshape(N, 3)
            # Ensure validity
            if np.all(np.isfinite(new_centers)):
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_centers = new_centers
                    best_radius = new_radius
    except Exception:
        pass
        
    # 3. If time remains, try a second phase with higher beta (sharper min) 
    # or a different initial guess (e.g., slightly perturbed Fibonacci)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try a perturbed version to escape local minima
        np.random.seed(42)
        perturbed = best_centers + np.random.normal(0, 0.01, (N, 3))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        try:
            result2 = minimize(
                _objective_and_grad,
                perturbed.flatten(),
                method='BFGS',
                jac=True,
                options={'gtol': 1e-7, 'maxiter': 500}
            )
            
            if result2.success:
                new_centers2 = result2.x.reshape(N, 3)
                if np.all(np.isfinite(new_centers2)):
                    new_radius2 = _compute_radius(new_centers2)
                    if new_radius2 > best_radius:
                        best_centers = new_centers2
                        best_radius = new_radius2
        except Exception:
            pass

    # Final check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback
        return _fibonacci_lattice(N)
        
    return best_centers

# EVOLVE-BLOCK-END
