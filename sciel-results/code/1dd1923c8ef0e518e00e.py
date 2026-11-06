import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius for the given points.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    # Pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def soft_min_radius_and_grad(points, beta):
    """
    Computes the soft-min radius and its gradient.
    We define the "radius" as the soft-min of all constraint distances.
    Constraints:
    1. Pairwise distances / 2: d_ij / 2
    2. Boundary distances: 1 - ||x_i||
    
    We want to maximize this radius.
    soft_min = -1/beta * log(sum(exp(-beta * f_k)))
    
    Gradient of soft_min w.r.t points:
    grad = (1/sum) * sum( exp(-beta * f_k) * grad_f_k )
    """
    points = np.asarray(points)
    N = len(points)
    
    # 1. Compute pairwise distances and their gradients
    # diff[i,j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We only care about upper triangle for pairs to avoid double counting in the sum?
    # The soft-min is over the set of all distinct pairs and boundaries.
    # It doesn't matter if we count each pair twice in the sum, it just changes the effective beta slightly.
    # But for correctness of the "min" interpretation, it's fine.
    
    # Values f_k for pairs
    f_pairs = dists / 2.0
    
    # Values f_k for boundaries
    norms = np.linalg.norm(points, axis=1)
    f_bound = 1.0 - norms
    
    # Combine all f values
    # We need to compute sum(exp(-beta * f)) and sum(exp(-beta * f) * grad_f)
    
    # To save memory and time, we can compute the exponential weights
    # exp_pairs = exp(-beta * f_pairs)
    # exp_bound = exp(-beta * f_bound)
    
    # However, exp(-beta * f) can underflow if beta * f is large.
    # We can shift by the minimum f to stabilize.
    # Let m = min(all f)
    # exp(-beta * (f - m)) = exp(-beta * f) * exp(beta * m)
    # Sum = exp(beta * m) * sum(exp(-beta * (f - m)))
    # soft_min = -1/beta * log(exp(beta * m) * S') = -1/beta * (beta * m + log(S')) = -m - 1/beta * log(S')
    
    # This is more stable.
    
    # Get all f values
    # Pairwise: N*(N-1)/2 values
    iu = np.triu_indices(N, k=1)
    f_pairs_vals = f_pairs[iu]
    
    # Boundary: N values
    f_bound_vals = f_bound
    
    # Combine for finding min
    all_f = np.concatenate([f_pairs_vals, f_bound_vals])
    m = np.min(all_f)
    
    # Compute shifted exponentials
    # For pairs
    # We need to accumulate the sum of exp(-beta * (f - m))
    # And the sum of exp(-beta * (f - m)) * grad_f
    
    # Gradients of f_pairs = dists / 2
    # d/dx_i (||x_i - x_j|| / 2) = (x_i - x_j) / (2 * ||x_i - x_j||)
    # grad_f_pairs[i] = sum_{j != i} (x_i - x_j) / (2 * d_ij)
    
    # Let's compute the weighted gradients directly.
    
    # Weights for pairs
    # w_pairs = exp(-beta * (f_pairs - m))
    # To avoid underflow, we compute this on the fly or in chunks.
    
    # For N=50, N*(N-1)/2 = 1225 pairs. This is small.
    
    # Calculate weights
    # f_pairs shape (N, N)
    # We only need upper triangle for the sum, but gradient accumulation is symmetric.
    # Let's compute the full matrix of weights for simplicity, then mask diagonal.
    
    # f_pairs is (N, N)
    # w_pairs = exp(-beta * (f_pairs - m))
    # Note: f_pairs[i,i] is inf, so exp(-inf) = 0. This is fine.
    
    w_pairs = np.exp(-beta * (f_pairs - m))
    np.fill_diagonal(w_pairs, 0.0)
    
    # Sum of weights for pairs
    S_pairs = np.sum(w_pairs)
    
    # Gradient contribution from pairs
    # grad_i = sum_j w_pairs[i,j] * (x_i - x_j) / (2 * d_ij)
    # Note: f_pairs[i,j] = d_ij / 2.
    # grad_f_pairs[i,j] = (x_i - x_j) / (2 * d_ij)
    
    # diff[i,j] = x_i - x_j
    # dists[i,j] = d_ij
    
    # Term = w_pairs[i,j] * (x_i - x_j) / (2 * d_ij)
    # We can vectorize this.
    # diff / (2 * dists)[:,:,None]
    
    # Handle dists=0 (should not happen if points are distinct, but safe)
    dists_safe = np.maximum(dists, 1e-12)
    np.fill_diagonal(dists_safe, 1.0) # Diagonal is masked in w_pairs anyway
    
    grad_terms = w_pairs[:, :, np.newaxis] * diff / (2.0 * dists_safe[:, :, np.newaxis])
    
    # Sum over j for each i
    grad_pairs = np.sum(grad_terms, axis=1)
    
    # 2. Boundary
    # f_bound = 1 - ||x_i||
    # grad f_bound_i = -x_i / ||x_i||
    
    w_bound = np.exp(-beta * (f_bound - m))
    S_bound = np.sum(w_bound)
    
    # grad_terms_bound = w_bound[:, None] * (-x_i / ||x_i||)
    norms_safe = np.maximum(norms, 1e-12)
    grad_bound = w_bound[:, np.newaxis] * (-points / norms_safe[:, np.newaxis])
    
    # Total Sum S = S_pairs + S_bound
    S = S_pairs + S_bound
    
    # Soft min value
    # soft_min = -m - (1/beta) * log(S)
    soft_min_val = -m - (1.0 / beta) * np.log(S)
    
    # Total Gradient
    # grad_soft_min = (1/S) * (grad_pairs_sum + grad_bound_sum)
    # Wait, the derivation:
    # soft_min = -1/beta log S
    # d(soft_min)/dx = -1/beta * (1/S) * dS/dx
    # dS/dx = sum_k exp(-beta(f_k - m)) * (-beta) * df_k/dx * exp(beta*m)? No.
    
    # Let's re-derive carefully.
    # S = sum_k exp(-beta (f_k - m))
    # soft_min = -m - 1/beta log S
    # d(soft_min)/dx = -1/beta * (1/S) * dS/dx
    # dS/dx = sum_k d/dx [ exp(-beta (f_k - m)) ]
    # = sum_k exp(-beta (f_k - m)) * (-beta) * df_k/dx
    # = -beta * sum_k w_k * df_k/dx
    # So d(soft_min)/dx = -1/beta * (1/S) * (-beta) * sum_k w_k * df_k/dx
    # = (1/S) * sum_k w_k * df_k/dx
    
    # So yes, grad_soft_min = (1/S) * (grad_pairs + grad_bound)
    
    grad_soft_min = (grad_pairs + grad_bound) / S
    
    return soft_min_val, grad_soft_min

def optimize_from_seed(points, time_limit):
    """
    Optimizes the packing starting from a seed using soft-min gradient descent.
    """
    start_time = time.time()
    current_points = points.copy()
    
    # Annealing schedule for beta
    # Start with small beta (smooth), increase to large beta (sharp)
    beta_start = 10.0
    beta_end = 1000.0
    num_steps = 50 # Number of beta updates
    
    # We will use L-BFGS-B or BFGS for each beta step?
    # Or just run one long optimization with increasing beta?
    # Simpler: Run a loop. In each iteration, run minimize with current beta.
    # Then increase beta.
    
    best_score = compute_radius(current_points)
    best_points = current_points.copy()
    
    # We'll use a fixed number of iterations per beta, or until convergence.
    # Given time limit, we should be adaptive.
    
    for i in range(num_steps):
        if time.time() > time_limit:
            break
            
        # Interpolate beta
        # Linear in log space?
        beta = beta_start * (beta_end / beta_start) ** (i / (num_steps - 1))
        
        # Define objective and gradient for this beta
        def obj(x):
            val, grad = soft_min_radius_and_grad(x.reshape(N, 2), beta)
            # We want to maximize val, so minimize -val
            return -val, -grad.flatten()
        
        # Use L-BFGS-B
        # No bounds, but we can add a penalty if points go far outside?
        # The soft-min will naturally keep them inside if beta is high enough,
        # because boundary terms become dominant if they go out.
        
        try:
            res = minimize(obj, current_points.flatten(), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 100, 'gtol': 1e-6})
            current_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
        # Check score
        score = compute_radius(current_points)
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
    return best_points, best_score

def generate_seeds():
    seeds = []
    
    # 1. Hexagonal
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0], radius[radius <= 1.0]
        if len(points) >= N:
            # Pick N points closest to center? Or just first N?
            # Sorting by radius ensures we get a dense core.
            hex_seed = points[np.argsort(radius, kind="stable")[:N]]
            seeds.append(hex_seed)
            break
        spacing *= 0.98
        
    # 2. Rings
    # 1 center, 6, 12, 18, 13
    ring_radii = [0, 0.3, 0.6, 0.85, 1.0]
    ring_counts = [1, 6, 12, 18, 13]
    pts = []
    for r, count in zip(ring_radii, ring_counts):
        if count == 0: continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    ring_seed = np.array(pts[:N])
    if len(ring_seed) == N:
        seeds.append(ring_seed)
        
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(pts)
        
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    seeds = generate_seeds()
    
    best_points = None
    best_score = -1.0
    
    # Allocate time to each seed
    # If we have 7 seeds, ~15s each.
    # Or just run them sequentially until time is up.
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Give each seed a portion of the remaining time
        # Simple strategy: run optimization for each seed with a time limit
        # If we have 7 seeds and 110s, ~15s each.
        # But later seeds might be better.
        # Let's just run them.
        
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Cap time per seed to ensure we try multiple
        # If many seeds, give less time.
        seed_time = min(20.0, time_left / (len(seeds)))
        
        try:
            optimized_points, score = optimize_from_seed(seed, seed_time)
            if score > best_score:
                best_score = score
                best_points = optimized_points
        except Exception as e:
            pass
            
    # Final polish: Run one more optimization on the best found points with a longer time budget?
    # If we have time left, refine the best candidate further.
    if best_points is not None and time.time() < deadline - 5.0:
        time_left = deadline - time.time() - 2.0
        # Increase beta range or just run more iterations?
        # We can call optimize_from_seed again with the best points as seed
        # and a larger time limit.
        try:
            refined_points, score = optimize_from_seed(best_points, time_left)
            if score > best_score:
                best_score = score
                best_points = refined_points
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        best_points = seeds[0]
        
    return best_points

if __name__ == "__main__":
    # Quick test
    points = construct()
    r = compute_radius(points)
    print(f"Radius: {r}")
