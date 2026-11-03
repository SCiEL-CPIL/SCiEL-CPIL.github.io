"""
Packing 100 equal spheres in a unit cube using Stochastic Gradient Langevin Dynamics (SGLD).

Strategy:
1. Initialize with a structured grid.
2. Define a smooth objective function: -soft_min(min_pair_dist/2, min_wall_margin).
   The soft_min uses a log-sum-exp formulation to approximate the minimum smoothly.
3. Use SGLD to optimize the positions:
   x_{t+1} = x_t + alpha * grad(score) + sqrt(2 * alpha * T) * noise
   - Grad is computed analytically or via finite differences (analytical preferred for speed).
   - Temperature T is annealed over time to allow exploration then refinement.
   - Alpha is the step size.
4. The "score" is the radius r. We want to maximize it.
   To make it differentiable, we use:
   r_soft = -1/beta * log( sum_{constraints} exp(-b_i * beta) )
   where b_i are the constraints (dist/2, margins).
   As beta -> infinity, r_soft -> min(b_i).
   We optimize -r_soft.
5. We periodically project points back into the cube to maintain validity.
6. We keep track of the best valid configuration found (maximizing the true min radius).

Why SGLD?
- The packing problem has many local minima.
- Gradient descent (L-BFGS, SLSQP) gets stuck easily.
- SGLD adds noise proportional to temperature, allowing jumps over energy barriers.
- The smooth objective ensures the gradient is well-defined everywhere.
"""

import numpy as np
import time
import random

N = 100

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    # Efficient computation of min pairwise distance
    # For N=100, O(N^2) is fine (10k ops)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _grid_init(nx=5, ny=5, nz=4):
    """Generate a grid-based initialisation."""
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def _soft_min_radius(points, beta):
    """
    Compute the soft-min radius and its gradient.
    
    Constraints:
    1. Pairwise: d_ij / 2 >= r  =>  b_ij = d_ij / 2
    2. Wall: 0.5 - |p_ik| >= r =>  b_ik = 0.5 - |p_ik|
    
    We want to maximize r such that r <= b_c for all constraints c.
    Equivalent to minimizing -r.
    Using log-sum-exp approximation for the minimum:
    min(b) ≈ -1/beta * log( sum exp(-b * beta) )
    
    Let S = sum_{c} exp(-b_c * beta)
    r_soft = -1/beta * log(S)
    Objective f = -r_soft = 1/beta * log(S)
    
    Gradient of f w.r.t x:
    f = 1/beta * log( sum_c exp(-beta * b_c(x)) )
    df/dx = 1/beta * ( 1/S ) * sum_c exp(-beta * b_c) * (-beta * db_c/dx)
    df/dx = - sum_c ( exp(-beta * b_c) / S ) * db_c/dx
    df/dx = - sum_c w_c * db_c/dx
    where w_c = exp(-beta * b_c) / S.
    
    Note: This is a weighted average of the gradients of the constraints,
    with weights concentrated on the most violated (smallest b_c) constraints.
    """
    pts = points.reshape(-1, 3)
    
    # 1. Pairwise constraints
    # We only need to consider pairs that are "close" to avoid computing all 4950 exps if possible,
    # but for N=100, 4950 is small. However, exp can overflow if b is large negative?
    # b is distance, so positive. -beta * b is negative large. exp(very negative) -> 0.
    # So it's safe.
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We only care about unique pairs i < j
    # Create a mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask]
    b_pairs = pair_dists / 2.0
    
    # 2. Wall constraints
    # For each point i and dimension k, b_ik = 0.5 - |p_ik|
    abs_pts = np.abs(pts)
    b_walls = 0.5 - abs_pts
    # b_walls shape (N, 3)
    b_walls_flat = b_walls.flatten()
    
    # Combine all b values
    # Total constraints M = N*(N-1)/2 + 3*N
    # M = 4950 + 300 = 5250
    
    # Compute weights w_c = exp(-beta * b_c)
    # To avoid underflow/overflow, we can shift by max b? 
    # Actually, we want weights for the SMALL b.
    # Let b_min = min(b).
    # exp(-beta * b) is largest for smallest b.
    # We can compute z_c = -beta * b_c.
    # w_c = exp(z_c - max(z)) / sum(exp(z_j - max(z)))
    
    all_b = np.concatenate([b_pairs, b_walls_flat])
    
    # For numerical stability in log-sum-exp
    max_z = -beta * np.min(all_b)  # This is the max of -beta*b
    # Actually, simpler:
    # S = sum exp(-beta * b_c)
    # log S = max(-beta*b) + log(sum exp(-beta*b - max(-beta*b)))
    
    z = -beta * all_b
    z_max = np.max(z)
    log_S = z_max + np.log(np.sum(np.exp(z - z_max)))
    
    r_soft = -log_S / beta
    f = -r_soft  # We want to minimize f = -r_soft? 
    # Wait.
    # Objective to MINIMIZE: f = -r_soft = 1/beta * log_S.
    # Gradient of f:
    # f = 1/beta * log_S
    # df/dx = 1/beta * (1/S) * dS/dx
    # dS/dx = sum_c exp(-beta*b_c) * (-beta * db_c/dx)
    # df/dx = - sum_c (exp(-beta*b_c)/S) * db_c/dx
    # Let w_c = exp(-beta*b_c)/S.
    # w_c = exp(z_c - z_max) / sum(exp(z_j - z_max))
    
    w = np.exp(z - z_max)
    w_sum = np.sum(w)
    w = w / w_sum
    
    # Now compute gradient db_c/dx
    # grad_f = - sum_c w_c * db_c/dx
    
    grad = np.zeros((N, 3))
    
    # 1. Pairwise contributions
    # For each pair (i, j), constraint is d_ij/2.
    # d/dx_i (d_ij/2) = (x_i - x_j) / (2 * d_ij)
    # d/dx_j (d_ij/2) = (x_j - x_i) / (2 * d_ij) = -(x_i - x_j) / (2 * d_ij)
    
    # We have w_pairs corresponding to the upper triangle mask.
    # We need to map these back to the gradient array.
    
    # Iterate over pairs? 4950 iterations in Python is slow.
    # Vectorized approach:
    # Create a matrix W_pairs of shape (N, N) with weights in upper triangle.
    W_pairs = np.zeros((N, N))
    W_pairs[mask] = w[:len(b_pairs)]
    
    # grad_pair_i = sum_j W_ij * (x_i - x_j) / (2 * d_ij)
    # Let D be the distance matrix.
    # Let Vec be the vector difference matrix (x_i - x_j).
    # Term = W_ij * Vec_ij / (2 * D_ij)
    
    # To avoid division by zero, we already set diagonal to inf, so D_ij > 0 for i!=j.
    # But if D is very small, gradient is huge.
    
    # grad_i = sum_j W_ij * (x_i - x_j) / (2 * D_ij)
    
    # We can compute this as:
    # G_i = (1/2) * sum_j W_ij * (x_i/D_ij - x_j/D_ij)
    # This is essentially a graph Laplacian-like operation.
    
    # Let's compute it component-wise or using matrix multiplication tricks?
    # Actually, since N=100, a loop over 100 points might be acceptable if inner loop is vectorized.
    # Or loop over 4950 pairs? 5k iterations is fast in Python.
    
    # Let's do the loop over pairs for clarity and speed (5k is nothing).
    pair_indices = np.argwhere(mask)
    # pair_indices shape (4950, 2)
    
    # Vectorized accumulation for pairs
    # diff_vecs = pts[i] - pts[j]
    # weights = w_pairs
    # contrib_i = weights * diff_vecs / (2 * dists)
    # contrib_j = -contrib_i
    
    # We can accumulate into grad using np.add.at
    
    i_idx = pair_indices[:, 0]
    j_idx = pair_indices[:, 1]
    
    d_vecs = pts[i_idx] - pts[j_idx]
    d_norms = pair_dists
    
    # Avoid division by zero (should not happen due to inf diagonal, but just in case)
    d_norms_safe = np.where(d_norms < 1e-8, 1e-8, d_norms)
    
    # Factor: w * 0.5 / d
    factors = w[:len(b_pairs)] * 0.5 / d_norms_safe
    
    contrib = factors[:, np.newaxis] * d_vecs
    
    # grad[i] += contrib
    # grad[j] -= contrib
    
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    
    # 2. Wall contributions
    # Constraint: b_ik = 0.5 - |x_ik|
    # db_ik/dx_ik = -sign(x_ik)
    # grad_ik += - w_ik * (-sign(x_ik)) = w_ik * sign(x_ik)
    
    w_walls = w[len(b_pairs):].reshape(N, 3)
    signs = np.sign(pts)
    
    # grad += w_walls * signs
    grad += w_walls * signs
    
    # The total gradient of f (which we want to minimize) is:
    # grad_f = - (grad_pair_contrib + grad_wall_contrib)
    # Wait, let's re-verify the sign.
    # f = 1/beta log S
    # df/dx = - sum w_c db_c/dx
    # We computed sum w_c db_c/dx as `grad`.
    # So df/dx = -grad.
    
    return f, -grad

def _sgld_optimize(points, time_budget):
    """
    Run SGLD optimization.
    """
    start_time = time.time()
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Parameters
    beta = 50.0  # Sharpness of soft-min. Higher = closer to true min.
    # Start with low beta for smooth landscape, increase later?
    # Or keep it constant high enough to be accurate but smooth enough for gradient.
    # Let's ramp beta from 10 to 100.
    
    alpha = 0.01  # Step size
    T = 0.05      # Initial temperature
    T_end = 1e-4  # Final temperature
    
    # Number of steps
    # Estimate time per step.
    # _soft_min_radius is O(N^2) ~ 10k ops. Very fast.
    # Let's try 5000 steps.
    
    for step in range(10000):
        if time.time() > start_time + time_budget:
            break
            
        # Anneal temperature
        progress = step / 10000.0
        # Linear cooling? Exponential?
        T_curr = T * (T_end / T) ** progress
        # Or T_curr = T * exp(-k * step)
        # Let's use linear for simplicity
        T_curr = T + (T_end - T) * progress
        
        # Adjust alpha? Maybe decrease alpha as we cool.
        alpha_curr = alpha * (0.1 + 0.9 * (1 - progress))
        
        # Compute gradient
        try:
            f, grad = _soft_min_radius(pts, beta)
        except Exception:
            # If numerical issues, break
            break
            
        if not np.all(np.isfinite(grad)):
            break
            
        # SGLD update
        noise = np.random.normal(0, 1, pts.shape)
        # x_{t+1} = x_t + alpha * (-grad_f) + sqrt(2 * alpha * T) * noise
        # We want to MINIMIZE f, so move in direction -grad_f.
        # grad returned is grad_f.
        # So update: pts = pts - alpha * grad + noise_term
        # Wait, my _soft_min_radius returns f and -grad_f?
        # Let's check:
        # "return f, -grad" where grad was sum w db/dx.
        # So the second value is df/dx.
        # So we do: pts = pts - alpha * (df/dx) + noise
        
        noise_scale = np.sqrt(2 * alpha_curr * T_curr)
        pts = pts - alpha_curr * grad + noise_scale * noise
        
        # Project back to cube?
        # The wall constraints are in the objective, so they should push points in.
        # But if they go outside, the score becomes 0.
        # It's safer to clip to [-0.49, 0.49] to avoid invalid states?
        # No, clipping can distort the gradient.
        # Better to rely on the objective.
        # However, if a point goes to 0.6, b_wall is negative.
        # exp(-beta * negative) = exp(positive) -> large weight.
        # Gradient will be strong to push it back.
        
        # Check validity and score
        # Computing score every step is O(N^2).
        # Let's do it every 10 steps.
        if step % 10 == 0:
            # Ensure finite
            if np.all(np.isfinite(pts)):
                # Optional: clip to valid range to avoid extreme values
                # pts = np.clip(pts, -0.5, 0.5)
                curr_score = _score(pts)
                if curr_score > best_score:
                    best_score = curr_score
                    best_pts = pts.copy()
                    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution from grid
    points = _grid_init()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Multi-start SGLD
    # We have ~145 seconds.
    # Let's run multiple independent SGLD chains.
    # Each chain gets a portion of the time.
    
    num_chains = 5
    time_per_chain = (deadline - start_time) / num_chains - 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    for i in range(num_chains):
        if time.time() > deadline - 5:
            break
            
        # Initialize chain
        if i == 0:
            pts = points.copy()
        else:
            # Perturbed grid or random
            if random.random() < 0.5:
                pts = _grid_init() + np.random.normal(0, 0.05, (N, 3))
            else:
                # Random positions, then quick relaxation?
                # Just random might be too far.
                # Use grid with larger noise
                pts = _grid_init() + np.random.normal(0, 0.1, (N, 3))
        
        pts = np.clip(pts, -0.45, 0.45)
        
        # Run SGLD
        # Allocate time based on remaining
        remaining = deadline - 5 - time.time()
        if remaining <= 0:
            break
        
        opt_pts, opt_score = _sgld_optimize(pts, remaining)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
            
    # Final polish:
    # Take the best found and run a short, high-precision SGLD with very low T
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 1:
            # Use very low T and high beta
            # We can modify _sgld_optimize to accept params, or just call it again
            # For now, just call it again with the best points.
            # The annealing schedule in _sgld_optimize starts high T.
            # This might undo the precision.
            # Ideally, we'd want a "polish" mode.
            # Let's just run it. The low T end of the schedule will refine.
            
            opt_pts, opt_score = _sgld_optimize(best_points, remaining)
            if opt_score > best_score:
                best_points = opt_pts
                best_score = opt_score
                
    return best_points

# EVOLVE-BLOCK-END
