"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers))

    # Minimum pairwise distance / 2
    # Use cKDTree for efficiency, though N=50 is small enough for brute force if needed
    # But cKDTree is robust.
    tree = cKDTree(centers)
    # We need the minimum distance. query_pairs with r=inf is not ideal for min.
    # Better: query each point for nearest neighbor.
    # For N=50, brute force is ~1225 ops, which is trivial.
    # Let's just do brute force for simplicity and speed in Python loop? No, vectorized is better.
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_dist_to_wall, min_pair_dist / 2.0)

def _get_initial_centers():
    """
    Generate a good initial configuration using a jittered grid.
    """
    rng = np.random.default_rng(42)
    # 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select 50 points randomly from the 64 grid points
    indices = rng.permutation(64)[:50]
    centers = grid[indices].copy()
    
    # Add small jitter
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _soft_min_objective(x, beta):
    """
    Objective to minimize: -soft_min(constraints).
    We want to maximize the minimum radius r.
    r = min( min_wall, min_pair/2 )
    Maximize r  <=>  Minimize -r.
    Using log-sum-exp approximation for max:
    max(a, b) approx = beta * log(exp(a/beta) + exp(b/beta))
    Here we have many constraints.
    Let C be the set of constraints (wall distances and half-pairwise distances).
    We want max(C).
    soft_max(C) = beta * log( sum( exp(c_i / beta) ) )
    We minimize -soft_max(C).
    """
    centers = x.reshape(N, 3)
    
    # 1. Wall constraints: d_w = 0.5 - |centers|
    # Shape (N, 3)
    d_walls = 0.5 - np.abs(centers)
    
    # 2. Pairwise constraints: d_p = ||c_i - c_j|| / 2
    # For efficiency in the SA loop, we might not want to recompute full pairwise 
    # matrix every single step if we can avoid it, but N=50 is small.
    # Let's compute all pairwise distances.
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    d_pairs = dists / 2.0
    
    # Combine all constraints into one vector
    # d_walls is (N, 3) -> 150 values
    # d_pairs is (1225,) -> 1225 values
    # Total ~1375 constraints.
    
    # To avoid overflow in exp, we can subtract the max? 
    # But we want the max. 
    # log-sum-exp trick: 
    # M = max(c)
    # log(sum(exp(c_i))) = M + log(sum(exp(c_i - M)))
    # soft_max = beta * (M + log(sum(exp((c_i - M)/beta))))
    # This is stable.
    
    # However, doing this for every objective call in SA might be slow if done in Python loops.
    # Vectorized numpy is fast.
    
    # Let's just compute the max directly first to check validity? No, we need gradient/smoothness.
    # Actually, for SA, we just need the value.
    # The "soft min" (which is max of constraints) can be approximated.
    
    # Optimization: 
    # If we are in a valid configuration (all constraints > 0), the max is the bottleneck.
    # If we are invalid, we want to push out of the invalid region.
    
    # Let's use the stable log-sum-exp for the MAX.
    # c = concat(d_walls.flatten(), d_pairs)
    
    c = np.concatenate([d_walls.flatten(), d_pairs])
    
    # Stable log-sum-exp for max
    c_max = np.max(c)
    # If c_max is very negative, exp(c/beta) underflows to 0. 
    # If c_max is large, we are fine.
    # We want to minimize -soft_max.
    
    # soft_max = beta * log( sum( exp(c_i / beta) ) )
    # = beta * log( exp(c_max/beta) * sum( exp((c_i - c_max)/beta) ) )
    # = beta * ( c_max/beta + log( sum( exp((c_i - c_max)/beta) ) ) )
    # = c_max + beta * log( sum( exp((c_i - c_max)/beta) ) )
    
    # To prevent underflow in exp((c_i - c_max)/beta) when c_i << c_max, 
    # note that (c_i - c_max) <= 0.
    # If (c_i - c_max)/beta < -50, exp is ~0.
    
    scaled = (c - c_max) / beta
    # Clip to avoid underflow warnings (though exp(-100) is fine, just 0)
    # np.exp handles -inf to 0.
    sum_exp = np.sum(np.exp(scaled))
    
    soft_max_val = c_max + beta * np.log(sum_exp)
    
    # We want to maximize soft_max_val, so minimize -soft_max_val.
    return -soft_max_val

def _score(centers):
    """Fast scoring function for SA loop."""
    centers = np.asarray(centers, dtype=np.float64)
    # Walls
    min_wall = np.min(0.5 - np.abs(centers))
    # Pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial guess
    centers = _get_initial_centers()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Simulated Annealing Parameters
    # We need to tune beta. 
    # If beta is too small, it's like hard min (non-differentiable, but SA doesn't need gradients).
    # If beta is too large, it's an average, which might allow some points to be very close 
    # while others are far, not maximizing the *minimum*.
    # We want the objective to reflect the minimum. 
    # So beta should be small enough to approximate max.
    # Let's start with a moderate beta and anneal it? 
    # Or keep beta fixed and anneal the temperature of the noise?
    
    # Strategy: 
    # 1. Use a fixed small beta (e.g., 0.01) for the objective to guide the "energy".
    # 2. Use SA to explore.
    # 3. Periodically run a local optimizer (L-BFGS-B) on the current best to polish.
    
    beta_obj = 0.05  # Smoothing parameter for the objective
    
    # SA Parameters
    T = 0.05  # Initial temperature (size of moves)
    T_end = 0.001
    cooling_rate = 0.9995
    
    current_centers = centers.copy()
    current_obj = _soft_min_objective(current_centers.flatten(), beta_obj)
    
    # Pre-compute indices for pairwise distance
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    steps = 0
    while time.time() < deadline - 10:
        # Generate a move
        # Randomly pick a few points to move
        n_moves = np.random.randint(1, 5)
        indices_to_move = np.random.choice(N, n_moves, replace=False)
        
        # Move size scaled by T
        # We want moves to be in the range of the radius (~0.1)
        # So T should be around 0.01-0.05
        move = np.random.normal(0, T, size=(n_moves, 3))
        
        new_centers = current_centers.copy()
        new_centers[indices_to_move] += move
        
        # Clip to cube to ensure validity (or penalize? Clipping is faster and keeps valid)
        # Note: Clipping can create "stickiness" to walls. 
        # But for sphere packing, being inside is a hard constraint.
        # If we clip, we might lose the ability to "push" against the wall effectively?
        # Actually, if we clip, the distance to wall becomes 0, which is bad.
        # Better to allow moves outside and let the objective penalize it?
        # The objective _soft_min_objective penalizes negative wall distances.
        # So we should NOT clip in the SA loop, but check validity for the "best" tracker.
        
        # However, if points go far outside, the objective becomes very negative (bad).
        # Let's just compute the new objective.
        
        new_obj = _soft_min_objective(new_centers.flatten(), beta_obj)
        
        # Acceptance probability
        delta = new_obj - current_obj
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_centers = new_centers
            current_obj = new_obj
            
            # Check if this is a valid configuration and if it improves the actual score
            # We only update best if it's valid (all constraints > 0? No, score handles it)
            # Score is min(wall, pair/2). If any is negative, score is negative.
            # We want to maximize score.
            
            curr_score = _score(current_centers)
            if curr_score > best_score:
                best_score = curr_score
                best_centers = current_centers.copy()
            
            # Periodic Local Optimization
            # Every 500 steps, run a quick L-BFGS-B polish on the current best
            # This helps escape the "noisy" state of SA and find the local optimum of the basin.
            if steps % 500 == 0:
                try:
                    # Define a local objective for L-BFGS-B
                    # We want to maximize the min distance.
                    # Use the same soft min objective.
                    # Bounds: [-0.5, 0.5]
                    res = minimize(
                        _soft_min_objective,
                        best_centers.flatten(),
                        args=(beta_obj,),
                        method='L-BFGS-B',
                        bounds=[(-0.5, 0.5)] * (N * 3),
                        options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
                    )
                    
                    # Check result
                    new_best = res.x.reshape(N, 3)
                    new_best_score = _score(new_best)
                    if new_best_score > best_score:
                        best_score = new_best_score
                        best_centers = new_best.copy()
                        # Update current state to the polished best to continue from there
                        current_centers = best_centers.copy()
                        current_obj = _soft_min_objective(current_centers.flatten(), beta_obj)
                except Exception:
                    pass
        
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            # If we've cooled down, we might restart with a higher T if time permits?
            # Or just continue.
            
        steps += 1

    # Final Polish
    # Run one last, more intensive L-BFGS-B on the best found
    try:
        res = minimize(
            _soft_min_objective,
            best_centers.flatten(),
            args=(0.01,),  # Smaller beta for sharper max approximation
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
        )
        final_centers = res.x.reshape(N, 3)
        final_score = _score(final_centers)
        if final_score > best_score:
            best_centers = final_centers
            best_score = final_score
    except Exception:
        pass

    return best_centers

# EVOLVE-BLOCK-END
