"""
Packing 100 equal spheres in a unit cube.
Strategy: Simulated Annealing with LogSumExp objective, followed by L-BFGS-B polish.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 100
CUBE_BOUND = 0.5

def _get_objective(x, beta):
    """
    Computes the negative soft-min radius.
    x: shape (N, 3)
    beta: sharpness of the soft-min approximation.
    Returns: -r_soft
    """
    # Reshape x to (N, 3)
    x = x.reshape(N, 3)
    
    # Pairwise distances
    # dists[i, j] = ||x_i - x_j||
    # Using broadcasting
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal (self-distance)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: distance from center to nearest wall
    # x is in [-0.5, 0.5]
    # dist to wall = min(0.5 - x_i, 0.5 + x_i) for each dim
    # Since we want spheres inside, radius is limited by min(0.5 - |x_ik|)
    # Let's compute the minimum distance to any wall for each sphere
    # d_wall_i = min(0.5 - x_i0, 0.5 - x_i1, 0.5 - x_i2, 0.5 + x_i0, 0.5 + x_i1, 0.5 + x_i2)
    # Actually, simpler: min(0.5 - np.abs(x_i))
    abs_x = np.abs(x)
    wall_dists = 0.5 - np.max(abs_x, axis=1)
    
    # We want to maximize min( min_{i<j} dists[i,j]/2, min_i wall_dists[i] )
    # Note: The radius r is half the distance between centers for touching spheres.
    # The condition is r <= dist(i,j)/2 AND r <= wall_dist(i).
    # So we want to maximize min( min_{i<j} dists[i,j]/2, min_i wall_dists[i] ).
    
    # Let's define the "constraints" values we want to maximize the minimum of.
    # Pairwise constraint values: dists / 2
    # Boundary constraint values: wall_dists
    
    # Combine all constraints into one vector
    # Number of pairs is N*(N-1)/2. Number of wall constraints is N.
    # For efficiency, we can just compute softmin on the pairwise part and wall part separately?
    # No, we want the global min.
    # Let's create a vector of all "limiting" distances.
    # But N=100 means ~5000 pairs. Vector size ~5100. This is fine.
    
    # Extract upper triangle of dists
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Combine
    all_limits = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow/underflow, subtract max
    # We want to maximize min(x), so we minimize -min(x).
    # Soft min of x is approx min(x) for large beta.
    # soft_min(x) = -1/beta * log( sum(exp(-beta * x)) )
    
    # For numerical stability:
    # log(sum(exp(-beta * x))) = -beta * max(x) + log(sum(exp(-beta * (x - max(x)))))
    # Wait, soft min is usually defined as -1/beta log sum exp(-beta x).
    # Let m = min(all_limits).
    # exp(-beta * (all_limits - m)) will have one term ~1 and others > 1? No, if x > m, -beta(x-m) is negative, exp is < 1.
    # So max term is 1.
    
    m = np.min(all_limits)
    # Subtract m to make the smallest value 0
    shifted = all_limits - m
    # Now soft_min(shifted) is close to 0.
    # soft_min(all_limits) = m + soft_min(shifted)
    
    # Calculate log-sum-exp of -beta * shifted
    # Since shifted >= 0, -beta * shifted <= 0.
    # exp(-beta * shifted) <= 1.
    # Sum is at least 1.
    
    # Optimization: if beta is very large, only the minimum matters.
    # We can just return -m if we are in "hard" mode, but we want smoothness.
    
    # To be safe with large beta:
    # val = -1.0 / beta * np.log(np.sum(np.exp(-beta * shifted)))
    # However, if beta * shifted is very negative, exp is 0.
    # If beta is large, this is fine.
    
    # Let's use a stable logsumexp
    # max_val = np.max(-beta * shifted) = 0 (since shifted >= 0)
    # log_sum = 0 + np.log(np.sum(np.exp(-beta * shifted)))
    
    # Actually, if beta is huge, np.exp(-beta * shifted) might underflow for non-min elements, which is good.
    # The sum will be close to 1.
    
    # Let's just compute it directly with a cap on beta to prevent issues if needed, 
    # but standard float64 handles exp(-1000) -> 0 fine.
    
    # One potential issue: if all_limits are very close, the sum is large.
    
    r_soft = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    
    # We want to maximize r_soft, so minimize -r_soft
    return -r_soft

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization
    # Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    # Flatten for optimizer
    x_flat = x0.flatten()
    
    # 2. Simulated Annealing
    # We will use a coarse soft-min for SA to allow larger moves and escape local minima
    beta_sa = 50.0
    current_x = x_flat.copy()
    current_obj = _get_objective(current_x, beta_sa)
    
    best_x = current_x.copy()
    best_obj = current_obj
    
    T = 0.01
    T_min = 1e-5
    alpha = 0.99
    
    # Estimate number of steps based on time
    # Each step involves distance calculation.
    # Let's run a fixed number of steps or until time runs out.
    # N=100, dist calc is ~5000 ops. Python overhead dominates.
    # Let's aim for ~20000 steps in 60s?
    # Let's try 10000 steps first and see.
    
    np.random.seed(42)
    
    steps = 0
    max_steps = 50000  # Cap on steps
    
    while steps < max_steps:
        if time.time() > deadline - 20.0: # Leave 20s for polish
            break
            
        # Generate move: perturb a random subset of points
        n_perturb = np.random.randint(1, 4) # 1 to 3 points
        idx = np.random.choice(N, n_perturb, replace=False)
        
        new_x = current_x.copy()
        # Gaussian perturbation
        noise = np.random.normal(0, T, size=(n_perturb, 3))
        new_x[idx * 3:(idx + 1) * 3] += noise.flatten()
        
        # Check boundaries? 
        # The objective penalizes going out, but we might want to hard-constrain 
        # to keep values reasonable. 
        # For now, let objective handle it. If it goes too far, obj becomes very bad.
        
        new_obj = _get_objective(new_x, beta_sa)
        
        # Metropolis criterion
        if new_obj < current_obj:
            current_x = new_x
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = current_x.copy()
        else:
            # Accept with probability exp((current_obj - new_obj) / T)
            # Note: we are minimizing, so delta = new_obj - current_obj > 0
            prob = np.exp(-((new_obj - current_obj) / T))
            if np.random.rand() < prob:
                current_x = new_x
                current_obj = new_obj
        
        steps += 1
        T *= alpha
        
        if T < T_min:
            # Reheat occasionally? Or just stop.
            # For simplicity, let's stop SA phase if T is too low.
            break
    
    # 3. Local Polish with L-BFGS-B
    # Use the best found so far
    x_polish = best_x.copy()
    
    # Use a larger beta for polishing to approximate the hard min
    beta_polish = 1000.0
    
    def polish_objective(x_flat):
        return _get_objective(x_flat, beta_polish)
    
    # Constraints: keep points inside [-0.5, 0.5]
    # L-BFGS-B supports bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            polish_objective, 
            x_polish, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_obj:
            x_polish = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    # 4. Final High-Precision Polish
    # Run L-BFGS-B again with even tighter tolerances or higher beta if time permits
    if time.time() < deadline - 5.0:
        beta_final = 5000.0
        def final_objective(x_flat):
            return _get_objective(x_flat, beta_final)
        
        try:
            res2 = minimize(
                final_objective, 
                x_polish, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.success or res2.fun < best_obj:
                x_polish = res2.x
        except Exception:
            pass
    
    # Reshape and return
    final_x = x_polish.reshape(N, 3)
    
    # Ensure validity: clip to bounds just in case (though L-BFGS-B should respect them)
    final_x = np.clip(final_x, -0.5, 0.5)
    
    return final_x

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f}s")
    print(f"Shape: {res.shape}")
    
    # Calculate actual radius
    diff = res[:, np.newaxis, :] - res[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    abs_x = np.abs(res)
    wall_dists = 0.5 - np.max(abs_x, axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    print(f"Radius: {r}")
