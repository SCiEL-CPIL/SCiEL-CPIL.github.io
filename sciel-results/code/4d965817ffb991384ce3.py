"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    # Safety margin: The limit is 150s. We stop at 145s to ensure we return.
    deadline = start_time + 145.0
    
    # Helper to compute the true score (radius)
    def compute_radius(c):
        # c is (N, 3)
        # Check bounds first for validity (though we clip, it's good practice)
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 0.0
        
        # Pairwise distances
        # Use squared distances for speed, then sqrt only for the min? 
        # No, we need the actual min distance.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist_sq = np.min(dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    # Helper to compute a smooth objective function for optimization
    # We want to maximize r.
    # Let r_pair = min_pair_dist / 2
    # Let r_wall = min_wall_dist
    # We want to maximize min(r_pair, r_wall).
    # Use a smooth approximation: -log(sum(exp(-beta * x))) is a smooth min.
    # Alternatively, just use a penalty for violations if we fix a target r?
    # Better: Simulated Annealing on the "energy" E = -log(min_dist_smooth)
    
    def compute_energy_and_grad(c, beta):
        """
        Computes a smooth energy and its gradient.
        Lower energy is better.
        We define energy based on the minimum distance and wall distance.
        To make it differentiable, we use a soft-min approximation.
        
        Let d_min be the minimum pairwise distance.
        Let w_min be the minimum wall distance.
        We want to maximize r = min(d_min/2, w_min).
        
        Instead of a complex soft-min, we can use a penalty approach:
        Assume a target radius r_target.
        If constraints are violated, energy is high.
        But we don't know r_target.
        
        Let's stick to the gradient of the "worst case" distance.
        The gradient of the min of a set of functions is the gradient of the active one.
        If multiple are active, it's the average (subgradient).
        
        For Simulated Annealing, we need a scalar objective that is noisy/smooth enough.
        Let's use:
        Objective = - ( min_pair_dist / 2 + min_wall_dist ) / 2  ? No, that's not maximin.
        
        Let's use the "soft min" of the two constraints:
        r_soft = -1/beta * log( 0.5*exp(-beta * r_pair) + 0.5*exp(-beta * r_wall) )
        We want to maximize r_soft. So Energy = -r_soft.
        
        However, computing r_pair and r_wall requires finding the min, which is non-smooth.
        
        Alternative SA Objective:
        E = sum_{i<j} max(0, 2*r_target - dist_ij)^2 + sum_i max(0, r_target - wall_dist_i)^2
        This requires knowing r_target.
        
        Let's try a different approach for SA:
        Move points randomly. Accept if the new radius is better, or with probability exp(-delta/T) if worse.
        This is "standard" SA on the discrete state space of positions.
        The "energy" is just the negative radius: E = -r.
        If r_new > r_old, accept.
        If r_new < r_old, accept with prob exp((r_new - r_old)/T).
        
        This is simple and robust. The "gradient" is not needed, only the evaluation.
        Evaluation is O(N^2). 100^2 = 10,000 operations. Very fast in NumPy.
        We can do ~10,000 - 50,000 steps in 145s?
        Let's estimate: 1 step = 1 eval + 1 move.
        If eval takes 1ms, we can do 100k steps.
        If eval takes 5ms, we can do 20k steps.
        
        Let's implement standard SA.
        """
        pass

    # Initial Configuration: Grid
    # A grid is a good starting point because it's already "spread out".
    nx, ny, nz = 5, 5, 4
    x_coords = (np.arange(nx) + 0.5) / nx - 0.5
    y_coords = (np.arange(ny) + 0.5) / ny - 0.5
    z_coords = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    c = grid[:N].copy()
    
    # Ensure we start valid
    c = np.clip(c, -0.49, 0.49)
    
    best_c = c.copy()
    best_r = compute_radius(c)
    
    # SA Parameters
    # Start with a temperature that allows significant moves.
    # The "energy" difference is in radius (approx 0.1).
    # If we move by 0.01, radius might change by 0.001.
    # T should be comparable to expected radius changes.
    T_start = 0.05
    T_end = 1e-6
    # Number of steps? We will run until deadline.
    
    rng = np.random.default_rng(42)
    
    # To make SA effective, we need to propose moves.
    # Gaussian moves with adaptive sigma.
    sigma = 0.01
    
    current_c = c.copy()
    current_r = best_r
    
    # Pre-allocate arrays for speed?
    # We will just recompute radius every step.
    
    step = 0
    # Track time
    last_time = time.time()
    
    # We want to explore the landscape.
    # If we get stuck, increase T (reheat).
    # Simple strategy: Geometric cooling, but if no improvement for X steps, reheat slightly.
    
    no_improve_count = 0
    reheat_threshold = 500 # steps
    
    while time.time() < deadline:
        # Generate move
        # Pick a random subset of points to move? Or all?
        # Moving all points with small sigma is like a global jitter.
        # Moving a few points is like local refinement.
        # Let's move a random subset of 10 points per step to allow local adjustments.
        indices = rng.choice(N, size=10, replace=False)
        
        # Gaussian noise
        noise = rng.normal(0, sigma, size=(10, 3))
        
        new_c = current_c.copy()
        new_c[indices] += noise
        
        # Clip to bounds
        new_c = np.clip(new_c, -0.5, 0.5)
        
        # Evaluate
        new_r = compute_radius(new_c)
        
        # Acceptance Criterion
        if new_r >= current_r:
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Boltzmann acceptance
            # delta = current_r - new_r (positive)
            # prob = exp(-delta / T)
            delta = current_r - new_r
            # If delta is large, prob is small.
            # Use log for stability if T is small?
            # exp(-delta/T)
            # If T is very small, this underflows to 0.
            if T > 1e-9:
                accept_prob = np.exp(-delta / T)
            else:
                accept_prob = 0.0
            
            if rng.random() < accept_prob:
                current_c = new_c
                current_r = new_r
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        step += 1
        
        # Cooling Schedule
        # We want to cool down.
        # But we also want to adapt sigma.
        
        # If we are not improving, maybe we are stuck in a local min.
        # Reheat?
        if no_improve_count > reheat_threshold:
            # Reheat to T_start * 0.5
            T = max(T_end, T_start * 0.5)
            no_improve_count = 0
            # Also try larger moves?
            sigma = 0.02
        else:
            # Cool down
            # Geometric cooling: T = T * alpha
            # But we want to finish at T_end.
            # Let's use a linear cooling based on step count? No, time based is better.
            # Or just simple geometric.
            T *= 0.995
            
            # Adapt sigma:
            # If we are accepting many "bad" moves, T is too high or sigma too high?
            # If we are accepting few moves, T is too low or sigma too small?
            # Let's keep sigma constant for now, or decay it slightly.
            sigma *= 0.999
            
        # Periodic check for time
        if step % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break

    return best_c
# EVOLVE-BLOCK-END
