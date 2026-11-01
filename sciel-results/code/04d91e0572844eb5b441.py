# EVOLVE-BLOCK-START
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """
    Compute the maximum valid radius for a set of centers.
    Returns r = min(min_pair_dist, min_boundary_dist) / 2.0
    """
    # Pairwise distances
    # Using broadcasting for speed
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    # Centers are in [-0.5, 0.5]
    # Distance to nearest face is 0.5 - |coord|
    abs_centers = np.abs(centers)
    dist_to_faces = 0.5 - np.max(abs_centers, axis=1)
    min_boundary_dist = np.min(dist_to_faces)
    
    return min(min_pair_dist, min_boundary_dist) / 2.0

def _sa_optimize(initial_centers, time_limit=100.0):
    """
    Simulated Annealing to maximize the minimum distance.
    """
    centers = initial_centers.copy()
    current_radius = _compute_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    # Temperature schedule
    # Start with a high temperature that allows large moves
    # The "energy" is -radius. So delta_E = -new_radius + current_radius
    # We want to accept if new_radius > current_radius (delta_E < 0)
    # Or with probability exp(-delta_E / T) = exp((new_radius - current_radius) / T)
    
    # We need to scale T appropriately.
    # Radius is ~0.1. Changes are small.
    # Let's use T in the range of radius changes.
    T = 0.05
    T_min = 1e-6
    alpha = 0.999
    
    rng = np.random.RandomState(42)
    
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < time_limit:
        # Perform a batch of moves to amortize time checking
        # Or just one move? One move is O(N^2) = 2500 ops. 
        # 120s / 2500 ops is huge. We can do many moves.
        # Let's do 1000 moves per loop iteration to check time.
        
        for _ in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            # Choose a move
            # 1. Move a random sphere by a random vector
            i = rng.randint(N)
            # Step size should be adaptive or fixed?
            # Fixed step size scaled by T?
            # Let's use a fixed step size that shrinks with T
            step_size = 0.05 * (T / 0.05) # Linear scaling, or just fixed?
            # Actually, fixed step size is often better, T controls acceptance.
            # Let's try fixed step size 0.01 to 0.05
            move = rng.randn(3) * 0.02
            
            new_centers = centers.copy()
            new_centers[i] += move
            
            # Ensure inside cube
            # If it goes out, we can either reject or clip.
            # Clipping changes the distance. Rejecting is safer for SA logic.
            # But if we reject, we waste a step.
            # Let's clip to [-0.499, 0.499] to keep valid, but this creates bias.
            # Better: if any coord > 0.5 or < -0.5, reject immediately?
            # Or just let _compute_radius handle it (boundary dist becomes negative).
            # If boundary dist is negative, radius is negative.
            # We want to maximize radius. Negative radius is bad.
            
            new_radius = _compute_radius(new_centers)
            
            # Calculate energy difference
            # E = -R
            # dE = E_new - E_old = -R_new - (-R_old) = R_old - R_new
            delta_E = current_radius - new_radius
            
            # Acceptance
            if delta_E < 0:
                # Improvement
                centers = new_centers
                current_radius = new_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
            else:
                # Worse, accept with probability exp(-dE / T)
                # Note: dE is positive here.
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    centers = new_centers
                    current_radius = new_radius
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
        
        # Occasionally, if stuck, do a large global move?
        # For now, rely on the temperature.
        
    return best_centers

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Generate multiple initializations and run SA on each
    # Or just one good initialization?
    # SA is stochastic. Multiple runs are better.
    
    best_centers_overall = None
    best_radius_overall = 0.0
    
    rng_global = np.random.RandomState(0)
    
    # We have ~110 seconds.
    # Let's run 5 SA iterations with 20s each.
    n_runs = 5
    time_per_run = time_limit / n_runs
    
    for i in range(n_runs):
        # Different initialization for each run
        # Random uniform
        centers = rng_global.uniform(-0.45, 0.45, (N, 3))
        
        # Run SA
        optimized_centers = _sa_optimize(centers, time_limit=time_per_run)
        radius = _compute_radius(optimized_centers)
        
        if radius > best_radius_overall:
            best_radius_overall = radius
            best_centers_overall = optimized_centers.copy()
            
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return best_centers_overall
# EVOLVE-BLOCK-END
