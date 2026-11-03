import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 50 equal circles inside a unit circle.
    Uses Particle Swarm Optimization (PSO) with vectorized fitness evaluation.
    """
    np.random.seed(42)
    
    def evaluate_swarm(swarm):
        """
        Vectorized evaluation of radius for a swarm of particles.
        swarm: (S, N, 2)
        returns: (S,) array of radii
        """
        # Calculate pairwise distances for all particles
        # diff shape: (S, N, N, 2)
        diff = swarm[:, np.newaxis, :, :] - swarm[:, :, np.newaxis, :]
        dists = np.linalg.norm(diff, axis=3) # (S, N, N)
        
        # Fill diagonal with inf to ignore self-distance
        # We need to fill the diagonal of the N x N matrix for each S
        # Create an identity mask
        idx = np.arange(N)
        dists[:, idx, idx] = np.inf
        
        min_pair_dist = np.min(dists, axis=(1, 2)) # (S,)
        
        # Calculate boundary distances
        norms = np.linalg.norm(swarm, axis=2) # (S, N)
        max_norm = np.max(norms, axis=1) # (S,)
        min_boundary_dist = 1.0 - max_norm
        
        radii = np.minimum(min_pair_dist / 2.0, min_boundary_dist)
        
        # Handle any potential NaNs/Infs from numerical issues
        radii = np.where(np.isfinite(radii), radii, -1.0)
        
        return radii

    def repair(swarm):
        """
        Ensure all centers in the swarm are strictly inside the unit circle.
        """
        norms = np.linalg.norm(swarm, axis=2)
        outside = norms >= 1.0
        if np.any(outside):
            # Scale back points that are outside
            # norms[outside] is (num_outside,)
            # swarm[outside] is (num_outside, 2)
            # We need to divide each row by its norm
            # Use broadcasting
            scale_factors = 0.999 / norms[outside, np.newaxis]
            swarm[outside] = swarm[outside] * scale_factors
        return swarm

    # PSO Parameters
    swarm_size = 40
    max_iter = 1500
    w_start = 0.9
    w_end = 0.4
    c1 = 1.5
    c2 = 1.5
    
    # Initialize Swarm
    swarm = np.zeros((swarm_size, N, 2))
    velocities = np.zeros((swarm_size, N, 2))
    
    for i in range(swarm_size):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        swarm[i] = np.column_stack((x, y))
        
    velocities = np.random.uniform(-0.1, 0.1, (swarm_size, N, 2))
    
    # Initial Evaluation
    pbest_values = evaluate_swarm(swarm)
    pbest_positions = swarm.copy()
    
    gbest_index = np.argmax(pbest_values)
    gbest_position = swarm[gbest_index].copy()
    gbest_value = pbest_values[gbest_index]
    
    # Main Loop
    for it in range(max_iter):
        w = w_start - (w_start - w_end) * (it / max_iter)
        
        # Update velocities and positions
        # Vectorize the update if possible, but PSO updates are per-particle in terms of pbest/gbest
        # However, we can vectorize the random numbers and the arithmetic
        
        r1 = np.random.uniform(0, 1, (swarm_size, N, 2))
        r2 = np.random.uniform(0, 1, (swarm_size, N, 2))
        
        # v_i = w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
        velocities = w * velocities + c1 * r1 * (pbest_positions - swarm) + c2 * r2 * (gbest_position[np.newaxis, :, :] - swarm)
        
        swarm = swarm + velocities
        swarm = repair(swarm)
        
        # Evaluate
        current_values = evaluate_swarm(swarm)
        
        # Update pbest
        mask = current_values > pbest_values
        pbest_values[mask] = current_values[mask]
        pbest_positions[mask] = swarm[mask]
        
        # Update gbest
        gbest_idx = np.argmax(current_values)
        if current_values[gbest_idx] > gbest_value:
            gbest_value = current_values[gbest_idx]
            gbest_position = swarm[gbest_idx].copy()
            
    return gbest_position

# EVOLVE-BLOCK-END
