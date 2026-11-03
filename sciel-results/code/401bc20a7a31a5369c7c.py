"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses Particle Swarm Optimization (PSO) to explore the configuration space.
    1. Initialize particles with grid + jitter.
    2. Iteratively update positions based on personal and global bests.
    3. Maximize the minimum radius (distance to walls and between circles).
    """
    
    def score(points):
        """Calculate the maximum radius for a given configuration."""
        # points shape: (N, 2)
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Minimum pairwise distance
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        dist_x = 0.5 - np.abs(points[:, 0])
        dist_y = 0.5 - np.abs(points[:, 1])
        min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def batch_score(points):
        """
        Calculate scores for multiple configurations.
        points shape: (M, N, 2) where M is number of particles.
        Returns array of shape (M,).
        """
        M = points.shape[0]
        scores = np.zeros(M)
        
        # Pairwise distances for all particles
        # diff shape: (M, N, 1, 2) - (M, 1, N, 2) -> (M, N, N, 2)
        diff = points[:, :, np.newaxis, :] - points[:, np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=3)
        
        # Fill diagonal with inf for each particle
        # We need to set dists[i, i, i] = inf for all i, j in N
        idx = np.arange(N)
        dists[:, idx, idx] = np.inf
        
        min_pair_dists = np.min(dists, axis=(1, 2))
        
        # Boundary distances
        dist_x = 0.5 - np.abs(points[:, :, 0])
        dist_y = 0.5 - np.abs(points[:, :, 1])
        min_boundary_dists = np.min(np.minimum(dist_x, dist_y), axis=1)
        
        scores = np.minimum(min_pair_dists / 2.0, min_boundary_dists)
        return scores

    def pso():
        """Run Particle Swarm Optimization."""
        np.random.seed(42)
        
        swarm_size = 100
        max_iters = 5000
        w = 0.9  # Inertia weight
        c1 = 1.5 # Cognitive parameter
        c2 = 1.5 # Social parameter
        
        # Initialize swarm with grid + jitter
        # Grid: 7x7 is 49, need 50. Let's use a dense grid and jitter.
        # Or just random uniform in a tighter box.
        # Grid init is often better for packing.
        # Let's create a 7x7 grid and add one point in center? 
        # 7x7 = 49. Add 1 at (0,0)? No, (0,0) is center.
        # Let's just use random uniform for simplicity but constrained to [-0.4, 0.4]
        # Actually, grid is better. Let's use a 5x10 grid? 50 points.
        # Spacing: 1.0/5 = 0.2 (x), 1.0/10 = 0.1 (y).
        # This gives a very bad initial radius.
        # Better: Random uniform is fine if swarm is large enough.
        
        positions = np.random.uniform(-0.45, 0.45, size=(swarm_size, N, 2))
        velocities = np.random.uniform(-0.01, 0.01, size=(swarm_size, N, 2))
        
        # Clip initial positions to valid range to ensure finite scores
        positions = np.clip(positions, -0.5, 0.5)
        
        # Calculate initial scores
        scores = batch_score(positions)
        
        # Personal bests
        pbest_positions = positions.copy()
        pbest_scores = scores.copy()
        
        # Global best
        global_idx = np.argmax(pbest_scores)
        gbest_position = pbest_positions[global_idx].copy()
        gbest_score = pbest_scores[global_idx]
        
        # Track best score ever
        best_score_ever = gbest_score
        best_pos_ever = gbest_position.copy()
        
        start_time = time.time()
        time_limit = 110.0 # Leave buffer for overhead
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Update inertia weight (linear decrease)
            w = 0.9 - 0.4 * (i / max_iters)
            
            r1 = np.random.rand(swarm_size, N, 1)
            r2 = np.random.rand(swarm_size, N, 1)
            
            # Update velocities
            velocities = (w * velocities + 
                          c1 * r1 * (pbest_positions - positions) + 
                          c2 * r2 * (gbest_position - positions))
            
            # Update positions
            positions += velocities
            
            # Clip positions to square
            positions = np.clip(positions, -0.5, 0.5)
            
            # Calculate new scores
            new_scores = batch_score(positions)
            
            # Update personal bests
            improved = new_scores > pbest_scores
            pbest_positions[improved] = positions[improved]
            pbest_scores[improved] = new_scores[improved]
            
            # Update global best
            current_global_idx = np.argmax(pbest_scores)
            if pbest_scores[current_global_idx] > gbest_score:
                gbest_score = pbest_scores[current_global_idx]
                gbest_position = pbest_positions[current_global_idx].copy()
                
                if gbest_score > best_score_ever:
                    best_score_ever = gbest_score
                    best_pos_ever = gbest_position.copy()
                    
            # Optional: Restart stagnant particles
            # If a particle hasn't improved in a long time, re-initialize it near global best
            # This is a common heuristic but requires tracking history. 
            # For simplicity, we rely on the swarm dynamics.
            
        return best_pos_ever

    # Run PSO
    result = pso()
    
    # Final polish with a few local SA steps? 
    # Or just return the PSO result.
    # Let's add a quick local refinement using gradient-free local search (e.g. Nelder-Mead is too slow for 100D).
    # Simple local perturbation:
    # Try 100 random small moves around the best point.
    best_r = score(result)
    current = result.copy()
    
    for _ in range(200):
        # Perturb 1-5 points
        num_points = np.random.randint(1, 6)
        idxs = np.random.choice(N, num_points, replace=False)
        noise = np.random.normal(0, 0.005, size=(num_points, 2))
        new_pos = current.copy()
        new_pos[idxs] += noise
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_r = score(new_pos)
        if new_r > best_r:
            best_r = new_r
            current = new_pos
            
    return current

# EVOLVE-BLOCK-END
