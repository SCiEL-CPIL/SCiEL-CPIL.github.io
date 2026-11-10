"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc

def _compute_score(pos):
    """
    Computes the score for a given configuration of centers.
    Score is the maximum radius r such that spheres of radius r
    centered at pos are disjoint and inside the unit cube [-0.5, 0.5]^3.
    """
    # Pairwise distances
    # pos shape: (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Ignore self-distance
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    # For each point, distance to nearest wall is 0.5 - max(|x|, |y|, |z|)
    # Wait, distance to wall x=0.5 is 0.5 - x.
    # Distance to wall x=-0.5 is 0.5 + x.
    # Min of these is 0.5 - |x|.
    # So for point (x,y,z), min dist to any wall is min(0.5-|x|, 0.5-|y|, 0.5-|z|).
    # This is 0.5 - max(|x|, |y|, |z|).
    max_abs_coord = np.max(np.abs(pos), axis=1)
    wall_dists = 0.5 - max_abs_coord
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def construct():
    # Deadline guard: 120s to be safe against 150s limit
    deadline = time.time() + 120.0
    
    # 1. Initialization
    # Use Halton sequence for quasi-random distribution
    try:
        qmc = scipy.stats.qmc.Halton(3, scramble=True)
        init_pos = qmc.random(N) * 1.0 - 0.5
    except Exception:
        # Fallback to random if SciPy QMC fails
        init_pos = np.random.uniform(-0.5, 0.5, (N, 3))
        
    # Ensure points are strictly inside to avoid boundary issues initially
    init_pos = np.clip(init_pos, -0.48, 0.48)
    
    best_pos = init_pos.copy()
    best_score = _compute_score(best_pos)
    current_pos = init_pos.copy()
    current_score = best_score
    
    # 2. Simulated Annealing Parameters
    T = 0.05          # Initial temperature
    T_min = 1e-7      # Minimum temperature
    cool_factor = 0.99 # Multiplicative cooling factor
    cool_interval = 1000 # Cool down every N iterations
    step_scale = 1.0   # Scale for Gaussian steps
    
    iter_count = 0
    
    # 3. Optimization Loop
    while time.time() < deadline:
        # Determine how many points to move
        # Moving a subset allows for larger steps and better mixing
        n_move = max(1, int(N * 0.2))
        indices = np.random.choice(N, n_move, replace=False)
        
        # Generate Gaussian steps
        steps = np.random.normal(0, T * step_scale, (n_move, 3))
        
        new_pos = current_pos.copy()
        new_pos[indices] += steps
        
        # Check if any point moved outside the cube
        # If so, reject the move to maintain validity
        if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
            # Rejection
            pass
        else:
            new_score = _compute_score(new_pos)
            
            # Metropolis Acceptance Criterion
            if new_score > current_score:
                # Always accept improvement
                current_pos = new_pos
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pos = current_pos.copy()
            else:
                # Accept worse solution with probability exp(delta / T)
                delta = new_score - current_score
                if delta < 0:
                    # Prevent underflow/overflow
                    exponent = delta / T
                    if exponent > -700:
                        prob = np.exp(exponent)
                    else:
                        prob = 0.0
                    
                    if np.random.random() < prob:
                        current_pos = new_pos
                        current_score = new_score
        
        iter_count += 1
        
        # Cooling schedule
        if iter_count % cool_interval == 0:
            T *= cool_factor
            if T < T_min:
                T = T_min

    return best_pos
# EVOLVE-BLOCK-END
