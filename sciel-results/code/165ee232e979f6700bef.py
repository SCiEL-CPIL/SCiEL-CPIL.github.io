"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _radius(c):
    """Compute the packing radius for a configuration c of shape (N, 3)."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Use a more memory-efficient approach if needed, but for N=100 it's fine
    # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # min_pair = np.min(dists)
    
    # More efficient:
    from scipy.spatial.distance import pdist
    dists = pdist(c, metric='euclidean')
    if len(dists) > 0:
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
        
    return min(min_wall, min_pair / 2.0)


def _objective(c_flat):
    """Objective function to minimize (negative radius)."""
    c = c_flat.reshape(N, 3)
    # Clip to valid range to ensure feasibility during optimization
    c = np.clip(c, -0.5, 0.5)
    r = _radius(c)
    return -r


def construct():
    # Use Particle Swarm Optimization (PSO)
    # Since scipy doesn't have built-in PSO, we implement a simple one.
    
    np.random.seed(42)
    
    swarm_size = 50
    max_iter = 100
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    # Initialize swarm
    # Start with a mix of random and grid-based initializations
    grid_m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(grid_m) + 0.5) / grid_m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    particles = np.zeros((swarm_size, N * 3))
    for i in range(swarm_size):
        if i < 5:
            # Some particles start near the grid solution
            particles[i] = grid.ravel() + np.random.randn(N * 3) * 0.01
        else:
            # Others start randomly
            particles[i] = np.random.uniform(-0.4, 0.4, N * 3)
            
    # Evaluate initial positions
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i]
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    for it in range(max_iter):
        for i in range(swarm_size):
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.5, 0.5)
            
            # Evaluate
            val = _objective(particles[i])
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < global_best_val:
                    global_best_val = val
                    global_best = particles[i].copy()
                    
    # Final refinement using L-BFGS-B on the best solution
    c_best = global_best.reshape(N, 3)
    result = minimize(_objective, c_best.ravel(), method='L-BFGS-B', 
                      bounds=[(-0.5, 0.5)] * (N * 3), options={'maxiter': 200})
    
    c_final = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
