"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    # min_i min_k (0.5 - |c_ik|)
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    # To save memory and time for N=100, we can compute full matrix or use cdist
    # cdist is usually faster in C backend
    # But let's stick to vectorized numpy for simplicity and speed in this context
    # N=100 is small enough for full matrix
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_bcc():
    """Body-Centered Cubic initialization."""
    m = int(np.ceil((N / 2) ** (1.0 / 3.0) - 1e-9))
    if m < 1: m = 1
    s = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corners of the sub-cube
                points.append(((i + 0.25) * s - 0.5, (j + 0.25) * s - 0.5, (k + 0.25) * s - 0.5))
                # Center of the sub-cube
                points.append(((i + 0.75) * s - 0.5, (j + 0.75) * s - 0.5, (k + 0.75) * s - 0.5))
    
    pts = np.array(points)
    # Shuffle and take N
    np.random.shuffle(pts)
    return pts[:N].copy()

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

def _psolve(c0, time_limit=10.0):
    """
    Particle Swarm Optimization (PSO) to refine a single configuration.
    This is used to polish local minima found by other methods or as a primary search.
    """
    start_time = time.time()
    
    # Parameters
    n_particles = 10
    w = 0.8  # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    max_iter = 500
    
    # Initialize swarm
    # Center around c0
    spread = 0.05
    positions = c0 + np.random.normal(0, spread, size=(n_particles, N, 3))
    # Clamp to valid bounds
    positions = np.clip(positions, -0.499, 0.499)
    
    velocities = np.random.normal(0, 0.01, size=(n_particles, N, 3))
    
    # Initialize bests
    p_best = np.zeros_like(positions)
    p_best_score = np.zeros(n_particles)
    g_best = c0.copy()
    g_best_score = _score(c0)
    
    for i in range(n_particles):
        s = _score(positions[i])
        if s > p_best_score[i]:
            p_best_score[i] = s
            p_best[i] = positions[i].copy()
        if s > g_best_score:
            g_best_score = s
            g_best = positions[i].copy()
            
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(N, 3)
            r2 = np.random.rand(N, 3)
            
            vel = (w * velocities[i] + 
                   c1 * r1 * (p_best[i] - positions[i]) + 
                   c2 * r2 * (g_best - positions[i]))
            
            # Clip velocity to prevent overshooting
            vel = np.clip(vel, -0.05, 0.05)
            velocities[i] = vel
            
            # Update position
            new_pos = positions[i] + vel
            new_pos = np.clip(new_pos, -0.499, 0.499)
            
            # Evaluate
            s = _score(new_pos)
            
            if s > p_best_score[i]:
                p_best_score[i] = s
                p_best[i] = new_pos.copy()
                if s > g_best_score:
                    g_best_score = s
                    g_best = new_pos.copy()
            
            positions[i] = new_pos
            
    return g_best, g_best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # 1. Generate diverse initial populations
    # We will run multiple independent PSO searches from different starting points
    # to avoid getting stuck in the same local minimum.
    
    best_c = None
    best_score = -1.0
    
    # Starting points: BCC, Grid, and a few Randoms
    starts = [_init_bcc(), _init_grid()]
    for _ in range(5):
        starts.append(_init_random())
        
    # We have ~140s budget.
    # Let's allocate time to each start.
    # 7 starts * 20s each = 140s.
    # But we need to be careful with overhead.
    
    time_per_start = 18.0
    
    for s_idx, start_c in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        # Quick score of start
        s0 = _score(start_c)
        if s0 > best_score:
            best_score = s0
            best_c = start_c.copy()
            
        # Run PSO from this start
        # Adjust time limit based on remaining time
        remaining = deadline - time.time() - 2.0
        t_limit = min(time_per_start, remaining)
        if t_limit <= 0:
            break
            
        c_res, s_res = _psolve(start_c, time_limit=t_limit)
        
        if s_res > best_score:
            best_score = s_res
            best_c = c_res.copy()
            
        # If we found a very good solution, we can try to polish it further with a tighter PSO
        # But for now, let's just keep the best.
        
    return best_c
# EVOLVE-BLOCK-END
