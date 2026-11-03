"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=100.0):
    """
    Compute a smooth approximation of the radius using log-sum-exp.
    r = min(wall_r, min_pair/2)
    """
    # Wall constraints
    c1 = 0.5 - centers[:, 0]
    c2 = 0.5 + centers[:, 0]
    c3 = 0.5 - centers[:, 1]
    c4 = 0.5 + centers[:, 1]
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Take upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dist[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([c1, c2, c3, c4, pair_dists])
    
    # Numerical stability: subtract max from exponents
    y = -beta * all_constraints
    max_y = np.max(y)
    sum_exp = np.sum(np.exp(y - max_y))
    log_sum_exp = max_y + np.log(sum_exp)
    
    return -log_sum_exp / beta

def _objective_smooth(x, beta=100.0):
    """Objective to minimize: -soft_min_radius."""
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _local_refine_smooth(centers, time_limit=5.0, beta=100.0):
    """
    Use L-BFGS-B to refine the configuration using the smooth objective.
    """
    x0 = centers.ravel()
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False

    try:
        res = minimize(
            _objective_smooth, 
            x0, 
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 1000,
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'disp': False
            },
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _pso_search(time_budget, n_particles=50, n_iter=1000):
    """
    Particle Swarm Optimization to find a good global configuration.
    """
    start_time = time.time()
    
    # Initialize particles
    # Start with some structured inits and some random
    positions = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    # Init 0: Grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    positions[0] = grid
    
    # Init 1: Hexagonal
    spacing = 0.14
    dy = spacing * np.sqrt(3) / 2
    hex_points = []
    for j in range(10):
        for i in range(10):
            x = i * spacing - 0.5
            y = j * dy - 0.5
            if j % 2 != 0:
                x += spacing / 2
            hex_points.append((x, y))
    hex_points = np.array(hex_points)[:N]
    # Center it
    hex_points -= np.mean(hex_points, axis=0)
    max_ext = np.max(np.abs(hex_points))
    if max_ext > 0.45:
        hex_points *= (0.45 / max_ext)
    positions[1] = hex_points
    
    # Rest random
    np.random.seed(42)
    positions[2:] = np.random.uniform(-0.45, 0.45, size=(n_particles - 2, N, 2))
    
    # Initial velocities
    velocities = np.random.uniform(-0.01, 0.01, size=(n_particles, N, 2))
    
    # Personal bests
    pbest_pos = positions.copy()
    pbest_val = np.zeros(n_particles)
    for i in range(n_particles):
        pbest_val[i] = _radius(positions[i])
        
    gbest_val = np.max(pbest_val)
    gbest_pos = positions[np.argmax(pbest_val)].copy()
    
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    iter_count = 0
    while time.time() - start_time < time_budget:
        for i in range(n_particles):
            if time.time() - start_time > time_budget:
                break
                
            r1 = np.random.random((N, 2))
            r2 = np.random.random((N, 2))
            
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest_pos[i] - positions[i]) + c2 * r2 * (gbest_pos - positions[i])
            
            # Clip velocities to prevent exploding
            np.clip(velocities[i], -0.1, 0.1, out=velocities[i])
            
            positions[i] += velocities[i]
            
            # Boundary handling: bounce back
            # If a point goes outside [-0.5, 0.5], reflect it
            # Simple reflection: if x > 0.5, x = 1.0 - x, vx = -vx
            # We just clip for stability in PSO, the radius function will penalize it heavily
            # Actually, better to just let the fitness function handle it, or clip to valid range
            # Clipping to valid range might trap particles on boundary.
            # Let's just evaluate the radius as is. If invalid (overlap/wall), radius is low.
            
            val = _radius(positions[i])
            
            if val > pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = positions[i].copy()
                if val > gbest_val:
                    gbest_val = val
                    gbest_pos = positions[i].copy()
        
        iter_count += 1
        
        # Adaptive parameters
        # Reduce inertia over time to converge
        w = 0.7 - 0.2 * (iter_count / n_iter)
        
    return gbest_pos, gbest_val

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Global Search with PSO
    # Allocate 60% of time to PSO
    pso_time_budget = 0.6 * DEADLINE
    pso_start = time.time()
    
    try:
        pso_centers, pso_r = _pso_search(pso_time_budget, n_particles=30, n_iter=2000)
        best_centers = pso_centers
        best_r = pso_r
    except Exception:
        # Fallback to grid if PSO fails
        ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
        ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
        grid = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
        best_centers = grid
        best_r = _radius(grid)

    # Phase 2: Local Refinement
    # Use remaining time to refine the best candidate found by PSO
    # We can also try refining a few other good candidates if we kept track, 
    # but for simplicity and robustness, we refine the global best.
    # We might also refine a few random restarts if time permits.
    
    time_left = DEADLINE - 5 - (time.time() - start_time)
    if time_left > 2:
        # Refine the PSO result
        refined = _local_refine_smooth(best_centers, time_limit=time_left * 0.7, beta=100.0)
        r = _radius(refined)
        if r > best_r:
            best_r = r
            best_centers = refined
            
    # If time remains, try a few random restarts + local refine
    time_left = DEADLINE - 5 - (time.time() - start_time)
    while time_left > 3:
        np.random.seed(int(time.time() * 1000) % 100000)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        # Quick local refine
        alloc = min(2.0, time_left * 0.5)
        refined = _local_refine_smooth(c_rand, time_limit=alloc, beta=100.0)
        r = _radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined
        
        time_left = DEADLINE - 5 - (time.time() - start_time)

    return best_centers
