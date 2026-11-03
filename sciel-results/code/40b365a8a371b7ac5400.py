import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing to maximize the minimum radius of 200 circles
    in a unit square. This is a stochastic global optimization approach,
    distinct from force-directed methods.
    """
    
    def evaluate(points):
        """
        Computes the maximum radius r for the given configuration.
        """
        # Wall distances
        # min(0.5 - x, 0.5 + x) for x and y
        wx = np.minimum(0.5 - points[:, 0], 0.5 + points[:, 0])
        wy = np.minimum(0.5 - points[:, 1], 0.5 + points[:, 1])
        wall_dists = np.minimum(wx, wy)
        min_wall = np.min(wall_dists)
        
        # Pair distances
        # Use broadcasting to compute all pair distances
        # diff[i, j, k] = points[i, k] - points[j, k]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    # Initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    best_points = grid[:N].copy()
    best_score = evaluate(best_points)
    
    start_time = time.time()
    time_limit = 110
    
    # Simulated Annealing Parameters
    T0 = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    step_scale = 0.02
    
    points = best_points.copy()
    current_score = best_score
    T = T0
    
    # Track history for adaptive step size if needed, but simple decay is often enough
    # We will run multiple restarts if time permits
    
    restarts = 0
    max_restarts = 5
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        restarts += 1
        
        # Reset or continue? For ILS/SA, restarting with a perturbed best is good.
        if restarts == 1:
            points = best_points.copy()
            # Add small noise to break symmetry
            points += np.random.normal(0, 0.01, size=points.shape)
            points = np.clip(points, -0.49, 0.49)
            current_score = evaluate(points)
            T = T0
        else:
            # Perturb the best found so far
            points = best_points.copy()
            # Larger perturbation for restart
            points += np.random.normal(0, 0.05, size=points.shape)
            points = np.clip(points, -0.49, 0.49)
            current_score = evaluate(points)
            T = T0 * 0.5  # Start cooler on restarts to exploit
            
        # SA Loop
        while time.time() - start_time < time_limit:
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Generate move
            # Step size decreases with T
            move_size = step_scale * T / T0
            if move_size < 1e-5:
                move_size = 1e-5
                
            new_points = points.copy()
            new_points[idx] += np.random.normal(0, move_size, size=2)
            
            # Keep inside bounds (soft constraint via penalty or hard clip)
            # Hard clip is safer for validity, but might trap in corners.
            # Let's clip to [-0.5, 0.5]
            new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
            
            new_score = evaluate(new_points)
            
            # Acceptance criterion
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Metropolis criterion
                delta = current_score - new_score # Positive if worse
                if delta < 1e-12:
                    accept = True
                else:
                    # P = exp(-delta / T)
                    # Note: delta is in radius units. T is in radius units.
                    # If T is small, exp(-large) -> 0.
                    prob = math.exp(-delta / T)
                    accept = (np.random.random() < prob)
                
                if accept:
                    points = new_points
                    current_score = new_score
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                break
                
    return best_points
# EVOLVE-BLOCK-END
