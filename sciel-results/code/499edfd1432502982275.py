"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses multi-restart Simulated Annealing with optimized scoring.
    """
    
    def score(points):
        """Calculate the maximum radius for a given configuration - optimized."""
        # Compute pairwise distances efficiently
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundaries
        dist_x = 0.5 - np.abs(points[:, 0])
        dist_y = 0.5 - np.abs(points[:, 1])
        min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def score_incremental(points, changed_idx, old_pos):
        """Compute score more efficiently by only checking pairs involving changed point."""
        # Only need to check distances from changed point to all others
        dists_to_changed = np.sqrt(np.sum((points - points[changed_idx]) ** 2, axis=1))
        # Exclude self
        min_pair_dist = np.min(np.delete(dists_to_changed, changed_idx))
        
        # Also need global min - but we can track it. For simplicity, use full score if needed
        # Actually, let's just use full score for correctness
        return score(points)

    def grid_init(rows, cols, jitter=0.0):
        """Initialize points on a grid with optional jitter."""
        xs = np.linspace(-0.45, 0.45, cols)
        ys = np.linspace(-0.45, 0.45, rows)
        X, Y = np.meshgrid(xs, ys)
        points = np.column_stack((X.ravel(), Y.ravel()))
        if jitter > 0:
            points += np.random.normal(0, jitter, size=points.shape)
            points = np.clip(points, -0.5, 0.5)
        return points

    def anneal(points, T_start, T_end, alpha, step_size, time_budget, rng):
        """Run simulated annealing from given initial configuration."""
        current_score = score(points)
        best_points = points.copy()
        best_score = current_score
        
        T = T_start
        ss = step_size
        start_time = time.time()
        
        while T > T_end:
            if time.time() - start_time > time_budget:
                break
                
            idx = rng.randint(N)
            delta = rng.normal(0, ss, size=2)
            
            new_points = points.copy()
            new_points[idx] += delta
            new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
            
            new_score = score(new_points)
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                prob = np.exp((new_score - current_score) / T)
                if rng.random() < prob:
                    points = new_points
                    current_score = new_score
                    
            T *= alpha
            ss *= 0.9995
            
        return best_points, best_score

    def hill_climb(points, time_budget, rng):
        """Refine configuration with local search."""
        current_score = score(points)
        best_points = points.copy()
        best_score = current_score
        step_size = 0.005
        
        start_time = time.time()
        
        while time.time() - start_time < time_budget:
            # Adaptive step size: decrease if no improvement recently
            idx = rng.randint(N)
            delta = rng.normal(0, step_size, size=2)
            
            new_points = points.copy()
            new_points[idx] += delta
            new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
            
            new_score = score(new_points)
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                step_size = min(step_size * 1.1, 0.02)  # Can increase slightly
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                step_size *= 0.95  # Decrease step size on failure
                step_size = max(step_size, 0.0005)
                
        return best_points, best_score

    # Multi-restart strategy with different initial configurations
    np.random.seed(42)
    overall_best_points = None
    overall_best_score = -1
    
    start_time = time.time()
    time_limit = 115.0
    
    # Try different grid configurations
    configs = [
        (10, 5, 0.0),    # 10x5 grid
        (5, 10, 0.0),    # 5x10 grid  
        (7, 7, 0.02),    # 7x7 grid with jitter (49 points, add 1)
        (10, 5, 0.01),   # 10x5 with small jitter
    ]
    
    # For 7x7, we have 49 points, need to add 1
    for rows, cols, jitter in configs:
        if time.time() - start_time > time_limit - 20:
            break
            
        if rows * cols == N:
            points = grid_init(rows, cols, jitter)
        elif rows * cols == N - 1:
            # Add one more point
            points = grid_init(rows, cols, jitter)
            # Add a point in a gap
            extra = np.array([0.0, 0.0])
            points = np.vstack([points, extra])
            points = np.clip(points, -0.5, 0.5)
        else:
            continue
            
        rng = np.random.RandomState(hash((rows, cols, jitter)) % (2**32))
        
        # Annealing phase
        anneal_time = min(40.0, time_limit - 20 - (time.time() - start_time))
        if anneal_time <= 5:
            continue
            
        best_pts, best_sc = anneal(points, 
                                    T_start=0.03, 
                                    T_end=0.0005, 
                                    alpha=0.9995, 
                                    step_size=0.04, 
                                    time_budget=anneal_time, 
                                    rng=rng)
        
        # Hill climbing phase
        hc_time = min(15.0, time_limit - 5 - (time.time() - start_time))
        if hc_time <= 1:
            best_pts, best_sc = best_pts, best_sc
        else:
            best_pts, best_sc = hill_climb(best_pts, hc_time, rng)
        
        if best_sc > overall_best_score:
            overall_best_score = best_sc
            overall_best_points = best_pts.copy()
    
    # Final refinement on best solution
    if overall_best_points is not None:
        remaining = time_limit - 2 - (time.time() - start_time)
        if remaining > 5:
            rng = np.random.RandomState(12345)
            overall_best_points, overall_best_score = hill_climb(overall_best_points, remaining, rng)
    
    if overall_best_points is None:
        # Fallback
        overall_best_points = grid_init(10, 5, 0.0)
    
    return overall_best_points

# EVOLVE-BLOCK-END
