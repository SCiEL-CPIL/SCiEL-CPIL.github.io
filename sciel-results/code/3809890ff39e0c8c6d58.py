"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Use simulated annealing to maximize the minimum distance between circles
    and from walls for 50 equal circles in a unit square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    
    Approach:
    1. Initialize with a hexagonal packing pattern.
    2. Use simulated annealing with random perturbations.
    3. Objective: maximize the minimum of pairwise distances (divided by 2) and wall distances.
    4. Use a cooling schedule and accept worse moves with probability exp(delta/T).
    5. Multiple restarts with different seeds to explore different local optima.
    """
    np.random.seed(42)  # For reproducibility
    
    def compute_score(centers):
        """Compute the radius r for the given centers."""
        n = len(centers)
        if n < 2:
            return 0.5
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances: min over all centers and all 4 walls
        min_wall = np.min(np.min(np.min(0.5 - np.abs(centers), axis=1)))
        
        return min(min_pair, min_wall)
    
    def init_hexagonal(n_circles):
        """Initialize with a hexagonal packing pattern."""
        # Hexagonal packing: rows offset by half spacing
        # For N circles, use approximately sqrt(N) rows
        rows = int(np.ceil(np.sqrt(n_circles * 2 / np.sqrt(3))))
        cols = int(np.ceil(n_circles / rows))
        
        # Spacing for hexagonal packing
        # In a hexagonal lattice, horizontal spacing = d, vertical spacing = d * sqrt(3)/2
        # Offset every other row by d/2
        d = 1.0 / (cols + 0.5)  # Approximate spacing
        y_spacing = d * np.sqrt(3) / 2
        
        centers = []
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n_circles:
                    break
                x = (col + 0.5) / cols - 0.5
                if row % 2 == 1:
                    x += 0.5 / cols
                y = (row + 0.5) / rows - 0.5
                centers.append([x, y])
                count += 1
        
        return np.array(centers[:n_circles])
    
    def init_grid(n_circles):
        """Initialize with a square grid pattern."""
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles].copy()
    
    def run_sa(centers_init, n_iterations, T_start, T_end, step_size_start, seed):
        """Run simulated annealing from given initial configuration."""
        np.random.seed(seed)
        centers = centers_init.copy()
        
        def compute_score_fast(centers):
            """Compute the radius r for the given centers."""
            n = len(centers)
            if n < 2:
                return 0.5
            # Pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            # Set diagonal to inf
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            # Wall distances
            min_wall = np.min(np.min(np.min(0.5 - np.abs(centers), axis=1)))
            
            return min(min_pair, min_wall)
        
        current_score = compute_score_fast(centers)
        best_centers = centers.copy()
        best_score = current_score
        
        # Cooling schedule
        alpha = (T_end / T_start) ** (1.0 / n_iterations)
        
        for it in range(n_iterations):
            T = T_start * (alpha ** it)
            
            # Pick a random center to perturb
            i = np.random.randint(N)
            old_pos = centers[i].copy()
            
            # Adaptive step size: larger early, smaller later
            progress = it / n_iterations
            step_size = step_size_start * (1.0 - 0.8 * progress)
            
            # Random perturbation
            centers[i] += np.random.normal(0, step_size, 2)
            
            # Clip to stay within bounds
            centers[i] = np.clip(centers[i], -0.4999, 0.4999)
            
            new_score = compute_score_fast(centers)
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_centers = centers.copy()
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if T > 1e-12 and np.random.rand() < np.exp(delta / T):
                    current_score = new_score
                else:
                    # Reject: revert
                    centers[i] = old_pos
        
        return best_centers, best_score
    
    # Try multiple initializations and SA runs
    best_overall_centers = None
    best_overall_score = -np.inf
    
    # Different initialization strategies
    init_strategies = [
        ('hex', init_hexagonal),
        ('grid', init_grid),
    ]
    
    # Multiple restarts with different parameters
    sa_configs = [
        {'n_iterations': 30000, 'T_start': 0.02, 'T_end': 1e-7, 'step_size_start': 0.08},
        {'n_iterations': 30000, 'T_start': 0.01, 'T_end': 1e-7, 'step_size_start': 0.05},
        {'n_iterations': 30000, 'T_start': 0.03, 'T_end': 1e-7, 'step_size_start': 0.10},
    ]
    
    for init_name, init_func in init_strategies:
        for cfg_idx, cfg in enumerate(sa_configs):
            seed = 42 + cfg_idx * 100 + (0 if init_name == 'hex' else 50)
            centers_init = init_func(N)
            best_centers, best_score = run_sa(
                centers_init,
                cfg['n_iterations'],
                cfg['T_start'],
                cfg['T_end'],
                cfg['step_size_start'],
                seed
            )
            
            if best_score > best_overall_score:
                best_overall_score = best_score
                best_overall_centers = best_centers
    
    return best_overall_centers
# EVOLVE-BLOCK-END
