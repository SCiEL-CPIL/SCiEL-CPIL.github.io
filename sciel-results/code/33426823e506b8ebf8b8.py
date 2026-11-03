import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 140s.
    DEADLINE = start_time + 140.0
    
    def elapsed():
        return time.time() - start_time
        
    def remaining():
        return DEADLINE - time.time()

    def compute_radius(pos):
        """
        Calculate the exact maximum radius for a given configuration.
        Returns 0.0 if invalid.
        """
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape != (N, 3) or not np.all(np.isfinite(pos)):
            return 0.0
        if np.any(pos < -CUBE_HALF - 1e-12) or np.any(pos > CUBE_HALF + 1e-12):
            return 0.0
        
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair_dist = np.sqrt(np.min(d2))
        
        # Wall distances
        walls = CUBE_HALF - np.abs(pos)
        min_wall_dist = np.min(walls)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return float(r)

    def soft_min_radius(pos, beta):
        """
        Smooth surrogate for the maximum radius.
        Uses Log-Sum-Exp to approximate the minimum of all constraint values.
        Higher beta -> sharper minimum approximation.
        """
        pos = pos.reshape(-1, 3)
        
        # Pairwise half-distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        half_d = 0.5 * np.sqrt(d2)
        
        # Wall distances
        walls = CUBE_HALF - np.abs(pos) # Shape (N, 3)
        
        # Collect all constraints: pairwise (N*(N-1)/2) and walls (3*N)
        # We need to flatten the upper triangle of half_d to avoid double counting, 
        # though for soft-min it's symmetric so it doesn't strictly matter for the value, 
        # just the weight. But keeping unique pairs is cleaner.
        mask = ~np.eye(N, dtype=bool)
        # half_d[mask] gives all i!=j. We want unique pairs? 
        # Actually, soft-min is dominated by the minimum. Duplicate terms just increase the 
        # "effective" weight of the minimum. Since all pairs are symmetric, 
        # including both (i,j) and (j,i) is fine, it just makes the soft-min slightly 
        # "softer" or equivalent to a higher beta. Let's just take all unique pairs to be precise.
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_vals = half_d[i_idx, j_idx]
        wall_vals = walls.ravel()
        
        all_vals = np.concatenate([pair_vals, wall_vals])
        
        # Log-Sum-Exp trick for numerical stability
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        scaled = -beta * all_vals
        m = np.max(scaled)
        if m == -np.inf:
            return 0.0
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        return -lse / beta

    def objective_smooth(x, beta):
        """Minimize the negative soft-min radius."""
        return -soft_min_radius(x, beta)

    def objective_direct(x):
        """Minimize the negative exact radius (for final polish)."""
        return -compute_radius(x.reshape(-1, 3))

    def local_optimize(pos, time_limit):
        """
        Run L-BFGS-B optimization on the smooth surrogate, then polish.
        """
        if time_limit - time.time() < 5:
            return pos, compute_radius(pos)

        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        x0 = pos.ravel()
        
        # Multi-stage beta annealing
        # Start with low beta (smooth), increase to high beta (sharp)
        betas = [10, 50, 200, 1000]
        
        current_x = x0
        current_r = compute_radius(pos)
        
        for beta in betas:
            if time.time() > time_limit - 2:
                break
                
            res = minimize(
                objective_smooth,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                current_x = res.x
                # Update current radius estimate
                current_r = compute_radius(current_x.reshape(-1, 3))
        
        # Final Polish: Use direct objective (nonsmooth) with a small step size
        # This helps snap to the exact kink of the minimum.
        if time.time() < time_limit - 1:
            res = minimize(
                objective_direct,
                current_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res.success:
                candidate_x = res.x
                candidate_r = compute_radius(candidate_x.reshape(-1, 3))
                if candidate_r > current_r:
                    current_x = candidate_x
                    current_r = candidate_r
        
        final_pos = current_x.reshape(-1, 3)
        final_r = compute_radius(final_pos)
        
        return final_pos, final_r

    # --- Initializations ---
    
    # 1. Jittered Grid (Structural)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    np.random.seed(42)
    init_configs = []
    
    # Jittered Grid
    init_configs.append(grid + np.random.normal(0, 0.02, size=grid.shape))
    
    # Random configs
    for _ in range(15):
        # Sample within a slightly smaller box to avoid immediate wall conflicts
        rand_c = np.random.uniform(-CUBE_HALF * 0.9, CUBE_HALF * 0.9, size=(N, 3))
        init_configs.append(rand_c)
        
    best_centers = init_configs[0]
    best_r = compute_radius(best_centers)
    
    # --- Optimization Loop ---
    
    num_configs = len(init_configs)
    for idx, config in enumerate(init_configs):
        if time.time() > DEADLINE - 10:
            break
            
        # Allocate time: give more time to early configs? Or equal?
        # Let's give equal time but ensure we finish.
        time_left = DEADLINE - time.time() - 5 # 5s safety buffer
        allocated = time_left / (num_configs - idx)
        
        c, r = local_optimize(config, time.time() + allocated)
        
        if r > best_r:
            best_r = r
            best_centers = c
            
    # Final check
    final_r = compute_radius(best_centers)
    if final_r < 0 or not np.all(np.isfinite(best_centers)):
        # Fallback to grid if something went wrong
        return grid
        
    return best_centers
