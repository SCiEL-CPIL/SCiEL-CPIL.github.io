"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parallel asynchronous Nelder-Mead simplex ensemble for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize 5 simplexes at diverse locations including corners and center
        # Use larger initial simplex size (1.0) to better explore Rastrigin's landscape
        start_points = [
            np.zeros(dim),           # center
            3.0 * np.ones(dim),      # corner-ish
            -3.0 * np.ones(dim),     # opposite corner-ish
            3.0 * np.array([1, -1, 1, -1, 1]),  # mixed signs
            3.0 * np.array([-1, 1, -1, 1, -1])  # mixed signs opposite
        ]
        
        n_simplexes = len(start_points)
        simplexes = []
        simplex_fvals = []
        call_count = 0
        
        # Helper to evaluate and track calls
        def eval_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            fx = func(x)
            call_count += 1
            return fx

        # Initialize each simplex with larger step size (1.0)
        init_step = 1.0
        for start in start_points:
            simplex = [start.copy()]
            for i in range(dim):
                v = start.copy()
                v[i] += init_step
                simplex.append(v)
            simplex = np.array(simplex)
            simplex = np.clip(simplex, -5, 5)
            simplexes.append(simplex)
            
            # Evaluate vertices
            fvals = np.array([eval_func(v) for v in simplex])
            simplex_fvals.append(fvals)
            
        # Global best tracking
        best_x = None
        best_f = np.inf
        
        for i in range(n_simplexes):
            min_idx = np.argmin(simplex_fvals[i])
            f_min = simplex_fvals[i][min_idx]
            if f_min < best_f:
                best_f = f_min
                best_x = simplexes[i][min_idx].copy()

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        stagnation_limit = 12  # Slightly lower to trigger restarts earlier
        stagnation_counters = [0] * n_simplexes
        simplex_best_f = [np.min(fv) for fv in simplex_fvals]
        
        # Track how many times each simplex has been restarted
        restart_counts = [0] * n_simplexes
        max_restarts = 10
        
        # Main loop
        while call_count < budget:
            for i in range(n_simplexes):
                if call_count >= budget:
                    break
                
                simplex = simplexes[i]
                fvals = simplex_fvals[i]
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check stagnation
                current_best = fvals[0]
                if current_best >= simplex_best_f[i] - 1e-10:
                    stagnation_counters[i] += 1
                else:
                    simplex_best_f[i] = current_best
                    stagnation_counters[i] = 0
                    if current_best < best_f:
                        best_f = current_best
                        best_x = simplex[0].copy()
                
                # If stagnant, perform a restart with adaptive jump
                if stagnation_counters[i] > stagnation_limit:
                    # If we've restarted too many times, try a larger jump
                    if restart_counts[i] < max_restarts:
                        restart_counts[i] += 1
                        # Adaptive jump size: larger jumps as we restart more
                        jump_size = 1.5 + 0.8 * restart_counts[i]
                        # Jump from the current best point
                        new_v = simplex[0].copy() + rng.normal(0, jump_size, dim)
                        new_v = np.clip(new_v, -4.5, 4.5)
                    else:
                        # Final resort: pure random restart
                        new_v = rng.uniform(-4.5, 4.5, dim)
                    
                    f_new = eval_func(new_v)
                    if call_count >= budget:
                        break
                    
                    # Replace worst vertex
                    simplex[-1] = new_v
                    fvals[-1] = f_new
                    
                    # Update global best if the new point is better
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_v.copy()
                        
                    # Re-sort and continue
                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    simplexes[i] = simplex
                    simplex_fvals[i] = fvals
                    continue
                
                # Standard Nelder-Mead step
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)
                if call_count >= budget:
                    break
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
                    if call_count >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)
                    if call_count >= budget:
                        break
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink: only shrink non-best vertices
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            fvals[j] = eval_func(simplex[j])
                            if call_count >= budget:
                                break
                        
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                
                simplexes[i] = simplex
                simplex_fvals[i] = fvals

        return best_x, best_f
# EVOLVE-BLOCK-END
