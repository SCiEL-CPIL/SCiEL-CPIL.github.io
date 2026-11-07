"""
Optimization strategy:
1. Latin Hypercube Sampling (LHS) for diverse initial exploration.
2. Multi-restart Adaptive (1+1)-ES for local refinement of top seeds.
   - Uses success-rate based step size adaptation for robustness.
   - Restarts are budgeted to allow exploration of multiple basins.
3. Final high-precision (1+1)-ES polish around the global best.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Latin Hypercube Sampling for diverse exploration
        # LHS gives better coverage than pure random sampling
        n_lhs = min(50, budget // 6)
        
        # Generate LHS points
        lhs_points = np.empty((n_lhs, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, n_lhs + 1)
            points_in_intervals = np.empty(n_lhs)
            for i in range(n_lhs):
                points_in_intervals[i] = rng.uniform(intervals[i], intervals[i+1])
            rng.shuffle(points_in_intervals)
            lhs_points[:, d] = points_in_intervals * 10.0 - 5.0
        
        # Evaluate LHS points
        lhs_values = np.empty(n_lhs)
        for i in range(n_lhs):
            if check_deadline():
                n_lhs = i
                break
            lhs_values[i] = evaluate(lhs_points[i])
        
        if n_lhs > 0:
            # Sort by fitness
            sort_idx = np.argsort(lhs_values)
            
            # Select up to 5 diverse seeds
            seeds_with_val = []
            for idx in sort_idx:
                if len(seeds_with_val) >= 5:
                    break
                seeds_with_val.append((lhs_points[idx].copy(), lhs_values[idx]))
        else:
            # Fallback to random
            x0 = rng.uniform(-5, 5, dim)
            f0 = evaluate(x0)
            seeds_with_val = [(x0, f0)]

        # Phase 2: Multi-restart Adaptive (1+1)-ES
        remaining_budget = budget - calls_made
        polish_budget = max(20, int(remaining_budget * 0.2))
        refine_budget = remaining_budget - polish_budget
        
        if refine_budget > 0 and seeds_with_val:
            n_seeds = len(seeds_with_val)
            iters_per_seed = max(1, refine_budget // n_seeds)
            
            for x_init, f_init in seeds_with_val:
                if check_deadline():
                    break
                
                x = x_init.copy()
                fx = f_init
                sigma = 1.0
                success_count = 0
                total_steps = 0
                
                for _ in range(iters_per_seed):
                    if check_deadline():
                        break
                    
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    total_steps += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        success_count += 1
                    
                    # Adapt sigma every 10 steps
                    if total_steps % 10 == 0:
                        success_rate = success_count / 10.0
                        if success_rate > 0.2:
                            sigma *= 1.5
                        else:
                            sigma *= 0.75
                        success_count = 0
                        
                    sigma = np.clip(sigma, 1e-6, 10.0)

        # Phase 3: Final Polish
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            success_count = 0
            total_steps = 0
            
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                
                # Adapt sigma
                if total_steps % 5 == 0:
                    success_rate = success_count / 5.0
                    if success_rate > 0.2:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    success_count = 0
                    
                sigma = np.clip(sigma, 1e-8, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
