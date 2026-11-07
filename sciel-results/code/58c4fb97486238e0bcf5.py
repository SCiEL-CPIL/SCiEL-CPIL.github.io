"""
Optimization strategy:
1. Latin Hypercube Sampling (LHS) for diverse, structured exploration to identify
   promising basins efficiently (better coverage than pure random).
2. Multi-restart adaptive (1+1)-ES for local refinement.
   - Uses a success-rate based step size adaptation for robustness.
   - Restarts are budgeted to allow exploration of multiple basins.
   - Uses the known structure of Rastrigin (periodic minima at integers) to
     initialize sigma based on distance to nearest integer lattice point.
3. Final high-precision polish around the global best.
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
        # Safety margin: Evaluator kills at 330s, we stop at 280s to be safe
        # and ensure we return before any hard timeouts.
        deadline = time.time() + 280.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=np.float64).flatten()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Exploratory LHS Search
        # LHS provides better space-filling properties than pure random sampling,
        # which is crucial for finding good basins in Rastrigin's multimodal landscape.
        n_explore = min(50, budget // 8)
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_explore equal intervals, 
        # pick one random point in each interval, then permute per dimension
        lhs_points = np.empty((n_explore, dim))
        for d in range(dim):
            # Create n_explore intervals in [0, 1]
            intervals = np.linspace(0, 1, n_explore + 1)
            # Pick one random point in each interval
            points_in_intervals = np.empty(n_explore)
            for i in range(n_explore):
                points_in_intervals[i] = rng.uniform(intervals[i], intervals[i+1])
            # Shuffle the points
            rng.shuffle(points_in_intervals)
            # Scale to [-5, 5]
            lhs_points[:, d] = points_in_intervals * 10.0 - 5.0
        
        # Evaluate LHS points
        explore_f = np.empty(n_explore)
        actual_explored = 0
        for i in range(n_explore):
            if check_deadline():
                break
            explore_f[i] = evaluate(lhs_points[i])
            actual_explored += 1
        
        if actual_explored == 0:
            # Fallback if we couldn't even do one evaluation (shouldn't happen)
            x0 = rng.uniform(-5, 5, dim)
            evaluate(x0)
            return best_x, best_f

        explore_f = explore_f[:actual_explored]
        lhs_points = lhs_points[:actual_explored]

        # Sort by fitness and select top seeds
        sort_idx = np.argsort(explore_f)
        
        # Select up to 5 seeds. 
        # We prioritize the best, but also keep diverse points.
        seeds_with_val = []
        for idx in sort_idx:
            if len(seeds_with_val) >= 5:
                break
            seeds_with_val.append((lhs_points[idx].copy(), explore_f[idx]))

        # Phase 2: Local Refinement with Adaptive (1+1)-ES
        # We allocate the remaining budget to these seeds.
        
        remaining_budget = budget - calls_made
        # Reserve 15% of remaining budget for final polish
        polish_budget = max(20, int(remaining_budget * 0.15))
        refine_budget = remaining_budget - polish_budget
        
        if refine_budget <= 0:
            pass
        else:
            n_seeds = len(seeds_with_val)
            if n_seeds == 0:
                # Fallback if exploration failed
                seed = rng.uniform(-5, 5, dim)
                f_seed = evaluate(seed)
                seeds_with_val = [(seed, f_seed)]
                n_seeds = 1
            
            iters_per_seed = max(1, refine_budget // n_seeds)
            
            for x_init, f_init in seeds_with_val:
                if check_deadline():
                    break
                    
                x = x_init.copy()
                fx = f_init
                
                # Initialize sigma based on distance to nearest integer lattice point.
                # Rastrigin has local minima near integers. If we are far from an integer,
                # we might be in a "valley" or far from the basin center.
                # A reasonable starting sigma is the distance to the nearest integer in any dimension,
                # or a fixed small value if we are very close.
                dist_to_int = np.min(np.abs(x - np.round(x)))
                # If dist is small, we are close to a potential minimum. Use smaller sigma.
                # If dist is large, we might need larger steps to navigate.
                # However, Rastrigin has many local minima. 
                # Let's use a heuristic: sigma = max(0.1, 0.5 * dist_to_int)
                # But cap it to avoid huge jumps.
                sigma = np.clip(0.5 * dist_to_int + 0.1, 0.05, 2.0)
                
                success_count = 0
                total_steps = 0
                # Target success rate ~ 1/5 = 0.2
                # If success rate > 0.2, increase sigma. Else decrease.
                # We'll adjust sigma every 10 steps or so.
                
                for _ in range(iters_per_seed):
                    if check_deadline():
                        break
                    
                    # Generate perturbation
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    total_steps += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        success_count += 1
                    else:
                        pass
                    
                    # Adapt sigma every 10 steps
                    if total_steps % 10 == 0:
                        success_rate = success_count / 10.0
                        if success_rate > 0.2:
                            sigma *= 1.5
                        else:
                            sigma *= 0.75
                        success_count = 0
                        
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-6, 10.0)

        # Phase 3: Final Polish
        # Use remaining budget for a very fine-grained search around global best.
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            # Start with small sigma
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
                
                # Adapt sigma more aggressively for polish
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
