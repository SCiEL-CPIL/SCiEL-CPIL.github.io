"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Basin-Hopping with Local ES Refinement for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= total_budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy:
        # 1. Global Exploration: Sample a coarse grid + random points to find the global basin.
        #    Lunacek has a deep global minimum and a shallow local minimum.
        #    The local min is typically around -1.5 in all dims (in standard space).
        #    The global min is further away. We need to find the "deep" basin.
        # 2. Local Refinement: Use (1+1)-ES starting from the best point found in step 1.
        
        # Phase 1: Global Search (50% of budget)
        # We use a combination of random sampling and a structured grid to ensure coverage.
        # Since the dimension is 5, a full grid is too expensive, so we use random + latin hypercube-like spread.
        
        exploration_budget = total_budget // 2
        
        # Generate diverse starting points
        # 1. Origin
        # 2. Corners of the box
        # 3. Random points
        # 4. Points near the known local minimum structure (but shifted)
        
        candidates = []
        
        # Add origin
        candidates.append(np.zeros(n))
        
        # Add corners (2^5 = 32 corners)
        for i in range(1 << n):
            corner = np.array([((i >> j) & 1) * 2 - 1 for j in range(n)]) * 5.0
            candidates.append(corner)
            
        # Add mid-points of edges? Maybe too many.
        # Add random points
        num_random = exploration_budget - len(candidates)
        if num_random > 0:
            random_points = rng.uniform(-5, 5, (num_random, n))
            for p in random_points:
                candidates.append(p)
                
        # Evaluate all candidates to find the best basin
        # We evaluate them one by one.
        # To be efficient, we can just evaluate all and keep the best.
        
        current_best_x = None
        current_best_f = np.inf
        
        for x in candidates:
            if calls_made >= exploration_budget:
                break
            if time.time() > deadline:
                break
                
            fx = safe_func(x)
            if fx < current_best_f:
                current_best_f = fx
                current_best_x = x.copy()
                
        # Phase 2: Local Refinement (50% of budget)
        # Use (1+1)-ES starting from current_best_x
        # If current_best_x is None (shouldn't happen), start random.
        
        if current_best_x is None:
            x = rng.uniform(-5, 5, n)
            fx = safe_func(x)
        else:
            x = current_best_x.copy()
            fx = current_best_f
            
        sigma = 1.0 # Start with moderate step
        no_improve = 0
        max_no_improve = 20
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
            
            # Adaptive step size
            # If we improve, increase sigma slightly. If not, decrease.
            # Also, if stuck for a long time, try a larger jump (basin hopping).
            
            if no_improve > max_no_improve:
                # Large jump to escape local basin
                jump_sigma = 2.0
                y = np.clip(x + jump_sigma * rng.standard_normal(n), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma = 1.0
                    no_improve = 0
                else:
                    # If jump failed, reduce sigma and continue
                    sigma *= 0.5
                    no_improve = 0 # Reset counter to try smaller steps again
                    # If we are really stuck, maybe we should restart from global best?
                    # But we are already at global best from Phase 1.
                    # Let's just continue with smaller steps.
                    continue
            else:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
