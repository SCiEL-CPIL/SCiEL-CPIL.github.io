"""Rewritten optimizer using a two-phase strategy: deterministic grid scan + local refinement."""
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse grid scan to find global basin, then local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Phase 1: Coarse deterministic grid scan
        # Use a small grid to identify promising regions
        # 3 points per dimension: -4, 0, 4 (within [-5,5], covering most of the space)
        grid_points = list(np.meshgrid(
            [-4.0, 0.0, 4.0] * self.dim
        ))
        # Flatten the meshgrid
        candidates = np.array([np.array(p) for p in zip(*grid_points)])
        
        # Shuffle with a fixed seed for reproducibility but add some diversity
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(candidates))
        candidates = candidates[perm]
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate grid points, stopping if we run low on budget
        for x in candidates:
            if calls_made >= self.budget - 50:  # reserve 50 for local refinement
                break
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement around the best point found
        # Use a local search with decreasing step sizes
        if best_x is not None and calls_made < self.budget:
            # Try multiple local refinement strategies
            x = best_x.copy()
            fx = best_f
            
            # Strategy: adaptive step local search
            step_size = 1.0
            no_improve = 0
            max_no_improve = 10
            
            while calls_made < self.budget and time.time() < deadline:
                # Generate neighbors in each coordinate direction
                improved = False
                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    # Try positive and negative directions
                    for sign in [1.0, -1.0]:
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        
                        y = x.copy()
                        y[d] += sign * step_size
                        y = np.clip(y, -5, 5)
                        
                        fy = func(y)
                        calls_made += 1
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            no_improve = 0
                            step_size *= 1.2  # increase step when improving
                        else:
                            no_improve += 1
                
                if not improved:
                    step_size *= 0.7  # decrease step when stuck
                    if step_size < 0.01:
                        break
                else:
                    # Reset no_improve counter after improvement
                    pass
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 3: If we still have budget, do a finer local search
        if best_x is not None and calls_made < self.budget:
            x = best_x.copy()
            fx = best_f
            
            # Fine-tuning with smaller steps
            step_size = 0.1
            for _ in range(20):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                improved = False
                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    for sign in [1.0, -1.0]:
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        
                        y = x.copy()
                        y[d] += sign * step_size
                        y = np.clip(y, -5, 5)
                        
                        fy = func(y)
                        calls_made += 1
                        
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                
                if not improved:
                    step_size *= 0.8
                    if step_size < 0.001:
                        break
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
