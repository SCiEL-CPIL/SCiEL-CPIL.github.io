"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        n = self.dim
        remaining = self.budget
        best_x = None
        best_f = float('inf')

        def time_ok():
            return time.time() < deadline - 5.0

        # Phase 1: Broad exploration with structured + random points
        rng = np.random.default_rng(12345)
        
        # Create diverse initial points
        init_points = []
        # Center
        init_points.append(np.zeros(n))
        # Axis-aligned points
        for i in range(n):
            for val in [-3.0, 0.0, 3.0]:
                p = np.zeros(n)
                p[i] = val
                init_points.append(p)
        # Random points
        for _ in range(20):
            init_points.append(rng.uniform(-4.5, 4.5, n))
        
        # Evaluate all initial points
        eval_results = []
        for x0 in init_points:
            if remaining <= 0 or not time_ok():
                break
            fx0 = func(x0)
            remaining -= 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            eval_results.append((fx0, x0.copy()))
        
        # Sort by function value
        eval_results.sort(key=lambda t: t[0])
        
        # Phase 2: Local refinement from top starts using adaptive CMA-ES
        # Pick top candidates for local optimization
        top_k = min(8, len(eval_results))
        for idx in range(top_k):
            if remaining < 30 or not time_ok():
                break
            
            x0 = eval_results[idx][1].copy()
            fx0 = eval_results[idx][0]
            
            # Adaptive step size local search
            sigma = 0.5
            x = x0.copy()
            fx = fx0
            no_improve_count = 0
            max_no_improve = 15
            
            while remaining > 10 and time_ok() and no_improve_count < max_no_improve:
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve_count = 0
                    sigma *= 1.2  # increase step on success
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve_count += 1
                    sigma *= 0.8  # decrease step on failure
            
            # If we found a good point, try to refine further with smaller steps
            if fx < best_f * 0.9 or (best_x is not None and np.linalg.norm(x - best_x) < 1.0):
                sigma = 0.1
                no_improve_count = 0
                max_no_improve = 10
                while remaining > 5 and time_ok() and no_improve_count < max_no_improve:
                    y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    remaining -= 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        no_improve_count = 0
                        sigma *= 1.1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        no_improve_count += 1
                        sigma *= 0.7
        
        # Phase 3: Final fine-tuning around the best point
        if best_x is not None and remaining > 10 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve_count = 0
            max_no_improve = 20
            
            while remaining > 5 and time_ok() and no_improve_count < max_no_improve:
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve_count = 0
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve_count += 1
                    sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
