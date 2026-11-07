"""Multi-objective optimizer for RE21 with budget 200, dim 4, n_obj 2."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using Sobol sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for the Sobol generator to ensure reproducibility
        # Sobol requires dimension to be a power of 2? No, it works for any, but performance is best for powers of 2.
        # 4 is a power of 2.
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        
        # Phase 1: Global Sampling
        n_global = 120
        # Generate Sobol points in [0, 1]^d
        points_unit = sampler.random(n_global)
        
        # Scale to bounds
        X_global = lo + points_unit * (hi - lo)
        
        F_global = np.array([problem(x) for x in X_global])
        used = n_global
        
        # Phase 2: Local Refinement
        # Find non-dominated points
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(points):
            n = len(points)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(points[j], points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Get top non-dominated points for refinement
        # We want to refine the most "promising" points. 
        # Hypervolume contribution is a good metric, but expensive.
        # Simpler: just take the first 10 non-dominated points.
        
        front_indices = non_dominated_front(F_global)
        
        # If we have time and budget, refine the best few
        n_refine = 10
        if len(front_indices) > 0:
            # Sort front indices by sum of objectives (lower is better) to pick "best" ones
            front_points = F_global[front_indices]
            front_X = X_global[front_indices]
            
            # Sort by sum of objectives
            order = np.argsort(np.sum(front_points, axis=1))
            # Take top n_refine
            refine_indices = front_indices[order[:n_refine]]
            
            # Local search: random walks with decreasing step size
            # Remaining budget
            remaining_budget = budget - used
            
            # We have 10 points to refine. 
            # Let's do 5 steps per point. 10 * 5 = 50 evaluations.
            steps_per_point = max(1, remaining_budget // n_refine)
            
            for idx in refine_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                current_x = X_global[idx]
                current_f = F_global[idx]
                
                # Simple random walk with shrinking step
                # Start with step size 0.1 of range, shrink by 0.5 each step
                step_scale = 0.1
                for _ in range(steps_per_point):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Generate random perturbation
                    perturbation = np.random.randn(dim) * step_scale * (hi - lo)
                    new_x = current_x + perturbation
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    used += 1
                    
                    # Update if it dominates or is non-dominated and better in HV proxy
                    # Simple check: if new_f dominates current_f, or if they are non-dominated and new_f has lower sum
                    if dominates(new_f, current_f):
                        current_x = new_x
                        current_f = new_f
                        # Update the main population
                        X_global[idx] = current_x
                        F_global[idx] = current_f
                    elif not dominates(current_f, new_f):
                        # They are non-dominated. 
                        # Keep the one with lower sum of objectives as a heuristic for "better" local point
                        if np.sum(new_f) < np.sum(current_f):
                            current_x = new_x
                            current_f = new_f
                            X_global[idx] = current_x
                            F_global[idx] = current_f
                    
                    step_scale *= 0.7

        # Combine all evaluated points (Global + Refined)
        # Note: The refined points are already in X_global/F_global
        # We just need to return the set of all unique points? 
        # The evaluator records every returned objective vector.
        # We should return the non-dominated set from the final population.
        
        final_front = non_dominated_front(F_global)
        
        return X_global[final_front]

# EVOLVE-BLOCK-END
