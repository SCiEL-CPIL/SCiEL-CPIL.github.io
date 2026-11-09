"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structured Grid Search Optimizer for WFG9.
    Exploits the known structure of WFG9 where the Pareto front lies on a manifold
    determined by the first few variables, while the rest are fixed to specific values.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0 

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 Structure:
        # The Pareto front is typically defined by the first M-1 or M variables.
        # For M=3, x1, x2 (and sometimes x3) determine the front shape/position.
        # Variables x4...x12 are "traps" and should be fixed to values that minimize
        # the distance to the ideal point, which is often 0 or the midpoint.
        # We will fix x3...x12 to the midpoint of their bounds (or 0 if bounds are symmetric around 0, 
        # but midpoint is safer generally). 
        # Actually, for WFG, the "non-separability" is often minimized when variables are at specific values.
        # Let's try fixing x3...x12 to the midpoint.
        
        mid = (lo + hi) / 2.0
        
        # We will perform a grid search on x1 and x2.
        # Budget is 200. 
        # A 14x14 grid = 196 points. This leaves 4 points for potential refinement or checking x3.
        # Let's use a 14x14 grid on x1, x2.
        
        n_grid = 14
        points = []
        
        # Generate grid points for x1, x2
        x1_vals = np.linspace(lo[0], hi[0], n_grid)
        x2_vals = np.linspace(lo[1], hi[1], n_grid)
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                x = mid.copy()
                x[0] = x1
                x[1] = x2
                # x[2] is kept at mid[2] for now. 
                # If the front depends on x3, we might miss the exact PF, but x1,x2 variation is primary.
                # To be safer, we could also vary x3 slightly, but 14x14x2 is 392 > 200.
                # So we stick to x1, x2 variation with x3...x12 fixed.
                points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # If we have budget left, we can try to refine the best points or check if x3 matters.
        # However, the grid search on the main variables is the primary structural improvement.
        # We can use the remaining budget to do a small local search around the best points.
        
        # Identify non-dominated points in the grid to guide local search
        # Simple dominance check
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        
        # Select best 4 non-dominated points for local refinement
        nd_indices = [i for i in range(n) if not is_dominated[i]]
        
        # If we have budget remaining, do a small random perturbation around the best points
        # to ensure we don't miss the exact PF if the fixed values were slightly off.
        remaining_budget = self.budget - len(points)
        if remaining_budget > 0 and nd_indices:
            # Pick the top 2 non-dominated points
            # Sort by sum of objectives (simple proxy)
            scores = np.sum(F[nd_indices], axis=1)
            best_idx_in_nd = np.argsort(scores)[:2]
            best_indices = [nd_indices[i] for i in best_idx_in_nd]
            
            # Perform a few local perturbations
            rng = np.random.default_rng(0)
            perturbations = 4
            for _ in range(min(perturbations, remaining_budget)):
                base_idx = best_indices[0] if len(best_indices) > 0 else 0
                base_x = points[base_idx].copy()
                # Small perturbation in x1, x2, x3
                for k in range(3):
                    base_x[k] += rng.uniform(-0.05, 0.05) * (hi[k] - lo[k])
                base_x = np.clip(base_x, lo, hi)
                points.append(base_x)
                F = np.vstack([F, [problem(base_x)]])
        
        return points
# EVOLVE-BLOCK-END
