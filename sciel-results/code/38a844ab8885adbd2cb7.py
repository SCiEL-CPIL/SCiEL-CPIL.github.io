"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid search exploiting Katsuura's periodic structure."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        dim = self.dim
        budget = self.budget

        def evaluate_grid(lower, upper, n_per_dim):
            """Evaluate a grid over [lower, upper]^dim with n_per_dim points per dimension."""
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline - 5:
                return
            
            # Generate grid points
            coords = []
            for i in range(dim):
                if n_per_dim == 1:
                    coords.append([(lower[i] + upper[i]) / 2.0])
                else:
                    coords.append(np.linspace(lower[i], upper[i], n_per_dim))
            
            # Evaluate all combinations
            grid_points = []
            for idx in np.ndindex(*([n_per_dim] * dim)):
                x = np.array([coords[d][idx[d]] for d in range(dim)], dtype=float)
                x = np.clip(x, -5, 5)
                grid_points.append(x)
            
            results = []
            for x in grid_points:
                if calls_made >= budget or time.time() > deadline - 5:
                    break
                try:
                    f = counted_func(x)
                    results.append((f, x))
                except RuntimeError:
                    break
            
            return results

        def hierarchical_search():
            """Hierarchical grid refinement search."""
            nonlocal calls_made, best_f, best_x
            
            # Phase 1: Coarse grid over entire domain
            # Katsuura has period ~0.5 in each dimension, so use 3 points per dim initially
            lower = np.full(dim, -5.0)
            upper = np.full(dim, 5.0)
            
            # Start with a coarse 3^5 = 243 point grid
            results = evaluate_grid(lower, upper, 3)
            if not results:
                return
            
            # Find the best sub-region (1/3 of each dimension around best point)
            best_f_phase1, best_x_phase1 = min(results, key=lambda t: t[0])
            
            # Phase 2: Refine around best with 5^5 = 3125 points in a smaller box
            # Use a box of width 2.0 (one period) centered on best
            half_width = 1.0
            lower2 = np.clip(best_x_phase1 - half_width, -5, 5)
            upper2 = np.clip(best_x_phase1 + half_width, -5, 5)
            
            results2 = evaluate_grid(lower2, upper2, 5)
            if results2:
                best_f_phase2, best_x_phase2 = min(results2, key=lambda t: t[0])
                if best_f_phase2 < best_f:
                    best_f = best_f_phase2
                    best_x = best_x_phase2
            else:
                best_f_phase2, best_x_phase2 = best_f_phase1, best_x_phase1
            
            # Phase 3: Further refine with 7^5 = 16807 points in an even smaller box
            # Use a box of width 0.5 (half period) centered on current best
            half_width2 = 0.25
            lower3 = np.clip(best_x_phase2 - half_width2, -5, 5)
            upper3 = np.clip(best_x_phase2 + half_width2, -5, 5)
            
            # Check if we have enough budget for phase 3
            remaining = budget - calls_made
            if remaining > 100 and time.time() < deadline - 30:
                results3 = evaluate_grid(lower3, upper3, 7)
                if results3:
                    best_f_phase3, best_x_phase3 = min(results3, key=lambda t: t[0])
                    if best_f_phase3 < best_f:
                        best_f = best_f_phase3
                        best_x = best_x_phase3
            
            # Phase 4: Final local polish using fine grid around very best
            if calls_made < budget and time.time() < deadline - 10:
                half_width3 = 0.05
                lower4 = np.clip(best_x - half_width3, -5, 5)
                upper4 = np.clip(best_x + half_width3, -5, 5)
                
                results4 = evaluate_grid(lower4, upper4, 5)
                if results4:
                    best_f_phase4, best_x_phase4 = min(results4, key=lambda t: t[0])
                    if best_f_phase4 < best_f:
                        best_f = best_f_phase4
                        best_x = best_x_phase4
        
        hierarchical_search()

        return best_x, best_f
# EVOLVE-BLOCK-END
