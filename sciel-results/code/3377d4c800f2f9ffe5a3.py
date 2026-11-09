"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        rng = np.random.default_rng(42)
        
        # Divide budget into multiple local searches
        n_searches = 10
        evals_per_search = budget // n_searches
        
        all_points = []
        
        for s in range(n_searches):
            if (time.time() - start_time) > deadline:
                break
                
            # Initialize simplex
            x0 = lo + (hi - lo) * rng.random(dim)
            step = 0.05 * (hi - lo)
            
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for j in range(1, dim + 1):
                simplex[j] = x0.copy()
                simplex[j][j-1] += step[j-1]
            simplex = np.clip(simplex, lo, hi)
            
            # Evaluate initial simplex
            f_vals = np.array([problem(v) for v in simplex])
            used = dim + 1
            
            # Nelder-Mead iterations
            while used < evals_per_search:
                if (time.time() - start_time) > deadline:
                    break
                    
                # Sort by sum of objectives
                order = np.argsort(np.sum(f_vals, axis=1))
                simplex = simplex[order]
                f_vals = f_vals[order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, lo, hi)
                
                f_ref = problem(x_ref)
                used += 1
                
                # Update simplex
                if np.sum(f_ref) < np.sum(f_vals[-1]):
                    # Better than worst
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                    # Check if better than second worst for expansion? 
                    # For simplicity, we just replace worst. 
                    # Expansion would require another eval.
                else:
                    # Worse than worst: Contract
                    gamma = 0.5
                    x_con = centroid + gamma * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = problem(x_con)
                    used += 1
                    
                    if np.sum(f_con) < np.sum(f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            # We don't re-evaluate shrunk points to save budget.
                            # This is a heuristic approximation.
                            # We break out of this local search if we shrink too much?
                            # Or just stop this local search.
                        # If we shrink, the simplex is very small. 
                        # Let's just stop this local search to save budget for next start.
                        break

            # Record the best point from this search
            best_idx = np.argmin(np.sum(f_vals, axis=1))
            all_points.append(simplex[best_idx])
            
        # Return the last best point (or any, since return is ignored)
        if all_points:
            return all_points[-1]
        else:
            return lo

# EVOLVE-BLOCK-END
