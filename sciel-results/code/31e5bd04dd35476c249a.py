"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made
            val = func(x)
            calls_made += 1
            return val

        # Phase 1: Scout with random sampling to find a good basin
        scout_count = min(30, budget // 5)
        best_x = None
        best_f = np.inf
        scout_points = []

        for _ in range(scout_count):
            x = rng.uniform(-5, 5, n)
            f = eval_func(x)
            scout_points.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort scouts to keep top candidates
        scout_points.sort(key=lambda item: item[0])
        
        # Phase 2: Nelder-Mead refinement
        # We will try to refine the best scout, and if it converges quickly, 
        # we might use remaining budget to try the 2nd best, etc.
        
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        initial_step = 0.5  # Initial simplex size
        
        def nelder_mead(start_x, start_f):
            """
            Standard Nelder-Mead implementation.
            Uses eval_func which updates calls_made.
            Returns (best_x, best_f).
            """
            # Initialize simplex
            simplex = [start_x.copy()]
            simplef = [start_f]
            
            # We need to generate n more vertices.
            # Check if we have budget for them.
            for i in range(n):
                if calls_made >= budget:
                    break
                x_new = start_x.copy()
                x_new[i] += initial_step
                x_new = np.clip(x_new, -5, 5)
                f_new = eval_func(x_new)
                simplex.append(x_new)
                simplef.append(f_new)
            
            if len(simplex) < n + 1:
                # Not enough budget to form a simplex, return best so far
                idx = np.argmin(simplef)
                return simplex[idx], simplef[idx]

            while calls_made < budget:
                # Sort simplex by function value
                order = np.argsort(simplef)
                simplex = [simplex[i] for i in order]
                simplef = [simplef[i] for i in order]
                
                # Check convergence: spread of function values
                if abs(simplef[-1] - simplef[0]) < 1e-12:
                    break
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                if calls_made >= budget: break
                f_r = eval_func(x_r)
                
                if f_r < simplef[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    
                    if calls_made >= budget:
                        # Can't expand, accept reflection
                        simplex[-1] = x_r
                        simplef[-1] = f_r
                        break
                    
                    f_e = eval_func(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplef[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplef[-1] = f_r
                elif f_r < simplef[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplef[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if calls_made >= budget:
                        break
                    
                    f_c = eval_func(x_c)
                    
                    if f_c < simplef[-1]:
                        simplex[-1] = x_c
                        simplef[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if calls_made >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_s = eval_func(simplex[i])
                            simplef[i] = f_s
                        if calls_made >= budget:
                            break
                
                if calls_made >= budget:
                    break

            # Find best in simplex
            best_idx = np.argmin(simplef)
            return simplex[best_idx], simplef[best_idx]

        # Try refining the best scout
        if len(scout_points) > 0:
            best_f_scout, best_x_scout = scout_points[0]
            x_best, f_best = nelder_mead(best_x_scout, best_f_scout)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            # If we have significant budget left and a 2nd best scout, try it
            if calls_made < budget and len(scout_points) > 1:
                f_2nd, x_2nd = scout_points[1]
                # Only try if the 2nd best is reasonably close or if we have lots of budget
                # Simple heuristic: try it if we have > 10 evals left
                if budget - calls_made > 10:
                    x_2, f_2 = nelder_mead(x_2nd, f_2nd)
                    if f_2 < best_f:
                        best_f = f_2
                        best_x = x_2

        return best_x, best_f
# EVOLVE-BLOCK-END
