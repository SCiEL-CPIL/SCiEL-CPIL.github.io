"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead Simplex optimizer with multistart for BBOB-style Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        # Weierstrass functions are multimodal. 
        # We use a multistart strategy with Nelder-Mead.
        # Budget is 500. Dim is 5.
        # Simplex size is dim+1 = 6.
        # One NM run costs roughly 50-100 calls to converge or stagnate.
        # Let's try 5 restarts, allocating ~100 calls each.
        
        n_restarts = 5
        calls_per_restart = budget // n_restarts
        
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        total_calls = 0
        
        # Initial points: mix of center and random
        # To ensure coverage, we can use a grid or random points.
        # Given the shift/rotation, random is safer for global coverage.
        
        for r in range(n_restarts):
            if total_calls >= budget:
                break
            
            # Determine start point
            if r == 0:
                x0 = np.zeros(dim)
            else:
                # Random start within [-4, 4] to stay safe from boundaries initially?
                # Or [-5, 5]? The minimizer is in [-4, 4].
                # Let's sample from [-4, 4] to be closer to the optimum.
                x0 = rng.uniform(-4, 4, dim)
            
            # Initialize simplex
            # Standard Nelder-Mead initialization:
            # x0 is the first vertex.
            # Other vertices are x0 + delta * e_i
            delta = 0.5 # Initial step size
            
            simplex = [x0.copy()]
            fvals = [func(x0)]
            total_calls += 1
            if total_calls > budget:
                break
                
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += delta
                simplex.append(x_i)
                f_i = func(x_i)
                fvals.append(f_i)
                total_calls += 1
                if total_calls > budget:
                    break
            
            if total_calls > budget:
                break

            # Nelder-Mead Main Loop
            # We need to be careful with the number of calls.
            # Each iteration typically involves 1 function evaluation (reflection, expansion, etc.)
            # except when shrinking (which involves dim evaluations).
            
            max_iters = calls_per_restart - (dim + 1) # Remaining budget for this restart
            
            for _ in range(max_iters):
                if total_calls >= budget:
                    break
                
                # Sort simplex by function value
                # indices sorted by fvals
                sorted_idx = np.argsort(fvals)
                simplex = [simplex[i] for i in sorted_idx]
                fvals = [fvals[i] for i in sorted_idx]
                
                # Check for convergence
                # If the difference between best and worst is small, stop
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                
                # Calculate centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                gamma = 1.0
                x_ref = centroid + gamma * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                
                f_ref = func(x_ref)
                total_calls += 1
                if total_calls > budget:
                    break
                
                # Update simplex
                if f_ref < fvals[0]:
                    # Expansion
                    if total_calls < budget:
                        alpha = 2.0
                        x_exp = centroid + alpha * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        total_calls += 1
                        if total_calls > budget:
                            break
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Just replace worst with reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if total_calls < budget:
                        beta = 0.5
                        x_con = centroid + beta * (x_worst - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        total_calls += 1
                        if total_calls > budget:
                            break
                        
                        if f_con < fvals[-1]:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            # This is expensive: dim evaluations
                            # If we don't have enough budget for dim evals, we might just stop
                            # or do a partial shrink? Standard NM shrinks all points except best.
                            # Let's check if we have enough budget for dim calls.
                            if total_calls + dim <= budget:
                                for i in range(1, dim + 1):
                                    x_shrink = x_best + 0.5 * (simplex[i] - x_best)
                                    x_shrink = np.clip(x_shrink, -5, 5)
                                    f_shrink = func(x_shrink)
                                    total_calls += 1
                                    simplex[i] = x_shrink
                                    fvals[i] = f_shrink
                                    if total_calls > budget:
                                        break
                            else:
                                # Not enough budget for full shrink, break
                                break
                    else:
                        break
                
                # Update global best
                if fvals[0] < global_best_f:
                    global_best_f = fvals[0]
                    global_best_x = simplex[0].copy()
            
            # Update global best after each restart loop finishes (or if broken)
            if fvals[0] < global_best_f:
                global_best_f = fvals[0]
                global_best_x = simplex[0].copy()
                
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
