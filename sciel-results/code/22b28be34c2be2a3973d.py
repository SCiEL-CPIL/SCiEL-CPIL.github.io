"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_fx = float('inf')
        
        def wrapped_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        def nelder_mead(x0, budget_limit):
            """
            Runs Nelder-Mead starting from x0 until budget_limit is reached.
            Returns best x, best fx, and number of calls used.
            """
            nonlocal best_x, best_fx
            n = len(x0)
            
            # Initial simplex
            delta = 0.5
            simplex = []
            values = []
            
            # Center
            p0 = x0.copy()
            simplex.append(p0)
            values.append(wrapped_func(p0))
            
            # Vertices
            for i in range(n):
                p = x0.copy()
                p[i] += delta
                simplex.append(p)
                values.append(wrapped_func(p))
            
            # One more point for n+1 simplex
            p = x0.copy()
            p[:] -= delta
            simplex.append(p)
            values.append(wrapped_func(p))
            
            # Update global best
            min_idx = np.argmin(values)
            if values[min_idx] < best_fx:
                best_fx = values[min_idx]
                best_x = simplex[min_idx].copy()
                
            calls_used = len(values)
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls < total_budget and calls_used < budget_limit:
                # Sort
                order = np.argsort(values)
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_second_worst = simplex[-2]
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = wrapped_func(x_r)
                calls_used += 1
                
                if f_r < values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = wrapped_func(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    values[-1] = f_r
                else:
                    # Contraction
                    if f_r < values[-1]:
                        x_c = centroid + rho * (x_worst - centroid)
                    else:
                        x_c = centroid - rho * (centroid - x_worst)
                    
                    f_c = wrapped_func(x_c)
                    calls_used += 1
                    
                    if f_c < min(f_r, values[-1]):
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            values[i] = wrapped_func(simplex[i])
                            calls_used += 1
                            if calls >= total_budget or calls_used >= budget_limit:
                                break
                        
                        if calls >= total_budget or calls_used >= budget_limit:
                            break
                
                # Update global best
                min_val = min(values)
                if min_val < best_fx:
                    best_fx = min_val
                    best_x = simplex[np.argmin(values)].copy()
                    
                # Convergence
                if max(values) - min(values) < 1e-9:
                    break
            
            return best_x, best_fx, calls_used

        # Phase 1: Exploration
        # 5 restarts, 50 evals each
        num_restarts = 5
        budget_per_restart = 50
        
        for i in range(num_restarts):
            if calls >= total_budget:
                break
            
            x_start = rng.uniform(-4, 4, n)
            nelder_mead(x_start, budget_per_restart)
            
        # Phase 2: Exploitation
        if calls < total_budget and best_x is not None:
            # Run a longer NM from the best point found
            remaining_budget = total_budget - calls
            nelder_mead(best_x, remaining_budget)
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
