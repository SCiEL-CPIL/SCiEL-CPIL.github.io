"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        # We will use multiple restarts.
        # For dim=5, Nelder-Mead uses a simplex of size 6.
        # Each step costs ~dim+1 evaluations on average (reflection, expansion, etc.)
        # Let's allocate budget to a few independent runs.
        num_restarts = 10
        budget_per_restart = max(10, total_budget // num_restarts)

        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize a random point
            x0 = rng.uniform(-5, 5, dim)
            
            # Create initial simplex
            # Standard Nelder-Mead initialization: x0 and x0 + delta * e_i
            delta = 0.5
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += delta
            
            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                if time.time() > deadline:
                    break
                fvals[i] = float(func(simplex[i]))
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrinkage
            tol = 1e-8     # Convergence tolerance
            
            # Run Nelder-Mead for this restart
            # Limit steps to ensure we don't exceed time/budget
            max_steps = budget_per_restart // (dim + 1) + 5
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check convergence
                if np.std(fvals) < tol:
                    break
                
                # Centroid of all points except the worst
                x_c = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                
                # Reflection
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = float(func(x_r))
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = float(func(x_e))
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = x_c + 0.5 * (x_r - x_c)
                    else:
                        # Inside contraction
                        x_c = x_c - 0.5 * (x_worst - x_c)
                    
                    x_cc = np.clip(x_c, -5, 5)
                    f_cc = float(func(x_cc))
                    calls_made += 1
                    if f_cc < best_f:
                        best_f = f_cc
                        best_x = x_cc.copy()

                    if f_cc < min(f_r, fvals[-1]):
                        simplex[-1] = x_cc
                        fvals[-1] = f_cc
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = float(func(simplex[i]))
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        # Re-sort not strictly necessary for shrink step logic if we break or continue, 
                        # but good practice. However, to save calls, we just continue.
                        # We break out of this restart if shrink happens repeatedly? 
                        # No, just let it run.
                        
                        # If we used too many calls, break
                        if calls_made >= total_budget:
                            break

            if calls_made >= total_budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
