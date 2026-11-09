"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        remaining = self.budget

        # Strategy: Multi-start Nelder-Mead
        # We will perform several independent Nelder-Mead runs.
        # For dim=5, Nelder-Mead is efficient.
        # Let's allocate the budget to 10 restarts, each getting 50 calls.
        # If a run finishes early (convergence), we can reallocate or just stop.
        # To be safe and simple, we fix the number of restarts and sub-budgets.
        
        n_restarts = 10
        sub_budget = self.budget // n_restarts
        # Adjust last restart to use remaining budget
        sub_budgets = [sub_budget] * (n_restarts - 1)
        sub_budgets.append(self.budget - sum(sub_budgets))

        # Generate diverse starting points
        # Lunacek has a bi-modal landscape.
        # One mode is often near the center, another near the boundaries.
        # We'll mix random points with some structured ones.
        
        starts = []
        # 1. Origin
        starts.append(np.zeros(self.dim))
        # 2. Random points
        for _ in range(n_restarts - 1):
            x = rng.uniform(-5, 5, self.dim)
            starts.append(x)

        for i, start_x in enumerate(starts):
            if remaining <= 0:
                break
            
            b = sub_budgets[i]
            if b < 10:
                b = max(10, remaining) # Ensure at least 10 calls if possible
            if b > remaining:
                b = remaining
                
            x_opt, f_opt, calls_used = self._nelder_mead(func, start_x, b, rng)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
                
            remaining -= calls_used

        # Fallback if no valid point was found (should not happen)
        if best_x is None:
            x = rng.uniform(-5, 5, self.dim)
            best_f = func(x)
            best_x = x

        return best_x, best_f

    def _nelder_mead(self, func, x0, max_calls, rng):
        dim = self.dim
        x = np.clip(x0, -5, 5).copy()
        
        # Initial Simplex
        # Standard Nelder-Mead: x0, x0 + delta*e_i
        delta = 0.5 # Initial step size
        simplex = [x.copy()]
        for i in range(dim):
            xi = x.copy()
            xi[i] += delta
            simplex.append(xi)
            
        # Evaluate initial simplex
        fvals = []
        calls = 0
        for i in range(dim + 1):
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals.append(func(simplex[i]))
            calls += 1
            
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        best_x = simplex[0].copy()
        best_f = fvals[0]
        
        # Nelder-Mead Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        while calls < max_calls:
            # Check for convergence
            # If the spread of function values is very small, stop
            if max(fvals) - min(fvals) < 1e-8:
                break
            
            # 1. Reflection
            x_worst = simplex[-1]
            x_centroid = np.mean(simplex[:-1], axis=0)
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                # 2. Expansion
                x_exp = x_centroid + gamma * (x_ref - x_centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Reflection is good, but not best
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # 3. Contraction
                # Check if we should do inside or outside contraction
                # Standard NM often does inside contraction first
                x_con = x_centroid + rho * (x_worst - x_centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # 4. Shrink
                    # Shrink all points except the best one towards the best point
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= max_calls:
                            break
                    
                    if calls >= max_calls:
                        break
            
            # Update best
            if fvals[-1] < best_f:
                best_f = fvals[-1]
                best_x = simplex[-1].copy()
                
            # Sort simplex again
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update global best within this run
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        return best_x, best_f, calls
# EVOLVE-BLOCK-END
