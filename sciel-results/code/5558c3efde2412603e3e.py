"""Optimized search for BBOB f13 Sharp Ridge."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Initialize simplex
        # Start with a reasonable spread
        x0 = rng.uniform(-3, 3, dim)
        simplex = [x0.copy() for _ in range(dim + 1)]
        for i in range(1, dim + 1):
            simplex[i] = simplex[0].copy()
            simplex[i][i - 1] += 0.5  # Initial step size 0.5

        fvals = np.array([safe_call(x) for x in simplex])
        
        # Update best
        idx_min = np.argmin(fvals)
        best_f = fvals[idx_min]
        best_x = simplex[idx_min].copy()

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        # Stagnation counter
        no_improve_count = 0
        max_no_improve = 50

        while calls[0] < budget and time.time() < deadline:
            if time.time() > deadline - 5:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]

            # Check for convergence/stagnation
            if fvals[-1] - fvals[0] < 1e-8:
                break
            
            # If we haven't improved in a while, restart
            if no_improve_count > max_no_improve:
                # Random restart around best point
                new_x0 = best_x + rng.normal(0, 1.0, dim)
                new_x0 = np.clip(new_x0, -5.0, 5.0)
                simplex = [new_x0.copy() for _ in range(dim + 1)]
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0].copy()
                    simplex[i][i - 1] += 1.0 # Larger initial step for restart
                fvals = np.array([safe_call(x) for x in simplex])
                
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()
                    no_improve_count = 0
                else:
                    # If restart didn't help immediately, continue but reset counter slightly
                    no_improve_count = 0 
                    # Re-sort
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = fvals[order]
                continue

            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = safe_call(x_ref)

            if fvals[0] <= f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
                no_improve_count += 1
            elif f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = safe_call(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                no_improve_count += 1
            else:
                # Contraction
                # Check if inside or outside contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = safe_call(x_con)

                if f_con < min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                    no_improve_count += 1
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = safe_call(simplex[i])
                    
                    # Re-sort
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = fvals[order]
                    no_improve_count += 1

            # Update global best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = simplex[idx_min].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

        return best_x, best_f
