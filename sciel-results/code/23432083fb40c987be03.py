"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        def elapsed_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Helper to safely evaluate and track calls
        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not elapsed_ok():
                return np.inf
            # Ensure x is finite and correct dimension
            if not np.all(np.isfinite(x)) or x.shape != (dim,):
                return np.inf
            # Clip to box to ensure finite input to func (though func clips, this is safer for 'finite vector' check)
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Nelder-Mead implementation
        def nelder_mead(func, x0, max_iter=100):
            nonlocal calls, best_x, best_f
            alpha = 1.0
            gamma = 2.0
            rho = -0.5
            sigma = 0.5
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                step = x0.copy()
                if step[i] != 0:
                    step[i] += 0.05 * abs(step[i])
                else:
                    step[i] += 0.05
                simplex.append(step)
            
            # Evaluate initial simplex
            fvals = [safe_eval(x) for x in simplex]
            
            for _ in range(max_iter):
                if calls >= budget or not elapsed_ok():
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if np.std(fvals) < 1e-8:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_eval(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + sigma * (simplex[-1] - centroid)
                    f_c = safe_eval(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_eval(simplex[i])
            
            return best_x, best_f

        # Multi-start strategy
        # Start 1: Center
        nelder_mead(func, np.zeros(dim), max_iter=80)
        
        # If we have budget left, try a few random starts or offsets
        if calls < budget and elapsed_ok():
            # Try starting from best_x with a perturbation
            x_start = best_x + np.random.randn(dim) * 0.5
            nelder_mead(func, x_start, max_iter=50)
            
        if calls < budget and elapsed_ok():
            # Try a random start
            x_start = np.random.uniform(-4, 4, size=dim)
            nelder_mead(func, x_start, max_iter=50)

        return best_x, best_f
# EVOLVE-BLOCK-END
