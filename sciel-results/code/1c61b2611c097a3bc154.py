"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start Nelder-Mead optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def timed_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            return f

        def time_left():
            return deadline - time.time()

        # Phase 1: Initial coarse search to find a good starting point
        # Sample 20 points to ensure we are in a reasonable basin
        n_init = 20
        for i in range(n_init):
            if calls >= budget or time_left() <= 10.0:
                break
            x = rng.uniform(-5, 5, dim)
            timed_call(x)
        
        if calls >= budget or time_left() <= 10.0:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead
        # We will run NM from the current best, and if it stagnates, 
        # we will perturb the best point and run NM again.
        
        max_nm_runs = 10
        max_nm_iterations = 200 # Cap iterations per run to allow restarts
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Initial simplex size
        delta = 0.5
        
        for run_idx in range(max_nm_runs):
            if calls >= budget or time_left() <= 5.0:
                break
            
            # Initialize simplex around current best
            x0 = np.array(best_x, dtype=float)
            simplex = [x0.copy()]
            
            # Add perturbed points
            for i in range(dim):
                x_pert = x0.copy()
                x_pert[i] += delta
                simplex.append(x_pert)
            
            # Evaluate initial simplex
            fs = [timed_call(x) for x in simplex]
            
            # If we already exceeded budget during init, break
            if calls >= budget or time_left() <= 5.0:
                break
            
            # Nelder-Mead loop
            for it in range(max_nm_iterations):
                if calls >= budget or time_left() <= 5.0:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check convergence: if the spread is tiny, break
                f_spread = fs[-1] - fs[0]
                if f_spread < 1e-8:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = timed_call(x_r)
                
                if calls >= budget or time_left() <= 5.0:
                    break
                
                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = timed_call(x_e)
                    
                    if calls >= budget or time_left() <= 5.0:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        # Outside contraction
                        x_c = centroid + sigma * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = timed_call(x_c)
                    
                    if calls >= budget or time_left() <= 5.0:
                        break
                    
                    if f_c <= min(f_r, fs[-1]):
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fs[i] = timed_call(simplex[i])
                        
                        if calls >= budget or time_left() <= 5.0:
                            break

            # After NM run, if we haven't improved significantly or budget remains, 
            # we could restart. But since we always update best_x inside timed_call,
            # the next run starts from the new best.
            
            # Reduce delta for subsequent runs if we are stuck? 
            # Actually, if we are stuck, increasing delta might help escape local minima.
            # But Schaffers F7 is ill-conditioned, not necessarily many local minima.
            # Let's keep delta constant or slightly increase to ensure exploration.
            delta *= 1.1

        return best_x, best_f
# EVOLVE-BLOCK-END
