"""Multi-start Nelder-Mead ensemble with adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start Nelder-Mead ensemble optimizer.
    Launches multiple independent simplexes from diverse initial points
    to handle hidden shifts and maximize early progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # --- Phase 1: Multi-Start Nelder-Mead Ensemble ---
        # We will run multiple Nelder-Mead instances in a round-robin fashion.
        # Each instance starts from a different point.
        # We allocate a portion of the budget to "setup" (evaluating initial points)
        # and the rest to the actual optimization loops.
        
        # Number of starts: let's try 5 diverse starts.
        num_starts = 5
        
        # Generate initial points using a Latin Hypercube-like strategy for diversity
        # We'll use a simple stratified sampling approach.
        # Divide [-5, 5] into num_starts bins for each dimension? 
        # Actually, for 5D, let's just use random points from a set of well-spaced candidates.
        # Or simply random points from [-4, 4] to stay away from boundaries initially.
        
        starts = []
        for i in range(num_starts):
            # Ensure diversity: mix center and random
            if i == 0:
                x_start = np.zeros(dim)
            else:
                x_start = rng.uniform(-4, 4, dim)
            starts.append(x_start)
        
        # Evaluate initial points
        simplexes = []
        f_simplexes = []
        
        for x0 in starts:
            if not budget_ok() or not time_ok():
                break
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            
            # Build initial simplex for this start
            alpha = 0.5  # Initial step size for simplex
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            for i in range(dim):
                if not budget_ok() or not time_ok():
                    break
                x_i = x0.copy()
                # Perturb dimension i
                # Use a step that keeps us in bounds
                if x0[i] < 4.0:
                    x_i[i] += alpha
                else:
                    x_i[i] -= alpha
                x_i = np.clip(x_i, -5, 5)
                
                f_i = func(x_i)
                calls += 1
                record(x_i, f_i)
                
                simplex.append(x_i)
                f_simplex.append(f_i)
            
            if len(simplex) == dim + 1:
                simplexes.append(simplex)
                f_simplexes.append(f_simplex)
        
        if not simplexes:
            return best_x, best_f

        # Nelder-Mead parameters
        gamma = 1.0   # Reflection
        rho = 2.0     # Expansion
        sigma_nm = 0.5 # Contraction
        shrink = 0.5

        # Round-robin optimization
        # We will iterate through the active simplexes and perform one NM step on each
        # until budget or time is exhausted.
        
        active_indices = list(range(len(simplexes)))
        
        while budget_ok() and time_ok() and active_indices:
            next_active = []
            for idx in active_indices:
                if not budget_ok() or not time_ok():
                    break
                
                simplex = simplexes[idx]
                f_simplex = f_simplexes[idx]
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                
                if not budget_ok() or not time_ok():
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)
                
                converged = False
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if not budget_ok() or not time_ok():
                        break
                    f_e = func(x_e)
                    calls += 1
                    record(x_e, f_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                        
                elif f_r < f_second_worst:
                    # Reflection accepted
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if not budget_ok() or not time_ok():
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not budget_ok() or not time_ok():
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            record(simplex[i], f_simplex[i])
                        
                        if not budget_ok() or not time_ok():
                            break
                        
                        # Check for convergence after shrink
                        f_spread = np.max(f_simplex) - np.min(f_simplex)
                        if f_spread < 1e-10:
                            converged = True

                # Check for convergence before moving to next
                f_spread = np.max(f_simplex) - np.min(f_simplex)
                if f_spread < 1e-10:
                    converged = True
                
                # Update simplex in list
                simplexes[idx] = simplex
                f_simplexes[idx] = f_simplex
                
                # If not converged, keep it active
                if not converged:
                    next_active.append(idx)
            
            active_indices = next_active

        return best_x, best_f
# EVOLVE-BLOCK-END
