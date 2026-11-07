"""Multi-Start Nelder-Mead with Sobol Global Seeding for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for BBOB f21 (Gallagher 101 peaks).
    Strategy:
    1. Global Exploration: Use a Sobol sequence to sample the domain [-5, 5]^d.
    2. Local Refinement: Identify the best candidates and refine them using Nelder-Mead.
    3. Iterative Process: Repeat exploration and refinement until budget is exhausted.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # State
        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time_left() < 0.1:
                return None
            # Ensure x is a valid numpy array within bounds
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            
            try:
                f = func(x)
            except Exception:
                return np.inf
            
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Global Sobol Seeding ---
        # We want to cover the space well.
        # f21 has 101 peaks. We should sample at least a few hundred points.
        # Let's allocate ~20% of the budget to initial global seeding.
        # But we also need to leave room for local refinement.
        
        # Sobol sequence
        sobol = qmc.Sobol(d=dim, scramble=True)
        
        # Determine number of points for global phase.
        # Let's start with 100 points.
        n_global = 100
        # Ensure power of 2 for Sobol efficiency, though scipy handles non-powers
        # but powers are standard. 128 is good.
        n_global = 128
        
        # Generate points in [0, 1]^d
        points_unit = sobol.random(n=n_global)
        # Map to [-5, 5]^d
        points = points_unit * 10.0 - 5.0
        
        # Evaluate all global points
        candidates = []
        for p in points:
            f = make_call(p)
            if f is None:
                break
            candidates.append((f, p))
            
        if time_left() < 1.0 or calls_made >= budget:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement with Nelder-Mead ---
        # We will pick the top K candidates and run Nelder-Mead on them.
        # Nelder-Mead is very fast in 5D.
        # Each NM run takes roughly (dim+1) * iterations calls.
        # Let's say we allow 200 calls per NM run.
        # If we have ~400 calls left, we can run 2 NM instances.
        # Let's dynamically determine how many NM runs we can afford.
        
        def nelder_mead(func, x0, max_iter=100, tol=1e-8):
            """
            Custom Nelder-Mead implementation to strictly control function calls and time.
            """
            dim = len(x0)
            # Initial simplex
            simplex = [x0.copy() for _ in range(dim + 1)]
            # Perturb vertices
            for i in range(1, dim + 1):
                simplex[i][i-1] += 0.05  # Small step
            
            # Evaluate initial simplex
            values = [make_call(simplex[i]) for i in range(dim + 1)]
            
            if any(v is None for v in values):
                return None, None # Out of budget or time

            # Sort simplex
            def sort_simplex():
                nonlocal simplex, values
                idx = np.argsort(values)
                simplex = [simplex[i] for i in idx]
                values = [values[i] for i in idx]

            sort_simplex()
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if time_left() < 0.5:
                    break
                if calls_made >= budget:
                    break
                
                # Check convergence
                if abs(values[-1] - values[0]) < tol:
                    break
                
                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = make_call(x_r)
                if f_r is None:
                    break
                
                if f_r < values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = make_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        values[-1] = f_r
                elif f_r < values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    values[-1] = f_r
                else:
                    # Contraction
                    if f_r < values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = make_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < min(f_r, values[-1]):
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            x_sh = np.clip(simplex[i], -5.0, 5.0)
                            f_sh = make_call(x_sh)
                            if f_sh is None:
                                return simplex[0], values[0]
                            values[i] = f_sh
                        
                        sort_simplex()
                        continue # Skip normal update step after shrink
                
                sort_simplex()
            
            return simplex[0], values[0]

        # Determine number of local runs
        # We have `candidates`. Let's take the top `n_local` ones.
        # Each NM run might use up to ~100-200 calls.
        # Let's estimate remaining budget.
        remaining = budget - calls_made
        # Assume ~150 calls per NM run on average.
        n_local = max(1, remaining // 150)
        
        # Also limit n_local to not exceed number of candidates
        n_local = min(n_local, len(candidates))
        
        # It's often better to run fewer, deeper searches if time permits, 
        # but for AOCC, we want to keep finding better points early.
        # Let's process them sequentially.
        
        for i in range(n_local):
            if time_left() < 1.0 or calls_made >= budget:
                break
            
            # Start from the i-th best candidate
            x_start = candidates[i][1]
            
            # Run Nelder-Mead
            # We pass a wrapper that uses our make_call
            x_opt, f_opt = nelder_mead(func, x_start, max_iter=200)
            
            # Update global best if NM found something better
            # (make_call already updates best_x/best_f, so we don't strictly need to do it here,
            #  but it's good to be explicit if we were tracking locally)
            
            # Optional: If NM converged to a very good point, we might want to 
            # restart NM from there with tighter tolerance? 
            # For now, sequential processing is sufficient.
            
        # If we still have budget and time, do a final random polish?
        # Or just return.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
