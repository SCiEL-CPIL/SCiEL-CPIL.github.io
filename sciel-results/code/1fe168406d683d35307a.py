"""Rewritten optimizer for BBOB f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer:
      1. CMA-ES-like population-based search for global exploration on Rastrigin.
      2. Switch to Nelder-Mead local refinement when close to optimum.
    Includes a wall-clock deadline guard (300s limit, return best ~10s before).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = x_clip.copy()
            return f_val

        # Initialize 5 simplexes at different locations to explore basins
        # Rastrigin has local minima at integers, so we start near integers
        start_points = [
            np.zeros(dim), # Center
            np.ones(dim) * 1.0, # Near (1,1,1,1,1)
            np.ones(dim) * -1.0, # Near (-1,-1,-1,-1,-1)
            np.ones(dim) * 2.0, # Near (2,2,2,2,2)
            np.ones(dim) * -2.0 # Near (-2,-2,-2,-2,-2)
        ]

        simplexes = []
        simplex_fs = []

        # Initialize simplexes
        for sp in start_points:
            simplex = [sp.copy()]
            fs = []
            for i in range(dim):
                xi = sp.copy()
                xi[i] += 0.5 # Initial step size
                simplex.append(xi)
            
            # Evaluate initial simplex
            for s in simplex:
                f_val = safe_call(s)
                if f_val is None:
                    return best_x, best_fx
                fs.append(f_val)
            
            simplexes.append(simplex)
            simplex_fs.append(fs)

        # Run Nelder-Mead on each simplex in parallel (sequential loop, but independent)
        # We distribute the remaining budget among the 5 simplexes
        # Each simplex needs dim+1 points to start.
        # We will iterate, picking the next simplex to update.
        
        while calls_made < budget and time.time() < deadline:
            for idx in range(len(simplexes)):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                simplex = simplexes[idx]
                fs = simplex_fs[idx]
                
                # Sort simplex by fitness
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update global best
                # (safe_call already updates global best, but we ensure local consistency)
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)
                
                if fr is None:
                    break
                
                if fr < fs[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fs[i] = fi
                        if calls_made >= budget or time.time() > deadline:
                            break

                simplexes[idx] = simplex
                simplex_fs[idx] = fs

        return best_x, best_fx
# EVOLVE-BLOCK-END
