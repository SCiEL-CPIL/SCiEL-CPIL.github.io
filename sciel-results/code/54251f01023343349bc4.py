"""Multi-start local search: Sobol global sampling + Nelder-Mead refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration using Sobol sequence for quasi-random coverage
        # Sobol sequences provide better space-filling than LHS for low dimensions
        n_sobol = min(64, budget // 8)
        
        # Generate Sobol sequence points in [0,1]^dim, then map to [-5,5]^dim
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        # Map [0,1] to [-5,5]
        sobol_points = -5.0 + 10.0 * sobol_points
        
        # Add origin and a few structured points
        candidates = [np.zeros(dim)]
        for p in sobol_points:
            candidates.append(p)
        
        # Evaluate initial candidates
        for x in candidates:
            if calls_made >= budget or time_left() < 10:
                break
            make_call(np.clip(x, -5.0, 5.0))
        
        if best_x is None:
            make_call(np.zeros(dim))
            return best_x, best_f

        # Phase 2: Nelder-Mead refinement on top candidates
        # Sort evaluated points by fitness and refine the best few
        remaining_budget = budget - calls_made
        if remaining_budget < 10:
            return best_x, best_f

        # Track all evaluated points
        eval_points = []
        eval_fits = []
        
        # We need to re-evaluate or track from the start. Let's redo with tracking.
        # Actually, let's just use best_x as the starting point for NM.
        
        def nelder_mead(start_x, max_iter=200):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time_left() < 5:
                return
            
            # Initialize simplex
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            # Initial simplex: start_x + small perturbations
            eps = 0.25
            simplex = [start_x.copy()]
            for d in range(dim):
                pt = start_x.copy()
                pt[d] += eps
                simplex.append(np.clip(pt, -5.0, 5.0))
            
            # Evaluate simplex
            simplex_f = []
            for pt in simplex:
                if calls_made >= budget or time_left() < 2:
                    return
                f = make_call(pt)
                if f is None:
                    return
                simplex_f.append(f)
            
            # Nelder-Mead iterations
            for _ in range(max_iter):
                if calls_made >= budget or time_left() < 2:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                if calls_made >= budget or time_left() < 2:
                    break
                f_r = make_call(x_r)
                if f_r is None:
                    break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    if calls_made >= budget or time_left() < 2:
                        break
                    f_e = make_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < worst_f:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    if calls_made >= budget or time_left() < 2:
                        break
                    f_c = make_call(x_c)
                    if f_c is None:
                        break
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            if calls_made >= budget or time_left() < 2:
                                return
                            f_s = make_call(simplex[i])
                            if f_s is None:
                                return
                            simplex_f[i] = f_s
                        break
            
            # Check convergence
            if len(simplex_f) > 1:
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    return

        # Run Nelder-Mead on the best point
        nelder_mead(best_x.copy())

        return best_x, best_f
# EVOLVE-BLOCK-END
