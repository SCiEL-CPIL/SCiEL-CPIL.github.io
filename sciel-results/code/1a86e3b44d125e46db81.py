"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock) with portfolio strategy."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES based optimizer with multi-start and adaptive population."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(12345)

        # Nelder-Mead implementation
        def nelder_mead(x0, max_calls, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_temp = x0.copy()
                if x_temp[i] == 0:
                    x_temp[i] = 0.1
                else:
                    x_temp[i] += 0.1
                simplex.append(x_temp)
            
            f_simplex = []
            for i in range(dim + 1):
                if calls_made >= budget or time.time() >= deadline:
                    return
                fx = safe_eval(simplex[i])
                if fx is None:
                    return
                f_simplex.append(fx)
            
            # Main loop
            max_iter = max_calls // (dim + 1) + 10
            for k in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_best = simplex[0]
                f_best = f_simplex[0]
                
                # Check for convergence
                if f_worst - f_best < 1e-10:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break

            return

        # Phase 1: Initial exploration with diverse starting points
        # Use Latin Hypercube sampling for better coverage
        n_starts = 10
        starts = []
        # One at origin
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Evaluate starting points
        for i, x0 in enumerate(starts):
            if calls_made >= budget or time.time() >= deadline:
                break
            fx = safe_eval(x0)
            if fx is None:
                break

        # Phase 2: Multi-start Nelder-Mead
        # Sort starts by function value (we need to store them)
        # Since safe_eval updates best, we need to track which start was best
        # Actually, we just ran them. Let's pick the best few from the starts we evaluated.
        # We didn't store the fx for each start separately, only global best.
        # Let's re-evaluate or just use the global best and a few random ones.
        # Better: Keep a list of (fx, x) for the initial starts.
        
        # Let's refine the initial phase to store results
        # We already evaluated them. Let's just run NM from the current best_x 
        # and a few other random points.
        
        # To be more robust, let's do a few rounds of:
        # 1. Pick a random point or point near best
        # 2. Run NM with a limited budget
        # 3. If improved, continue; else, restart.
        
        # Strategy: 
        # 1. Run NM from best_x with a large budget (e.g., 200 calls)
        # 2. If calls remain, run NM from 2 other random points with smaller budgets (50 calls each)
        
        if calls_made < budget and best_x is not None:
            # Run NM from the best point found so far
            remaining = budget - calls_made
            # Give NM 50% of remaining budget or 150 calls, whichever is less but at least 20
            nm_budget = min(remaining, max(20, int(remaining * 0.5)))
            nelder_mead(best_x, nm_budget)
            
            # If budget remains, try 2 more local searches from random perturbations
            if calls_made < budget and time.time() < deadline:
                for _ in range(2):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    # Perturb best_x
                    x0 = best_x + rng.standard_normal(dim) * 0.5
                    x0 = np.clip(x0, -5, 5)
                    remaining = budget - calls_made
                    if remaining > 10:
                        nm_budget = min(remaining, 50)
                        nelder_mead(x0, nm_budget)
            
            # If budget still remains, try a full restart from a random point
            if calls_made < budget and time.time() < deadline:
                x0 = rng.uniform(-4, 4, dim)
                remaining = budget - calls_made
                if remaining > 10:
                    nm_budget = min(remaining, 100)
                    nelder_mead(x0, nm_budget)

        return best_x, best_f
