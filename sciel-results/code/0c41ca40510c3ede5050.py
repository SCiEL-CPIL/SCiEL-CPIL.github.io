"""Optimizer for BBOB f8 (Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Wrap func with clipping and a call counter
        call_count = [0]
        best_x = [np.zeros(dim)]
        best_f = [np.inf]

        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            call_count[0] += 1
            f = float(func(x))
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x.copy()
            return f

        def time_ok():
            return call_count[0] < budget and time.time() < deadline

        # --- Phase 1: Structured global exploration ---
        # Evaluate a set of structured points to find a good starting region
        scan_points = [np.zeros(dim)]
        
        # Axis-aligned points at key locations
        for v in [-4.0, -2.0, 0.0, 2.0, 4.0]:
            for i in range(dim):
                pt = np.zeros(dim)
                pt[i] = v
                scan_points.append(pt)
        
        # Diagonal points
        for v in [-3.0, -1.5, 0.0, 1.5, 3.0]:
            pt = np.full(dim, v)
            scan_points.append(pt)
        
        # Halton sequence for low-discrepancy coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        n_halton = 100
        for i in range(1, n_halton + 1):
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = halton(i, primes[j]) * 8.0 - 4.0
            scan_points.append(pt)

        # Evaluate scan points
        for p in scan_points:
            if not time_ok():
                break
            safe_func(p)

        # --- Phase 2: CMA-ES style adaptive search ---
        # Use a population-based approach with adaptive step sizes
        rng = np.random.default_rng(42)
        
        # Initialize population around the best point found so far
        pop_size = 12
        sigma = 1.5  # initial step size
        pop = []
        for _ in range(pop_size):
            x = best_x[0].copy() + sigma * rng.standard_normal(dim)
            x = np.clip(x, -5.0, 5.0)
            pop.append(x)
        
        # Track best in population
        pop_best_x = best_x[0].copy()
        pop_best_f = best_f[0]
        
        # CMA-ES-like loop with adaptive sigma
        max_iter = 30
        for it in range(max_iter):
            if not time_ok():
                break
            
            # Evaluate population
            evals = []
            for x in pop:
                if not time_ok():
                    break
                f = safe_func(x)
                evals.append((f, x))
            
            if not evals:
                break
            
            # Sort by fitness
            evals.sort(key=lambda t: t[0])
            
            # Update best
            if evals[0][0] < pop_best_f:
                pop_best_f = evals[0][0]
                pop_best_x = evals[0][1].copy()
            
            # Update best overall
            if pop_best_f < best_f[0]:
                best_f[0] = pop_best_f
                best_x[0] = pop_best_x.copy()
            
            # If no improvement for several iterations, reduce sigma
            if it > 0 and it % 5 == 0:
                # Check if we've made progress
                if pop_best_f >= best_f[0] * (1 - 1e-8):
                    sigma *= 0.7
                    # Re-center population
                    pop = []
                    for _ in range(pop_size):
                        x = best_x[0].copy() + sigma * rng.standard_normal(dim)
                        x = np.clip(x, -5.0, 5.0)
                        pop.append(x)
            
            # Generate next population: keep top half, replace bottom half
            half = pop_size // 2
            new_pop = [e[1].copy() for e in evals[:half]]
            for _ in range(pop_size - half):
                # Sample around best few points
                base_x = evals[rng.integers(0, max(1, half))][1]
                x = base_x.copy() + sigma * rng.standard_normal(dim)
                x = np.clip(x, -5.0, 5.0)
                new_pop.append(x)
            pop = new_pop
        
        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 3: L-BFGS-B local optimization ---
        x0 = best_x[0].copy()
        try:
            res = minimize(safe_func, x0, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim,
                           options={'maxiter': 300, 'maxls': 50, 'gtol': 1e-8})
            if res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = res.x
        except:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 4: Nelder-Mead polish ---
        x0 = best_x[0].copy()
        try:
            res = minimize(safe_func, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_f[0]:
                best_f[0] = res.fun
                best_x[0] = res.x
        except:
            pass

        if not time_ok():
            return best_x[0], best_f[0]

        # --- Phase 5: Multiple restarts with local optimization ---
        rng2 = np.random.default_rng(123)
        for _ in range(15):
            if not time_ok():
                break
            x0 = rng2.uniform(-4.0, 4.0, size=dim)
            try:
                res = minimize(safe_func, x0, method='L-BFGS-B', bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 150, 'maxls': 30, 'gtol': 1e-7})
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x
            except:
                pass

            if not time_ok():
                break

            # Quick NM polish after each restart
            try:
                res = minimize(safe_func, best_x[0].copy(), method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13})
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x
            except:
                pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
