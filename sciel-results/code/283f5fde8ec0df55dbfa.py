import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 5:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with Latin Hypercube + Sobol-like points
        # Use a mix of random and structured sampling
        n_explore = 60
        rng = np.random.RandomState(42)
        
        # Latin Hypercube sampling
        lhs_points = []
        for i in range(n_explore):
            point = np.zeros(n)
            for d in range(n):
                point[d] = -5.0 + 10.0 * (i + rng.random()) / n_explore
            lhs_points.append(point)
        
        # Add some Sobol-like quasi-random points
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_vector(index, dims):
            point = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dims):
                point.append(halton(index, bases[d]))
            return np.array(point)

        explore_points = lhs_points.copy()
        for i in range(1, 31):
            h_point = halton_vector(i, n)
            x_try = -5.0 + 10.0 * h_point
            explore_points.append(x_try)

        # Add structured points
        structured = [np.zeros(n), 
                      np.ones(n)*2.0, -np.ones(n)*2.0, 
                      np.ones(n)*-2.0, np.ones(n)*3.0, -np.ones(n)*3.0,
                      np.ones(n)*1.0, -np.ones(n)*1.0,
                      np.ones(n)*4.0, -np.ones(n)*4.0]
        explore_points.extend(structured)

        # Evaluate all exploration points
        results = []
        for p in explore_points:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f = try_eval(p)
            if f is not None:
                results.append((f, p.copy()))

        if not results:
            return best_x, best_f

        results.sort(key=lambda x: x[0])

        # Phase 2: Local refinement with multiple methods
        remaining_budget = budget - total_calls
        if remaining_budget <= 0:
            return best_x, best_f

        # Select top candidates
        num_candidates = min(10, len(results))
        candidates = [results[i][1] for i in range(num_candidates)]

        # Use L-BFGS-B for each candidate with limited budget
        budget_per_candidate = max(15, remaining_budget // (num_candidates * 2))

        for i, start_x in enumerate(candidates):
            if total_calls >= budget or time.time() > deadline - 10:
                break

            def wrapped_func(x):
                f = try_eval(x)
                if f is None:
                    return 1e10
                return f

            try:
                res = minimize(wrapped_func, start_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * n,
                               options={'maxiter': 50, 'gtol': 1e-5, 'ftol': 1e-8})
                if hasattr(res, 'fun') and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: CMA-ES-like refinement around best point
        if total_calls < budget - 20 and time.time() < deadline - 10:
            sigma = 0.5
            pop_size = min(20, max(10, (budget - total_calls) // 5))
            for gen in range(5):
                if total_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Generate population
                pop = []
                for j in range(pop_size):
                    x_try = best_x + sigma * rng.randn(n)
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = try_eval(x_try)
                    if f_try is not None:
                        pop.append((f_try, x_try))
                
                if not pop:
                    break
                
                pop.sort(key=lambda x: x[0])
                
                # Update best
                if pop[0][0] < best_f:
                    best_f = pop[0][0]
                    best_x = pop[0][1].copy()
                
                # Adapt sigma
                if len(pop) >= 2:
                    improvement = best_f - pop[0][0]
                    if improvement < 1e-8:
                        sigma *= 0.7
                    else:
                        sigma *= 1.1
                    sigma = max(0.01, min(2.0, sigma))

        # Phase 4: Nelder-Mead polish
        if total_calls < budget - 10 and time.time() < deadline - 5:
            def wrapped_func_nm(x):
                f = try_eval(x)
                if f is None:
                    return 1e10
                return f

            try:
                res = minimize(wrapped_func_nm, best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                if hasattr(res, 'fun') and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: Final coordinate-wise refinement
        if total_calls < budget - 5 and time.time() < deadline - 3:
            for _ in range(2):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                improved = True
                while improved:
                    improved = False
                    for d in range(n):
                        if total_calls >= budget or time.time() > deadline - 3:
                            break
                        for delta in [0.05, -0.05, 0.01, -0.01, 0.001, -0.001]:
                            if total_calls >= budget or time.time() > deadline - 3:
                                break
                            x_try = best_x.copy()
                            x_try[d] += delta
                            x_try = np.clip(x_try, -5.0, 5.0)
                            f_try = try_eval(x_try)
                            if f_try is not None and f_try < best_f:
                                improved = True
                                break

        return best_x, best_f
