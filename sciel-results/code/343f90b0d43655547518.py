"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with adaptive (1+1)-ES polish for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        calls = [0]

        def safe_call(x):
            calls[0] += 1
            return func(np.clip(x, -5.0, 5.0))

        def time_ok():
            return (time.time() - t0) < deadline and calls[0] < budget - 5

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start sampling to find promising basins
        # Include origin and several random points
        n_samples = 15
        samples = []
        
        # Always try origin first
        x0 = np.zeros(n)
        fx0 = safe_call(x0)
        samples.append((fx0, x0.copy()))
        
        for _ in range(n_samples - 1):
            if not time_ok():
                break
            x = rng.uniform(-5.0, 5.0, n)
            fx = safe_call(x)
            samples.append((fx, x.copy()))

        samples.sort(key=lambda t: t[0])
        best_fx = samples[0][0]
        best_x = samples[0][1].copy()

        # Phase 2: Nelder-Mead on the top-k most promising starts
        # Use a robust implementation that tracks best point
        def nelder_mead(func, x0, budget_nm, deadline, rng):
            """
            Robust Nelder-Mead simplex method.
            Returns (best_x, best_f, calls_used)
            """
            n = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = func(x0)
            simplex_f = [f0]
            
            # Expand simplex along each axis
            step = 0.5
            for i in range(n):
                xi = x0.copy()
                xi[i] += step
                fi = func(xi)
                simplex.append(xi.copy())
                simplex_f.append(fi)
            
            calls_used = n + 1
            best_x = x0.copy()
            best_f = f0
            
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            best_x = simplex[0].copy()
            best_f = simplex_f[0]
            
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrinkage
            
            max_iter = budget_nm // (n + 1)
            
            for _ in range(max_iter):
                if calls_used >= budget_nm or time.time() > deadline:
                    break
                
                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = func(x_ref)
                calls_used += 1
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = func(x_exp)
                    calls_used += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = func(x_con)
                    calls_used += 1
                    
                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_shrink = func(simplex[i])
                            calls_used += 1
                            simplex_f[i] = f_shrink
                
                # Re-sort
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # Check convergence
                if max(simplex_f) - min(simplex_f) < 1e-10:
                    break
            
            return best_x, best_f, calls_used

        # Run Nelder-Mead on top 2 starts
        top_k = 2
        for i in range(min(top_k, len(samples))):
            if not time_ok():
                break
            
            x_start = samples[i][1].copy()
            f_start = samples[i][0]
            
            # Allocate budget for NM: ~100 calls per start
            budget_nm = min(100, budget - calls[0] - 5)
            if budget_nm < n + 1:
                continue
            
            x_best_nm, f_best_nm, used_nm = nelder_mead(safe_call, x_start, budget_nm, deadline, rng)
            
            if f_best_nm < best_fx:
                best_fx = f_best_nm
                best_x = x_best_nm.copy()

        # Phase 3: Adaptive (1+1)-ES polish
        if best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            stall = 0
            max_stall = 30
            
            while time_ok():
                if calls[0] >= budget - 2:
                    break
                
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_call(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    stall = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    stall += 1
                
                if stall > max_stall:
                    # Try a few random perturbations to escape local minimum
                    for _ in range(3):
                        if not time_ok():
                            break
                        y = x + 0.5 * rng.standard_normal(n)
                        y = np.clip(y, -5.0, 5.0)
                        fy = safe_call(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            sigma = 0.1
                            stall = 0
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                    break
                
                if sigma < 1e-8:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
