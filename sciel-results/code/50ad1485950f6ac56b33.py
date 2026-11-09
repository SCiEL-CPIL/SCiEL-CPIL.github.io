import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with adaptive refinement for f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, f0, max_iter=30, initial_step=0.3):
            nonlocal calls_made
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            fvals = np.zeros(dim + 1)
            fvals[0] = f0

            for i in range(1, dim + 1):
                if not time_ok():
                    return x0, f0
                simplex[i] = x0.copy()
                simplex[i][i - 1] += initial_step
                fvals[i] = safe_call(simplex[i])

            for _ in range(max_iter):
                if not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.std(fvals) < 1e-8 * (1.0 + abs(fvals[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = safe_call(xc)

                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])

            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        def coordinate_descent(x, f, max_pass=6, step=0.5):
            nonlocal calls_made
            for _ in range(max_pass):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for sign in (1.0, -1.0):
                        if not time_ok():
                            break
                        cand = x.copy()
                        cand[d] = np.clip(cand[d] + sign * step, -5.0, 5.0)
                        fc = safe_call(cand)
                        if fc < f - 1e-12:
                            x = cand
                            f = fc
                            improved = True
                            break
                if not improved:
                    break
            return x, f

        # Phase 1: Diverse initial sampling
        rng = np.random.default_rng(42)
        initial_points = [np.zeros(dim)]
        
        # Add points along each axis
        for d in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[d] = val
                initial_points.append(p)
        
        # Add random points
        while len(initial_points) < 20:
            initial_points.append(rng.uniform(-4.5, 4.5, dim))
        
        for p in initial_points:
            if not time_ok():
                break
            safe_call(p)

        # Phase 2: Multi-start L-BFGS-B
        if time_ok() and best_x is not None:
            # Select diverse starting points based on current bests
            # We'll use the top few distinct regions
            observed_x = []
            observed_f = []
            
            # Re-evaluate a few key points to get a sense of landscape
            # Actually, let's just use random restarts around the best point 
            # and some random points to ensure coverage.
            
            # Strategy: Run L-BFGS-B from the current best and 4-5 other random points.
            # L-BFGS-B is good for smooth functions like Griewank-Rosenbrock locally.
            
            start_points = [best_x.copy()]
            
            # Add random points far from best
            for _ in range(5):
                if not time_ok():
                    break
                p = rng.uniform(-4.5, 4.5, dim)
                # Ensure some distance from best if possible
                if np.linalg.norm(p - best_x) > 1.0:
                    start_points.append(p)
            
            # Run L-BFGS-B from each start point
            for sp in start_points:
                if not time_ok():
                    break
                
                # Wrap func for scipy
                def wrapped_func(x):
                    return safe_call(x)
                
                try:
                    res = minimize(wrapped_func, sp, method='L-BFGS-B', 
                                   bounds=[(-5.0, 5.0)] * dim,
                                   options={'maxiter': 50, 'maxfun': 50, 'gtol': 1e-5})
                    if res.success or res.fun < best_f:
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 3: Local refinement with Nelder-Mead from best
        if time_ok() and best_x is not None:
            x, f = nelder_mead(best_x, best_f, max_iter=40, initial_step=0.2)
            if f < best_f:
                best_x, best_f = x, f

        # Phase 4: Final coordinate descent polish
        if time_ok() and best_x is not None:
            best_x, best_f = coordinate_descent(best_x, best_f, max_pass=5, step=0.2)
            
            # Try a finer step
            best_x, best_f = coordinate_descent(best_x, best_f, max_pass=5, step=0.1)

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
