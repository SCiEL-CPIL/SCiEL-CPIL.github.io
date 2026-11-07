"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: Diverse initialization + SLSQP local search + Nelder-Mead fallback + high-precision polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        def make_func_wrapper(fallback_val):
            def wrapper(x):
                nonlocal calls_made
                if calls_made >= budget or not time_ok():
                    return fallback_val
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                calls_made += 1
                try_update(x_clipped, fx)
                return fx
            return wrapper

        # Phase 1: Diverse structured + random exploration
        # Generate diverse starting points: axis-aligned, corners, random
        candidates = []
        
        # Axis-aligned points (positive and negative)
        for axis in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[axis] = sign * 4.0
                candidates.append(x)
        
        # Corner points (subset)
        for corner in [
            np.array([4.0]*dim),
            np.array([-4.0]*dim),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0])[:dim],
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0])[:dim],
        ]:
            candidates.append(np.clip(corner, -5, 5))
        
        # Random points
        for _ in range(15):
            x = rng.uniform(-5, 5, dim)
            candidates.append(x)
        
        # Evaluate all candidates
        seeds = []
        for x in candidates:
            if not time_ok() or calls_made >= budget:
                break
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            try_update(x_clipped, fx)
            seeds.append((x_clipped.copy(), fx))

        if not seeds:
            return best_x, best_f

        # Sort seeds by function value
        seeds.sort(key=lambda s: s[1])
        
        # Keep top diverse seeds (at most 5)
        max_seeds = 5
        if len(seeds) > max_seeds:
            top = [seeds[0]]
            for s in seeds[1:]:
                if len(top) >= max_seeds:
                    break
                diverse = True
                for ts in top:
                    if np.linalg.norm(s[0] - ts[0]) < 1.0:
                        diverse = False
                        break
                if diverse:
                    top.append(s)
            seeds = top

        # Phase 2: SLSQP local optimization from each seed (primary: quadratic convergence)
        for sx, sf in seeds:
            if not time_ok() or calls_made >= budget:
                break
            
            wrapper = make_func_wrapper(sf)
            try:
                result = minimize(
                    wrapper, 
                    sx, 
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-12, 'maxfun': 100}
                )
            except Exception:
                pass

        # Phase 3: Nelder-Mead derivative-free fallback from best point
        if best_x is not None and calls_made < budget and time_ok():
            wrapper = make_func_wrapper(best_f)
            try:
                result = minimize(
                    wrapper,
                    best_x.copy(),
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 80}
                )
            except Exception:
                pass

        # Phase 4: Final high-precision SLSQP polish from best point
        if best_x is not None and calls_made < budget and time_ok():
            wrapper = make_func_wrapper(best_f)
            try:
                result = minimize(
                    wrapper,
                    best_x.copy(),
                    method='SLSQP',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 300, 'ftol': 1e-14, 'maxfun': 150}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
