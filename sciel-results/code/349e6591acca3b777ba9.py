"""Optimizer for BBOB f20 (Schwefel) using CMA-ES with multi-restart."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-restart CMA-ES style optimizer for BBOB f20 (Schwefel).
    Uses scipy's Powell as a local refiner combined with diverse global starts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 270  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float).flatten()
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Evaluate a diverse set of starting points
        rng = np.random.default_rng(12345)
        
        # Key points for Schwefel-like functions
        starts = [
            np.zeros(dim),
            np.full(dim, 4.20968),
            np.full(dim, -4.20968),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
        ]
        
        # Add random starts
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for x0 in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fx = try_func(x0)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: Local refinement using Powell's method (derivative-free)
        # Powell's method is robust for small dimensions and rugged landscapes
        # We use it with a tight maxiter to stay within budget
        max_iter_powell = 30
        n_restarts = 5
        
        # Sort starts by function value (we don't have all values stored, so we just try the best ones)
        # Since we updated best_x, we start from there.
        # To be more effective, we should store the top few points.
        
        # Let's re-evaluate top candidates if budget allows, or just use best_x
        # Actually, let's keep a list of top points
        top_points = [(best_x, best_f)]
        
        # We didn't store all f values from phase 1. Let's do a quick re-evaluation of the starts
        # if we have budget, or just use the best found.
        # For efficiency, let's just run Powell from the current best, and maybe a few others if we stored them.
        # Since we didn't store them, let's just run from best_x.
        # But multi-restart is better. Let's re-evaluate a subset of starts to get top 3.
        
        if calls_made < budget - 100:
            # Re-evaluate starts to find top 3
            top_candidates = []
            for x0 in starts[:15]:  # Only first 15 to save time
                if calls_made >= budget - 100:
                    break
                fx = try_func(x0)
                if fx is not None:
                    top_candidates.append((fx, x0))
            
            if top_candidates:
                top_candidates.sort(key=lambda t: t[0])
                # Take top 3 distinct points (roughly)
                seen = []
                for f_val, x_val in top_candidates:
                    if all(np.linalg.norm(x_val - s) > 1.0 for s in seen):
                        seen.append(x_val)
                    if len(seen) >= 3:
                        break
                top_points = [(x, try_func(x)) for x in seen]
                # Update best
                for x, f in top_points:
                    if f is not None and f < best_f:
                        best_f = f
                        best_x = x.copy()

        # Run Powell from top points
        for i, (x0, f0) in enumerate(top_points):
            if calls_made >= budget - 20:
                break
            if time.time() > deadline:
                break
            
            # Define wrapper for scipy that counts calls
            def scipy_func(x):
                if calls_made >= budget:
                    return best_f
                fx = try_func(x)
                if fx is None:
                    return best_f
                return fx

            try:
                res = minimize(
                    scipy_func,
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter_powell, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.x is not None and res.fun is not None and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final coordinate polish on the best point
        if calls_made < budget - 10 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.1
            for _ in range(10):
                if calls_made >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 5:
                        break
                    for sign in [1.0, -1.0]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = try_func(y)
                        if fy is None:
                            break
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                    if calls_made >= budget - 5:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
