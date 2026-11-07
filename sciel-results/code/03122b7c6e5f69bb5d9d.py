"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-restart L-BFGS-B optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit

        # Count function calls
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count
            call_count += 1
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            return val

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Strategy: combine random points, grid-like points, and points near origin
        start_points = []
        
        # 1) Origin and nearby points
        start_points.append(np.zeros(self.dim))
        
        # 2) Random points in the box
        n_random = 20
        for _ in range(n_random):
            start_points.append(rng.uniform(-5, 5, self.dim))
        
        # 3) Points along axes
        for i in range(self.dim):
            for sign in [-1, 1]:
                pt = np.zeros(self.dim)
                pt[i] = sign * 3.0
                start_points.append(pt)
        
        # 4) Corner-like points
        for mask in range(min(16, 1 << self.dim)):
            pt = np.array([(-4.0 if (mask >> i) & 1 else 4.0) for i in range(self.dim)])
            start_points.append(pt)
        
        # Evaluate all starting points first to get a sense of the landscape
        for sp in start_points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                val = wrapped_func(sp)
                if val < best_f:
                    best_f = val
                    best_x = sp.copy()
            except:
                pass

        # Now run L-BFGS-B from the best starting points
        # Sort start points by their function values (we need to re-evaluate or store)
        # Since we already evaluated, let's just pick the best few and run L-BFGS-B from them
        
        # Re-evaluate to get values for sorting (or we can just use the best ones we found)
        # Let's collect all evaluated points and their values
        eval_points = []
        for sp in start_points:
            # We already evaluated these, but we need to track which ones
            # Actually, let's just run L-BFGS-B from the top candidates
            pass
        
        # Run L-BFGS-B from the best starting point found so far, and a few others
        # Sort by the values we got
        point_values = []
        for sp in start_points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            # We need the value; we didn't store it separately, so re-evaluate if needed
            # Actually, we did evaluate them above. Let's just use the best_x as primary
            # and try a few diverse ones
            pass
        
        # Simpler approach: run L-BFGS-B from the best point found, then from a few random ones
        # But we need to be careful about budget
        
        # Let's run L-BFGS-B from multiple starting points, stopping when budget is nearly exhausted
        # We'll prioritize the best starting point first
        
        # Create a list of (value, point) for the starting points we evaluated
        # We need to re-evaluate since we didn't store values properly
        # Let's restructure: evaluate and store
        
        # Clear and redo more carefully
        # Actually, let's just use the best_x we found and run L-BFGS-B from it,
        # then from a few other good points
        
        # Run L-BFGS-B from best_x
        if best_x is not None and call_count < self.budget - 10:
            try:
                res = minimize(wrapped_func, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 100, 'maxfun': self.budget - call_count - 5,
                                        'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except:
                pass
        
        # Run L-BFGS-B from a few other diverse points
        other_starts = [
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            np.zeros(self.dim),
        ]
        
        for sp in other_starts:
            if call_count >= self.budget - 10:
                break
            if time.time() > deadline:
                break
            try:
                res = minimize(wrapped_func, sp, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 50, 'maxfun': self.budget - call_count - 5,
                                        'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except:
                pass
        
        # Use remaining budget for local refinement around best_x
        if best_x is not None and call_count < self.budget - 5:
            # Do a few more L-BFGS-B runs from slightly perturbed best_x
            for _ in range(3):
                if call_count >= self.budget - 5:
                    break
                if time.time() > deadline:
                    break
                perturbed = best_x + 0.1 * rng.standard_normal(self.dim)
                perturbed = np.clip(perturbed, -5, 5)
                try:
                    res = minimize(wrapped_func, perturbed, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * self.dim,
                                   options={'maxiter': 30, 'maxfun': self.budget - call_count - 3,
                                            'ftol': 1e-12, 'gtol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except:
                    pass
        
        # Final check
        if best_x is None:
            best_x = np.zeros(self.dim)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
