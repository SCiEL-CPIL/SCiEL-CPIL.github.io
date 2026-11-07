# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse starting points for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.perf_counter()

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            update_best(x, f)
            return f

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Strategy: mix of structured and random starts
        starts = []

        # 1. Center
        starts.append(np.zeros(dim))

        # 2. Halton-like quasi-random points (low-discrepancy)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_halton = 20
        for i in range(1, n_halton + 1):
            pt = np.array([halton(i, b) for b in range(2, dim + 2)])
            pt = pt * 10.0 - 5.0  # Map [0,1] to [-5,5]
            starts.append(pt)

        # 3. Random points
        n_random = 30
        for _ in range(n_random):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        # 4. Axis-aligned points (for potential coordinate-wise structure)
        for i in range(dim):
            for val in [-3.0, 0.0, 3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Shuffle to mix strategies
        rng.shuffle(starts)

        # Phase 1: Quick evaluation of all starts
        for x0 in starts:
            if time_left() < 5.0 or calls_made >= budget:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)

        # Phase 2: L-BFGS-B from the best few starting points
        # Re-evaluate to get sorted list (we track best_f but need sorted starts)
        # Actually, let's just try L-BFGS-B from all starts we evaluated, prioritizing better ones
        # But we don't have a sorted list easily. Let's just try from all unique starts.
        
        # To be efficient, we'll try L-BFGS-B from starts in order of their initial function value.
        # We need to store (f, x) for evaluated starts.
        # Let's restructure: evaluate starts, store results, then sort.
        
        # Re-doing Phase 1 to store results properly
        evaluated_starts = []
        # We already called wrapped_func, but we didn't store the mapping.
        # Let's just re-evaluate the top candidates or all of them if budget allows.
        # Given budget 500 and ~60 starts, we have plenty of budget for L-BFGS-B.
        
        # Let's just iterate through starts again, but this time we know their f-value from best_f updates?
        # No, best_f is global. We need individual f-values.
        
        # Simpler approach: Just run L-BFGS-B from every start point we generated.
        # L-BFGS-B is fast. If it converges quickly, we move on.
        
        for x0 in starts:
            if time_left() < 10.0 or calls_made >= budget:
                break
            
            x0 = np.clip(x0, -5.0, 5.0)
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: If we still have budget, do more random restarts with L-BFGS-B
        while time_left() > 10.0 and calls_made < budget:
            x0 = rng.uniform(-5.0, 5.0, dim)
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 10, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
