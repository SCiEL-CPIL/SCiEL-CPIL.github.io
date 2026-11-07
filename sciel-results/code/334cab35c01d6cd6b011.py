"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14 with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def eval_wrapper(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return func(x)

        calls_made = 0

        # Generate diverse starting points
        n_starts = 15
        starts = []
        # Origin
        starts.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                starts.append(p)
        # Random points
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        for s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            s = np.clip(s, -5.0, 5.0)
            fs = eval_wrapper(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort starts by objective value
        start_fvals = []
        for s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            s = np.clip(s, -5.0, 5.0)
            # Re-evaluate if not already (we tracked calls but not which point)
            # Actually we need to be careful - let's just re-evaluate the best ones
            start_fvals.append((s, best_f))  # placeholder

        # Better approach: evaluate each start once, keep track
        # Let's redo this more carefully

        # Reset and do it properly
        calls_made = 0
        best_x = None
        best_f = np.inf
        start_results = []

        for s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            s = np.clip(s, -5.0, 5.0)
            fs = eval_wrapper(s)
            start_results.append((s.copy(), fs))
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort by objective value
        start_results.sort(key=lambda x: x[1])

        # Run Nelder-Mead from each start, allocating budget proportionally
        # Give more budget to better starts
        remaining_budget = budget - calls_made
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # Allocate budget: top 5 starts get 15% each, rest get 10% each
        # Adjust based on actual remaining
        n_nm = min(8, len(start_results))
        budget_per_nm = max(10, remaining_budget // (n_nm + 1))

        for i in range(n_nm):
            if calls_made >= budget or time.time() > deadline:
                break

            x0 = start_results[i][0]
            f0 = start_results[i][1]

            # If this start is worse than current best by a lot, skip
            if f0 > best_f + 50.0:
                continue

            # Run Nelder-Mead
            try:
                res = minimize(
                    eval_wrapper,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(20, budget_per_nm),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

            if time.time() > deadline - 5.0:
                break

        # Final polish with Nelder-Mead on the best point
        if calls_made < budget - 20 and time.time() < deadline - 10.0 and best_x is not None:
            try:
                res = minimize(
                    eval_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(10, (budget - calls_made) // 2),
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
