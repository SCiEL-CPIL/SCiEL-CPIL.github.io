"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track best point and value
        best_x = None
        best_f = np.inf
        calls = 0

        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            calls += 1
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Strategy: mix of structured starts and random starts
        # Phase 1: Grid-like structured starts to cover the space
        # Phase 2: Random starts to fill gaps
        # Phase 3: Fine-tune around best found

        # Number of restarts based on budget
        # Each L-BFGS-B call uses ~20-50 function evaluations typically
        # With 500 calls, we can afford about 10-20 restarts

        restarts = []

        # Center point
        restarts.append(np.zeros(dim))

        # Axis-aligned points
        for axis in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[axis] = val
                restarts.append(p)

        # Corner-like points
        for i in range(min(8, budget // 30)):
            p = rng.uniform(-4.5, 4.5, dim)
            restarts.append(p)

        # Additional random points
        while len(restarts) < min(15, budget // 25):
            p = rng.uniform(-5.0, 5.0, dim)
            restarts.append(p)

        # Shuffle to interleave structured and random
        indices = list(range(len(restarts)))
        rng.shuffle(indices)
        restarts = [restarts[i] for i in indices]

        # Adaptive budget per restart
        # Allocate more budget to early restarts, less to later ones
        # Use a decaying allocation

        for ri, x0 in enumerate(restarts):
            # Check deadline
            if time.time() > deadline:
                break

            # Check remaining budget
            remaining = budget - calls
            if remaining < 10:
                break

            # Allocate budget for this restart
            # Early restarts get more budget
            if ri < 3:
                alloc = min(remaining, max(30, budget // 8))
            elif ri < 7:
                alloc = min(remaining, max(20, budget // 12))
            else:
                alloc = min(remaining, max(10, budget // 20))

            if alloc < 10:
                break

            # Run L-BFGS-B
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': max(1, alloc),
                        'maxfun': max(1, alloc),
                        'gtol': 1e-6,
                    }
                )
            except Exception:
                continue

            # If this was the best result so far, consider a refinement pass
            if time.time() > deadline:
                break
            remaining = budget - calls
            if remaining < 5:
                break

            # Polish: do a second L-BFGS-B run from the result with tighter tolerance
            x_ref = result.x
            f_ref = result.fun
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

            # Final polish pass if we have budget
            if remaining >= 5:
                try:
                    result2 = minimize(
                        wrapped_func,
                        x_ref,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={
                            'maxiter': max(1, remaining),
                            'maxfun': max(1, remaining),
                            'gtol': 1e-8,
                        }
                    )
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

            # Early termination: if we're very close to likely optimum
            if best_f < 1e-6:
                break

        # If we still have budget, do a final polish around the best point
        remaining = budget - calls
        if remaining >= 3 and best_x is not None and time.time() < deadline:
            try:
                result3 = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': max(1, remaining),
                        'maxfun': max(1, remaining),
                        'gtol': 1e-10,
                    }
                )
                if result3.fun < best_f:
                    best_f = result3.fun
                    best_x = result3.x.copy()
            except Exception:
                pass

        # If no valid point was found, return a default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
