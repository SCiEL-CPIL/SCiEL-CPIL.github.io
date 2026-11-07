"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using diverse random sampling followed by L-BFGS-B local refinement.
    L-BFGS-B uses gradient estimation which is more effective in the Rosenbrock valley
    than Powell's derivative-free direction adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        calls = 0
        best_x = None
        best_f = float('inf')

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin from 330s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return float('inf')
            if calls >= self.budget:
                return float('inf')
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
            except Exception:
                val = float('inf')
            if val < best_f:
                best_f = val
                best_x = np.asarray(x, dtype=float).copy()
            return val

        # Phase 1: Diverse random exploration
        # Sample more points to better cover the basin of attraction
        n_explore = min(20, self.budget // 5)

        for i in range(n_explore):
            if calls >= self.budget or time.time() > deadline:
                break
            # Mix of uniform and slightly biased sampling
            if i < n_explore // 2:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Biased toward center where minimizer tends to be
                x = rng.uniform(-4, 4, self.dim)
            wrapped_func(x)

        # Phase 2: Local exploitation using L-BFGS-B from the best point
        if best_x is not None and calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxfun': remaining,
                        'maxiter': 500,
                        'gtol': 1e-8,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass

        # Phase 3: If budget remains, try L-BFGS-B from other promising points
        if calls < self.budget - 30 and time.time() < deadline - 5:
            # Try from the second-best or a new random point
            # First, let's try a few more random points to find another basin
            for attempt in range(3):
                if calls >= self.budget - 30 or time.time() > deadline - 5:
                    break
                x_new = rng.uniform(-5, 5, self.dim)
                f_new = wrapped_func(x_new)
                
                # Only pursue if this point is competitive
                if f_new < best_f * 0.7 or (best_f > 10 and f_new < best_f * 0.9):
                    remaining = self.budget - calls
                    if remaining > 20:
                        try:
                            res = minimize(
                                wrapped_func,
                                x_new,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * self.dim,
                                options={
                                    'maxfun': remaining,
                                    'maxiter': 300,
                                    'gtol': 1e-8,
                                    'ftol': 1e-10
                                }
                            )
                        except Exception:
                            pass
                        break  # Only do one additional local search

        # Phase 4: Final fine-tuning with Nelder-Mead if budget allows
        if calls < self.budget - 10 and time.time() < deadline - 2 and best_x is not None:
            remaining = self.budget - calls
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-10
                    }
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
