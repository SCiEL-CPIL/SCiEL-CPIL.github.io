"""Rewritten optimizer using multi-start Nelder-Mead with a time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead for 5-dim Schwefel-like f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._safety_margin = 10.0  # seconds to stop before the 330s kill

    def _wrap_func(self, func):
        """Wrap func to count calls and enforce the deadline."""
        def wrapped(x):
            self._call_count += 1
            if self._call_count > self.budget:
                raise RuntimeError("budget exceeded")
            if time.time() > self._deadline:
                raise TimeoutError("deadline reached")
            return float(func(np.asarray(x, dtype=np.float64)))
        return wrapped

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 330.0 - self._safety_margin
        self._call_count = 0

        dim = self.dim
        wrapped = self._wrap_func(func)

        # Deterministic set of starting points spread over the domain.
        # For 5-D, use a grid + a few random points.
        rng = np.random.default_rng(42)

        # Generate candidate starts: combine grid points and random samples.
        # We want to use the budget efficiently: each NM run uses ~ (dim+1)*iter_max evaluations.
        # With dim=5, one NM simplex has 6 vertices. Each iteration uses up to 6 evals.
        # Let's plan for ~8 restarts, each using up to ~50 evals on average.
        # Total budget is 500, so 8 restarts * 50 = 400, leaving some slack.

        # Grid starts: use a 2x2x2x2x2 grid = 32 points, but we can't do 32 restarts.
        # Instead, pick a subset: center, corners, and mid-points.
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Corners of [-4,4]^dim (minimizer is in [-4,4]^dim per task)
        for mask in range(2 ** dim):
            pt = np.array([4.0 if (mask >> i) & 1 else -4.0 for i in range(dim)])
            starts.append(pt)
        # Random starts
        for _ in range(20):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle deterministically
        perm = np.arange(len(starts))
        rng.shuffle(perm)
        starts = [starts[i] for i in perm]

        best_x = None
        best_f = np.inf

        for start in starts:
            if self._call_count >= self.budget - 10:
                break
            if time.time() > self._deadline:
                break

            # Clip start to domain
            x0 = np.clip(start, -5.0, 5.0)

            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'maxiter': 200,
                        'maxfev': min(80, self.budget - self._call_count),
                        'adaptive': True,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            # Fallback: evaluate center
            x0 = np.zeros(dim)
            best_f = wrapped(x0)
            best_x = x0

        return best_x, best_f
# EVOLVE-BLOCK-END
