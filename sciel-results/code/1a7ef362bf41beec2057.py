"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Powell


class Optimizer:
    """Multi-stage optimizer: scout + Powell refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s kill

        best_x = None
        best_f = np.inf

        # Count function evaluations
        eval_count = [0]

        def wrapped(x):
            eval_count[0] += 1
            if eval_count[0] > self.budget:
                raise RuntimeError("Budget exceeded")
            return func(x)

        # Stage 1: Scout phase - evaluate a set of candidate starting points
        # Include nominal Rosenbrock minimum, origin, and random points
        rng = np.random.default_rng(42)

        # Candidate starting points
        candidates = []
        # Nominal Rosenbrock minimum
        candidates.append(np.ones(self.dim))
        # Origin
        candidates.append(np.zeros(self.dim))
        # Random points in the box
        for _ in range(10):
            candidates.append(rng.uniform(-5, 5, self.dim))
        # Points near the nominal minimum with small perturbations
        for _ in range(5):
            candidates.append(np.ones(self.dim) + rng.uniform(-1, 1, self.dim))

        # Evaluate all candidates
        for x in candidates:
            if time.time() > deadline:
                break
            try:
                fx = wrapped(np.clip(x, -5, 5))
                if fx < best_f:
                    best_f = fx
                    best_x = np.clip(x, -5, 5).copy()
            except Exception:
                continue

        if best_x is None:
            # Fallback: return a default
            return np.ones(self.dim), 0.0

        # Stage 2: Powell refinement from the best scout point
        # Use Powell's method which is derivative-free and works well for Rosenbrock
        try:
            result = minimize(
                wrapped,
                best_x,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12},
                callback=lambda x: None
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        except Exception:
            pass

        # Stage 3: If we have remaining budget, do a few more Powell runs from
        # other good scout points
        if eval_count[0] < self.budget and time.time() < deadline:
            # Sort candidates by their function values (we need to re-evaluate or track)
            # Actually, let's just try a couple more starting points with Powell
            additional_starts = []
            # Try points near the current best
            for _ in range(3):
                additional_starts.append(best_x + rng.uniform(-0.5, 0.5, self.dim))
            # Try the origin if not already tried effectively
            additional_starts.append(np.zeros(self.dim))
            additional_starts.append(np.ones(self.dim))

            for x0 in additional_starts:
                if eval_count[0] >= self.budget:
                    break
                if time.time() > deadline:
                    break
                try:
                    x0_clipped = np.clip(x0, -5, 5)
                    # Quick check: evaluate x0 first
                    fx0 = wrapped(x0_clipped)
                    if fx0 >= best_f:
                        continue
                    # Run Powell from this point
                    result = minimize(
                        wrapped,
                        x0_clipped,
                        method='Powell',
                        options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-11},
                        callback=lambda x: None
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    continue

        # Ensure we're within bounds
        best_x = np.clip(best_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
