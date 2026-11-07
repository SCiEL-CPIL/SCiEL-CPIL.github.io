"""Pattern-search optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses a deterministic pattern search with adaptive step size and orthogonal +
diagonal directions, combined with sparse random restarts for global coverage.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Pattern search with adaptive step size and sparse random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _pattern_search(self, func, rng, x0, f0, max_calls, deadline, initial_step=1.0):
        """
        Pattern search: maintain a set of directions (orthogonal + diagonal),
        adapt step size per direction via binary search for progress.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = f0
        calls = 0

        # Build direction set: standard basis + sign vectors
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        # Add diagonal directions
        for sign in [1, -1]:
            d = np.ones(dim) * sign
            directions.append(d)
        # Normalize all directions
        directions = np.array(directions, dtype=float)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms

        n_dirs = len(directions)
        step = initial_step
        best_x = x.copy()
        best_f = fx
        no_improve_count = 0

        while calls < max_calls and time.time() < deadline:
            improved = False
            for i in range(n_dirs):
                if calls >= max_calls or time.time() > deadline:
                    break
                d = directions[i]

                # Try positive direction
                y = x + step * d
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    improved = True
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve_count = 0
                    # Expand step slightly for this direction
                    step = min(step * 1.3, 10.0)
                else:
                    # Try negative direction
                    y2 = x - step * d
                    y2 = np.clip(y2, -5, 5)
                    fy2 = func(y2)
                    calls += 1

                    if fy2 < fx:
                        x = y2
                        fx = fy2
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve_count = 0
                        step = min(step * 1.3, 10.0)
                    else:
                        # Shrink step
                        step = max(step * 0.7, 1e-8)

            if not improved:
                no_improve_count += 1
                # If stuck for a while, try a larger random kick
                if no_improve_count >= 5:
                    kick = rng.standard_normal(dim) * step * 2
                    yk = x + kick
                    yk = np.clip(yk, -5, 5)
                    fyk = func(yk)
                    calls += 1
                    if fyk < fx:
                        x = yk
                        fx = fyk
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve_count = 0
                    else:
                        step = max(step * 0.5, 1e-8)
                        if step < 1e-7:
                            break

            # Reset step occasionally to avoid getting too small
            if calls % 50 == 0:
                step = max(step, 0.1)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Phase 1: Sparse random sampling to find good starting points
        n_samples = min(20, budget // 8)
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 10.0:
                break
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Pattern search from the best point
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 10.0:
            # Estimate initial step from the spread of good points
            # Use a moderate initial step
            initial_step = 0.5

            x_best, f_best, calls_used = self._pattern_search(
                func, rng, best_x.copy(), best_f, remaining, deadline, initial_step
            )
            total_calls += calls_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 3: Try a second pattern search from a different random point
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline - 10.0:
            x2 = rng.uniform(-3, 3, size=dim)
            fx2 = func(x2)
            total_calls += 1
            if fx2 < best_f:
                best_f = fx2
                best_x = x2.copy()

            remaining = budget - total_calls
            if remaining > 5 and time.time() < deadline - 10.0:
                x_best2, f_best2, calls_used2 = self._pattern_search(
                    func, rng, x2.copy(), fx2, remaining, deadline, initial_step=0.3
                )
                total_calls += calls_used2
                if f_best2 < best_f:
                    best_f = f_best2
                    best_x = x_best2.copy()

        # Phase 4: Final local refinement with small steps
        remaining = budget - total_calls
        if remaining > 3 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            fx_local = best_f
            sigma = 0.02

            for _ in range(remaining):
                if time.time() > deadline - 5.0:
                    break
                z = rng.standard_normal(dim)
                y = x_local + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx_local:
                    x_local = y
                    fx_local = fy
                    sigma *= 1.1
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma *= 0.85

                sigma = np.clip(sigma, 1e-9, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
