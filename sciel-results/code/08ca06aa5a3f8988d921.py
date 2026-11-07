"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Hybrid global-local optimizer: diverse starts + L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # safety margin before 300s kill

        # Global evaluation counter
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def wrapped(x):
            if self._call_count >= self.budget:
                return self._best_f
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Phase 1: Diverse global sampling
        rng = np.random.default_rng(42)
        candidates = []

        # Center
        candidates.append(np.zeros(self.dim))

        # Corners of the box
        for corner in range(2 ** self.dim):
            pt = np.array([(-5.0 if (corner >> i) & 1 == 0 else 5.0) for i in range(self.dim)])
            candidates.append(pt)

        # Mid-edges
        for i in range(self.dim):
            for val in [-3.0, 0.0, 3.0]:
                pt = np.zeros(self.dim)
                pt[i] = val
                candidates.append(pt)

        # Random points
        n_random = max(1, 30 - len(candidates))
        candidates.extend(rng.uniform(-4, 4, size=(n_random, self.dim)).tolist())

        # Evaluate all candidates
        for cand in candidates:
            if self._call_count >= self.budget:
                break
            if time.time() - start_time > time_limit:
                break
            x = np.clip(cand, -5, 5)
            fx = wrapped(x)

        # Phase 2: L-BFGS-B local refinement from best points
        # Collect evaluated points sorted by fitness
        # We don't have all points stored, so re-evaluate top candidates
        # Actually, let's track evaluated points
        # Let me restructure: store all evaluated points

        # Re-do with tracking
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        evaluated_points = []

        def wrapped2(x):
            if self._call_count >= self.budget:
                return self._best_f
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            evaluated_points.append((x.copy(), val))
            return val

        # Phase 1: Diverse global sampling
        for cand in candidates:
            if self._call_count >= self.budget:
                break
            if time.time() - start_time > time_limit:
                break
            x = np.clip(cand, -5, 5)
            wrapped2(x)

        # Sort by fitness and pick top starts for local optimization
        if evaluated_points:
            evaluated_points.sort(key=lambda t: t[1])
            top_starts = [p[0] for p in evaluated_points[:min(5, len(evaluated_points))]]
        else:
            top_starts = [np.zeros(self.dim)]

        # Phase 2: L-BFGS-B from each top start
        for start_x in top_starts:
            if self._call_count >= self.budget:
                break
            if time.time() - start_time > time_limit:
                break

            # L-BFGS-B with bounds
            try:
                result = minimize(
                    wrapped2,
                    x0=start_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 200, 'maxfun': self.budget - self._call_count,
                             'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass

            if time.time() - start_time > time_limit:
                break

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
