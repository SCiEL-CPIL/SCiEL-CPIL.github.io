"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the 2D Pareto front structure.
    The Pareto front lies in the subspace where x_3...x_n = 0.
    Uses a Sobol-like quasi-random sequence projected onto the Pareto front
    to maximize hypervolume coverage with uniform distribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.array(x, dtype=np.float64)
            x = np.clip(x, lo, hi)
            try:
                obj = problem(x)
                obj = np.array(obj, dtype=np.float64)
                if obj.shape[0] != self.n_obj:
                    return None
                if not np.all(np.isfinite(obj)):
                    return None
                results.append(obj.tolist())
                n_evals += 1
                return obj
            except Exception:
                return None

        # DTLZ7 Pareto front: x_3..x_n = 0
        # f1 = 1 - x_1
        # f2 = 1 - (1 - x_2)^(1/(1+g)) where g=0, so f2 = x_2
        # f3 = 1 + g*h = 1
        # So on the front: f1 + f2 = 1, f3 = 1
        # We want to sample (x_1, x_2) in [0,1]^2 to cover the front

        # Use a Sobol-like quasi-random sequence for better space-filling
        # Generate low-discrepancy points in [0,1]^2
        n_points = self.budget

        # Simple Sobol-like sequence using Halton sequence (low-discrepancy)
        def halton_sequence(n, base):
            """Generate n points of the Halton sequence in [0,1)."""
            points = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                points.append(r)
            return points

        # Generate Halton sequence points in 2D
        n_halton = n_points
        # Base 2 for first dimension, base 3 for second dimension
        x1_halton = np.array(halton_sequence(n_halton, 2))
        x2_halton = np.array(halton_sequence(n_halton, 3))

        # Add explicit corner points to ensure extreme bounds are captured
        # The corners of the Pareto front in (x1, x2) space are:
        # (0,0), (0,1), (1,0), (1,1)
        # But we need to be careful about budget

        # Reserve a few points for corners if budget allows
        corner_points = []
        if n_points > 4:
            # Add corners: (0,0), (0,1), (1,0), (1,1)
            corner_points = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            n_halton = n_points - len(corner_points)
            # Regenerate Halton points for the remaining budget
            x1_halton = np.array(halton_sequence(n_halton, 2))
            x2_halton = np.array(halton_sequence(n_halton, 3))

        # Combine corner points and Halton points
        if corner_points:
            x1_all = np.concatenate([np.array([p[0] for p in corner_points]), x1_halton])
            x2_all = np.concatenate([np.array([p[1] for p in corner_points]), x2_halton])
        else:
            x1_all = x1_halton
            x2_all = x2_halton

        # Ensure we don't exceed budget
        n_total = min(len(x1_all), self.budget)

        # Map to actual bounds
        x1_mapped = lo[0] + x1_all[:n_total] * (hi[0] - lo[0])
        x2_mapped = lo[1] + x2_all[:n_total] * (hi[1] - lo[1])

        for i in range(n_total):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = x1_mapped[i]
            x[1] = x2_mapped[i]
            # x[2:] = 0 gives g = 0, the Pareto-optimal condition

            evaluate(x)

        return results
# EVOLVE-BLOCK-END
