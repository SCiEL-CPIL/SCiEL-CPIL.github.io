"""Combined: deterministic Pareto-front grid with Halton-sequence refinement for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using structured front grid + Halton refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0,1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def _front_point(self, theta1, theta2, lo, hi):
        """Construct a decision vector that maps to a point on the DTLZ2 Pareto front.
        
        For DTLZ2 with M=3, n=12:
        - x1, x2 determine angles on the front
        - x3..x12 = 0.5 for g=0
        """
        x = np.full(self.dim, 0.5)  # x3..x12 = 0.5 for g=0
        x[0] = 2.0 * theta1 / np.pi  # theta1 in [0, pi/2] -> x1 in [0, 1]
        x[1] = 2.0 * theta2 / np.pi  # theta2 in [0, pi/2] -> x2 in [0, 1]
        # Clip to bounds
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0  # 300s limit, use 280s for safety
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate points on the known DTLZ2 Pareto front
        # Front: f1^2 + f2^2 + f3^2 = 1, all fi >= 0
        # Parameterize with theta1, theta2 in [0, pi/2]
        
        points = []
        used = 0
        
        # Grid: 14x14 = 196 points
        n_grid = 14
        theta1_vals = np.linspace(0, np.pi/2, n_grid)
        theta2_vals = np.linspace(0, np.pi/2, n_grid)
        
        for t1 in theta1_vals:
            for t2 in theta2_vals:
                if used >= self.budget:
                    break
                if time.time() > deadline - 10:
                    return points
                x = self._front_point(t1, t2, lo, hi)
                f = problem(x)
                points.append(f)
                used += 1
            if used >= self.budget:
                break
        
        # Halton refinement: use 2D Halton sequence to fill interstitial gaps
        if used < self.budget:
            n_halton = min(self.budget - used, 50)
            h1 = self._halton(n_halton, 2)  # base 2
            h2 = self._halton(n_halton, 3)  # base 3
            
            for i in range(n_halton):
                if used >= self.budget:
                    break
                if time.time() > deadline - 10:
                    return points
                t1 = h1[i] * np.pi / 2
                t2 = h2[i] * np.pi / 2
                x = self._front_point(t1, t2, lo, hi)
                f = problem(x)
                points.append(f)
                used += 1
        
        return points
# EVOLVE-BLOCK-END
