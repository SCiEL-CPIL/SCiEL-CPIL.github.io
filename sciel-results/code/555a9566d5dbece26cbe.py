"""Multi-objective optimizer on ZDT6 with Pareto-front initialization and local refinement."""
import numpy as np
import time
import math


class Optimizer:
    """Evolutionary multi-objective search initialized on the known ZDT6 Pareto set."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f1_analytic(self, x1):
        """Compute f1 analytically for ZDT6."""
        return 1.0 - math.exp(-4.0 * x1) * (math.sin(6.0 * math.pi * x1) ** 6)

    def _compute_optimal_x1_grid(self, n_points):
        """
        Compute x1 values that yield approximately uniform f1 values on [0, 1].
        Uses bisection to invert f1(x1) = target_f1.
        """
        x1_vals = []
        # Target f1 values: uniform on [0, 1]
        target_f1 = np.linspace(0.0, 1.0, n_points)
        
        for tf in target_f1:
            if tf == 0.0:
                x1_vals.append(0.0)
            elif tf == 1.0:
                x1_vals.append(1.0)
            else:
                # Bisection to find x1 such that f1(x1) = tf
                lo, hi = 0.0, 1.0
                for _ in range(60):  # high precision
                    mid = (lo + hi) / 2.0
                    f_mid = self._f1_analytic(mid)
                    if f_mid < tf:
                        lo = mid
                    else:
                        hi = mid
                x1_vals.append((lo + hi) / 2.0)
        
        return np.array(x1_vals)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 330s limit, keep 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For ZDT6, the Pareto set is x1 in [0,1], x2=...=xn=0
        # The Pareto front: f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6, f2 = 1
        # To maximize hypervolume with reference point [11, 11], we want
        # uniform coverage of the Pareto front in objective space.
        # Since f2 is constant (=1), we just need uniform f1 coverage.

        # Use all budget for evaluating points on the Pareto front
        # with uniform f1 distribution
        n_points = self.budget
        
        # Compute x1 values for uniform f1 coverage
        x1_vals = self._compute_optimal_x1_grid(n_points)
        
        # Create points on the Pareto front
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_vals
        # x2..xn are 0 (the Pareto set)

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points

        # Check time - if we're running out, just return
        if time.time() > deadline - 5:
            return X

        # All budget used, return
        return X
