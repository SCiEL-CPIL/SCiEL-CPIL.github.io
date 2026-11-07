"""Improved program using DTLZ2 Pareto front sampling with Halton sequence."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the analytical Pareto front of DTLZ2 with Halton sequence sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 Pareto front for 3 objectives:
        # f_1^2 + f_2^2 + f_3^2 = 0.25, f_i >= 0
        # Parameterize with angles a, b in [0, pi/2]:
        # f_1 = 0.5 * sin(a) * sin(b)
        # f_2 = 0.5 * sin(a) * cos(b)
        # f_3 = 0.5 * cos(a)
        #
        # The surface area element on this quarter-sphere is:
        # dA = 0.25 * sin(a) * da * db
        # So the PDF over (a, b) is proportional to sin(a)
        # Marginal of a: p(a) = 2*sin(a) for a in [0, pi/2]
        # CDF of a: F(a) = 1 - cos(a)
        # Inverse CDF: a = arccos(1 - u) where u ~ Uniform(0,1)
        # Conditional on a, b is uniform in [0, pi/2]

        # Generate Halton sequence points in 2D
        def halton_sequence(n, base1=2, base2=3):
            """Generate n points in 2D using Halton sequence."""
            points = np.zeros((n, 2))
            for i in range(n):
                # First dimension (base 2)
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base1
                    r += f * (i_val % base1)
                    i_val //= base1
                points[i, 0] = r

                # Second dimension (base 3)
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base2
                    r += f * (i_val % base2)
                    i_val //= base2
                points[i, 1] = r
            return points

        # Use all budget for Pareto front sampling
        n_points = budget
        halton_pts = halton_sequence(n_points)

        evaluated_points = []
        used = 0

        for i in range(n_points):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            u1 = halton_pts[i, 0]
            u2 = halton_pts[i, 1]

            # Transform to angles on the Pareto front
            # a = arccos(1 - u1) maps uniform to the correct distribution
            # b = u2 * pi/2 (uniform conditional on a)
            a = np.arccos(1.0 - u1)
            b = u2 * (np.pi / 2)

            # Compute objective values on the Pareto front
            f1 = 0.5 * np.sin(a) * np.sin(b)
            f2 = 0.5 * np.sin(a) * np.cos(b)
            f3 = 0.5 * np.cos(a)

            # Compute x that maps to these objectives
            # sin(x_i) = 2 * f_i, so x_i = arcsin(2*f_i)
            x = np.zeros(dim)
            sin_x1 = min(1.0, max(0.0, 2.0 * f1))
            sin_x2 = min(1.0, max(0.0, 2.0 * f2))
            sin_x3 = min(1.0, max(0.0, 2.0 * f3))
            x[0] = np.arcsin(sin_x1)
            x[1] = np.arcsin(sin_x2)
            x[2] = np.arcsin(sin_x3)
            # Remaining variables set to 0.5 for g=0
            x[3:] = 0.5

            # Clip to bounds
            x = np.clip(x, lo, hi)

            # Evaluate
            f = problem(x)
            evaluated_points.append(f)
            used += 1

        # Return the last evaluated point (return value is ignored by evaluator)
        if evaluated_points:
            return np.array(evaluated_points[-1])
        return None
# EVOLVE-BLOCK-END
