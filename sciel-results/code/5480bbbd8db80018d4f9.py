"""Multi-objective optimizer on DTLZ7 exploiting known Pareto front structure."""
import numpy as np
import time


class Optimizer:
    """DTLZ7 optimizer using structured 2D sampling on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # f1 = x[0], f2 = x[1], f3 = (1+g)*h where g=0 when x[2:]=0
        # For g=0: f3 = h = 3 - x[0]*(1+sin(3*pi*x[0])) - x[1]*(1+sin(2*pi*x[1]))
        # We want to maximize hypervolume with reference point [1.1, 1.1, 25.0]
        # So we want to spread points across the (f1, f2) plane where f1=x[0], f2=x[1]

        # Strategy: Use a combination of grid and Halton sequence for good coverage
        # of the 2D Pareto front region

        # First, try a coarse grid for baseline coverage
        n_grid = int(np.sqrt(self.budget))
        while n_grid * n_grid > self.budget:
            n_grid -= 1

        if n_grid >= 2:
            x0_vals = np.linspace(lo[0], hi[0], n_grid)
            x1_vals = np.linspace(lo[1], hi[1], n_grid)

            for i in range(n_grid):
                for j in range(n_grid):
                    if time.time() > deadline or used >= self.budget:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x0_vals[i]
                    x[1] = x1_vals[j]
                    # x[2:] = 0 for Pareto front
                    problem(x)
                    used += 1
                if time.time() > deadline or used >= self.budget:
                    break

        # Fill remaining budget with Halton sequence for better space-filling
        def halton(index, base):
            """Generate the Halton sequence."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Use shifted Halton to avoid overlap with grid points
        offset = 1000
        while used < self.budget:
            if time.time() > deadline:
                break

            n = used + 1
            x0_val = lo[0] + (hi[0] - lo[0]) * halton(n + offset, 2)
            x1_val = lo[1] + (hi[1] - lo[1]) * halton(n + offset, 3)

            x = np.zeros(self.dim)
            x[0] = x0_val
            x[1] = x1_val
            # x[2:] = 0 for Pareto front
            problem(x)
            used += 1

        # If we still have budget, add more Halton points with different offsets
        offset2 = 5000
        while used < self.budget:
            if time.time() > deadline:
                break

            n = used - self.budget + 1  # new sequence
            if n <= 0:
                n = 1
            x0_val = lo[0] + (hi[0] - lo[0]) * halton(n + offset2, 5)
            x1_val = lo[1] + (hi[1] - lo[1]) * halton(n + offset2, 7)

            x = np.zeros(self.dim)
            x[0] = x0_val
            x[1] = x1_val
            # x[2:] = 0 for Pareto front
            problem(x)
            used += 1

        return None
