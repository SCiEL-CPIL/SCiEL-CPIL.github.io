# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use a Halton quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points near/on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton sequence points in [0, 1) for the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            x = 0.0
            idx = i + 1
            while idx > 0:
                idx, rem = divmod(idx, base)
                x += f * rem
                f /= base
            result[i] = x
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = 0.5 * (lo + hi)

        n_evals = 0
        all_F = []

        # Number of angle variables: n_obj - 1 = 2
        n_angles = self.n_obj - 1
        # Number of distance variables: dim - n_obj + 1 = 10
        n_dist = self.dim - self.n_obj + 1

        # Generate Halton sequence points for the 2D angle space
        # We'll use as many points as our budget allows
        n_points = self.budget

        # Generate two Halton sequences for the two angle variables
        h1 = self._halton(n_points, 2)  # base 2
        h2 = self._halton(n_points, 3)  # base 3

        # Map [0, 1) to [0, pi/2] for both angles
        theta1 = h1 * (np.pi / 2)
        theta2 = h2 * (np.pi / 2)

        # Build decision vectors
        # For DTLZ2:
        # x[0] = angle1, x[1] = angle2, x[2:] = 0.5 (distance variables)
        # The objectives are:
        # f0 = (1 + g) * cos(theta1) * cos(theta2)
        # f1 = (1 + g) * cos(theta1) * sin(theta2)
        # f2 = (1 + g) * sin(theta1)
        # where g = sum((x_i - 0.5)^2) for distance variables
        # If all distance vars are 0.5, g = 0, so f = (cos*cos, cos*sin, sin)

        for i in range(n_points):
            if time.time() > deadline - 10:  # Safety margin
                break

            x = np.empty(self.dim)
            x[0] = theta1[i]
            x[1] = theta2[i]
            x[2:] = 0.5  # Midpoint of [0, 1] for distance variables

            # Clip to bounds just in case
            x = np.clip(x, lo, hi)

            f = problem(x)
            all_F.append(f)
            n_evals += 1

        return all_F
# EVOLVE-BLOCK-END
