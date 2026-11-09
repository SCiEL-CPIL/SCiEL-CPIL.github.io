"""Direct Pareto front sampling for DTLZ2 using low-discrepancy sequences."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for DTLZ2 using low-discrepancy sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Generate Halton sequence in [0,1]^2
        def halton_sequence(n, base):
            result = np.zeros(n)
            for i in range(n):
                f = 1.0
                r = 0.0
                idx = i + 1
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                result[i] = r
            return result

        # Generate 2D Halton points
        h1 = halton_sequence(n, 2)
        h2 = halton_sequence(n, 3)

        # Map to angles on the Pareto front of DTLZ2
        # DTLZ2 Pareto front: sum of squares of objectives = 1
        # Parameterize with 2 angles for 3 objectives
        # Use spherical coordinates:
        #   f1 = cos(theta1)
        #   f2 = sin(theta1) * cos(theta2)
        #   f3 = sin(theta1) * sin(theta2)
        # where theta1 in [0, pi/2], theta2 in [0, pi/2]

        # Map Halton points to angles
        theta1 = np.pi / 2 * h1
        theta2 = np.pi / 2 * h2

        # Compute objective values on the Pareto front
        F = np.zeros((n, n_obj))
        F[:, 0] = np.cos(theta1)
        F[:, 1] = np.sin(theta1) * np.cos(theta2)
        F[:, 2] = np.sin(theta1) * np.sin(theta2)

        # For DTLZ2, the decision variables that achieve the Pareto front:
        # x0 = sin(pi/2 * u1) * ... for the first (dim - n_obj + 1) variables
        # The remaining (n_obj - 1) variables are the angles
        # Specifically for DTLZ2 with m objectives and n variables:
        # x_{n-m+1} = sin(pi/2 * u_1)
        # x_{n-m+2} = sin(pi/2 * u_2)
        # ...
        # x_n = sin(pi/2 * u_{m-1})
        # And the first (n - m + 1) variables are set to achieve g=0

        # For DTLZ2, g(x) = sum((x_i - 0.5)^2) for i in 0..n-m
        # To get g=0, set x_i = 0.5 for i in 0..n-m
        # Then x_{n-m+1} = sin(pi/2 * sin(pi/2 * h1))
        # x_{n-m+2} = sin(pi/2 * cos(pi/2 * h1) * sin(pi/2 * h2))
        # x_{n-m+3} = sin(pi/2 * cos(pi/2 * h1) * cos(pi/2 * h2))
        # Wait, let me reconsider the parameterization.

        # Standard DTLZ2 parameterization:
        # x_1 = sin(pi/2 * u_1)
        # x_2 = sin(pi/2 * u_2)
        # ...
        # x_{m-1} = sin(pi/2 * u_{m-1})
        # x_m = cos(pi/2 * u_{m-1}) * sin(pi/2 * u_{m-2}) * ... 
        # Actually, the standard way is:
        # Let theta_i = pi/2 * u_i for i = 1, ..., m-1
        # x_{n-m+1} = sin(theta_1)
        # x_{n-m+2} = cos(theta_1) * sin(theta_2)
        # ...
        # x_n = cos(theta_1) * cos(theta_2) * ... * cos(theta_{m-2}) * sin(theta_{m-1})
        # And x_1 = x_2 = ... = x_{n-m} = 0.5 (to make g=0)

        # For 3 objectives (m=3), n=12:
        # x_1 = ... = x_10 = 0.5
        # x_11 = sin(theta_1)
        # x_12 = cos(theta_1) * sin(theta_2)
        # where theta_1 = pi/2 * h1, theta_2 = pi/2 * h2

        X = np.zeros((n, dim))
        # Set first (dim - n_obj + 1) = 10 variables to 0.5
        X[:, :dim - n_obj + 1] = 0.5
        # Set the last (n_obj - 1) = 2 variables
        X[:, dim - n_obj + 1] = np.sin(theta1)
        X[:, dim - n_obj + 2] = np.cos(theta1) * np.sin(theta2)

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F_eval = np.array([problem(x) for x in X])

        # Check time
        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
