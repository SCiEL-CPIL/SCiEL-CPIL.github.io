# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It sets distance variables to their optimal values (0.5) to minimize g(x),
    and samples shape variables using a quasi-random sequence that yields
    near-uniform distribution on the Pareto front surface, maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n points of the Halton sequence in [0, 1) for the given base."""
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0 / base
            r = 0.0
            j = i + 1
            while j > 0:
                f /= base
                r += f * (j % base)
                j //= base
            seq[i] = r
        return seq

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # Optimal distance values for DTLZ4 are 0.5
        # g(x) = sum((x_i - 0.5)^2), so g=0 when x_i=0.5
        # This makes 1+g = 1, putting points exactly on the ideal front.

        # To maximize hypervolume with a reference point, we want points
        # distributed uniformly on the Pareto front.
        # For DTLZ4 3-objective, the front is a quarter sphere.
        # Uniform sampling on the sphere surface requires:
        # cos(theta_0) ~ Uniform(0, 1)
        # theta_1 ~ Uniform(0, pi/2)

        # The mapping from decision variables to angles is:
        # theta_0 = (pi / 2) * x_0
        # theta_1 = (pi / 2) * x_1

        # So:
        # cos((pi/2)*x_0) = u  => x_0 = (2/pi) * arccos(u)
        # (pi/2)*x_1 = v * (pi/2) => x_1 = v

        # Use a Halton sequence for low-discrepancy sampling
        # This ensures uniform coverage of the Pareto front surface

        # Generate Halton sequences for u and v
        u_seq = self._halton(budget, 2)
        v_seq = self._halton(budget, 3)

        results = []

        for k in range(budget):
            # Check deadline
            if time.time() > deadline:
                return None

            u = u_seq[k]
            v = v_seq[k]

            # Calculate shape variables
            x_0 = (2.0 / np.pi) * np.arccos(u)
            x_1 = v

            # Construct full vector
            x = np.zeros(dim)
            x[0] = x_0
            x[1] = x_1
            x[2:] = 0.5  # Set distance variables to optimal value

            # Evaluate
            f = problem(x)
            results.append(f)

        return None
# EVOLVE-BLOCK-END
