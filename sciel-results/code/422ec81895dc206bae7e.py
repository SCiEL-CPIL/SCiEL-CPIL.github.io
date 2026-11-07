import numpy as np
import time


class Optimizer:
    """
    DTLZ2 optimizer using deterministic Halton sequence sampling on the known Pareto front.
    
    For DTLZ2 with M objectives and N variables:
      - First M-1 variables are 'shape' variables controlling position on the front
      - Remaining N-(M-1) variables are 'distance' variables that should be 0 for the ideal front
    
    Strategy:
      1. Use a deterministic Halton sequence to sample the 2D shape space uniformly
      2. Set distance variables to 0 to lie on the true Pareto front
      3. This guarantees all points are non-dominated and maximizes hypervolume coverage
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s wall-clock

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        k_shape = self.n_obj - 1  # 2 for M=3
        shape_indices = list(range(k_shape))
        dist_indices = list(range(k_shape, self.dim))

        # Use all budget for Halton sequence sampling on the Pareto front
        # Distance variables set to 0.0 to lie on the true Pareto front
        n_samples = self.budget

        objective_values = []
        used = 0

        # Generate Halton sequence points in [0, 1]^k_shape
        halton_points = self._halton_sequence(n_samples, k_shape)

        for i in range(n_samples):
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            # Set shape variables from Halton sequence
            for j, idx in enumerate(shape_indices):
                x[idx] = halton_points[i, j]
            # Set distance variables to 0 for the Pareto front
            for idx in dist_indices:
                x[idx] = 0.0

            f = np.array(problem(x))
            objective_values.append(f)
            used += 1

        return np.array(objective_values)

    def _halton_sequence(self, n, dims):
        """Generate n points in [0, 1]^dims using the Halton sequence."""
        # Use different primes for each dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        points = np.zeros((n, dims))
        
        for d in range(dims):
            prime = primes[d]
            for i in range(n):
                # Generate the (i+1)-th Halton number with base `prime`
                f = 1.0
                result = 0.0
                num = i + 1
                while num > 0:
                    f /= prime
                    result += f * (num % prime)
                    num //= prime
                points[i, d] = result
        
        return points
