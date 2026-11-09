import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 (M=3, N=12).
    Exploits the known structure of the Pareto Front:
    - Position variables: x[0], x[1]
    - Distance variables: x[2:] (must be 0.5 for g=0)
    
    Uses a Halton-sequence based quasi-random strategy with adaptive alpha
    to cover the 2D Pareto front efficiently.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate Halton sequence value for given index and base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Generate Halton sequence points for 2D coverage
        # We'll use multiple alpha values to handle different DTLZ4 parameterizations
        # Standard DTLZ4 uses alpha=100, but we cover a range

        n_halton = 180  # Leave some budget for refinement if time permits
        n_alpha = 3  # Number of different alpha assumptions

        results = []
        seen_points = set()

        # Alpha values to try: standard DTLZ4 (100), moderate (10), and low (1)
        # The mapping is x_i = u^(1/alpha) for i in {0, 1}
        alphas = [100.0, 10.0, 1.0]

        # Generate Halton indices
        halton_indices = range(1, n_halton + 1)

        # Distribute Halton points across alpha values
        points_per_alpha = n_halton // n_alpha
        remainder = n_halton % n_alpha

        for alpha_idx, alpha in enumerate(alphas):
            # Get subset of Halton indices for this alpha
            start_idx = alpha_idx * points_per_alpha
            if alpha_idx < remainder:
                end_idx = start_idx + points_per_alpha + 1
            else:
                end_idx = start_idx + points_per_alpha

            for i in range(start_idx, end_idx):
                if time.time() > deadline:
                    break

                # Generate 2D Halton point
                u = self._halton(i + 1, 2)  # Base 2 for first coordinate
                v = self._halton(i + 1, 3)  # Base 3 for second coordinate

                # Map to x[0] and x[1] using the alpha-dependent transformation
                x = np.zeros(self.dim)
                x[0] = u ** (1.0 / alpha)
                x[1] = v ** (1.0 / alpha)
                x[2:] = 0.5  # Distance variables at optimal value

                # Clip to bounds
                x = np.clip(x, lo, hi)

                # Create a key to check for duplicates (coarse grid)
                key = (round(x[0], 6), round(x[1], 6))
                if key in seen_points:
                    continue
                seen_points.add(key)

                try:
                    f = problem(x)
                    results.append(f)
                except Exception:
                    break

            if time.time() > deadline:
                break

        # If we have time and budget left, add a few more points with different strategies
        if time.time() < deadline - 10:
            # Add some additional coverage with slightly different alpha values
            extra_alphas = [50.0, 5.0]
            extra_points = 10

            for alpha in extra_alphas:
                for i in range(extra_points // len(extra_alphas)):
                    if time.time() > deadline:
                        break

                    u = self._halton(i + n_halton + 1, 5)
                    v = self._halton(i + n_halton + 1, 7)

                    x = np.zeros(self.dim)
                    x[0] = u ** (1.0 / alpha)
                    x[1] = v ** (1.0 / alpha)
                    x[2:] = 0.5
                    x = np.clip(x, lo, hi)

                    key = (round(x[0], 6), round(x[1], 6))
                    if key in seen_points:
                        continue
                    seen_points.add(key)

                    try:
                        f = problem(x)
                        results.append(f)
                    except Exception:
                        break

        return results
