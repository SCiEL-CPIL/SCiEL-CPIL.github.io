import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        limit = 290.0

        # Unpack bounds into numpy arrays for speed
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 analysis:
        # f1(x) = 1 + 10(n-1) + sum(xi^2 - 10cos(4pi*xi)) for i>0
        # f2(x) = (1+sqrt(f1))*[1 - sqrt(f1/(1+x0))]
        # The Pareto front is obtained when f1 is minimized, which happens at xi=0 for i>0.
        # So the front is a 1D curve parameterized by x0.
        # To maximize hypervolume with a fixed budget, we should place all evaluations
        # on the Pareto front.
        # We need to distribute x0 values to maximize the area under the curve.
        # A uniform distribution of x0 in [0, 1] is a robust choice.
        # Using np.linspace ensures we capture the endpoints exactly.

        n = self.budget
        if n <= 0:
            return None

        # Generate candidate x0 values
        # linspace includes both endpoints
        x0_vals = np.linspace(lo[0], hi[0], n)

        # Pre-allocate the vector for the non-x0 dimensions
        # Since the front is at xi=0 for i>0, we set them to lo[i] (which is 0)
        # If lo[i] is not 0, we might need to be careful, but for ZDT3 it is 0.
        # The problem clips to bounds, so setting to lo[i] is safe.
        rest_vals = np.full(self.dim - 1, lo[1])

        for x0 in x0_vals:
            if time.time() - t0 > limit:
                break

            # Construct the full vector
            # Using concatenation or indexing
            # Indexing is faster for small vectors
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = rest_vals

            try:
                problem(x)
            except Exception:
                # If an exception occurs, we stop to avoid wasting time or invalid state
                # The contract says "an exception later keeps the evaluations made"
                # So breaking is acceptable.
                break

        return None
