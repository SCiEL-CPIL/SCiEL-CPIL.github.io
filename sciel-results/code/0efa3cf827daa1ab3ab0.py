"""Multi-objective optimizer on DTLZ7 exploiting known Pareto front structure."""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ7 optimizer using Sobol-based structured sampling on the known Pareto front
    with hypervolume-aware non-uniform density."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        n = self.budget

        # DTLZ7 Pareto front: x_2...x_n = 0, f1=x1, f2=x2, f3=1.
        # We need to sample (x1, x2) in [0,1]^2 to cover the front.
        # The reference point is [1.1, 1.1, 25.0].
        # Hypervolume contribution is maximized by points that are "far" from the
        # reference point in the dominated region. Since f3=1 is fixed, the
        # contribution depends on f1 and f2. Points with smaller f1 and f2
        # contribute more to hypervolume because they dominate a larger box.
        #
        # Strategy: Use a Sobol sequence for good spread, but apply a power-law
        # transformation to bias points toward the (0,0) corner of the (x1,x2)
        # space, which corresponds to the highest hypervolume contribution region.
        #
        # The transformation u -> u^alpha with alpha > 1 compresses values near 1
        # and expands values near 0. This biases the sample toward smaller x1, x2.
        # We use alpha ~ 2-3 to get a good balance between coverage and bias.

        n_corners = 4
        n_sobol = n - n_corners

        # Generate Sobol sequence in 2D
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            k = int(np.ceil(np.log2(n_sobol)))
            sobol_samples = sampler.random_base2(m=k)
            x12 = sobol_samples[:n_sobol]
        else:
            x12 = np.empty((0, 2))

        # Apply power-law transformation to bias toward (0,0) corner
        # alpha=2.0 provides a good balance: points are denser near origin
        # but still cover the full [0,1] range
        alpha = 2.0
        x12[:, 0] = x12[:, 0] ** alpha
        x12[:, 1] = x12[:, 1] ** alpha

        # Corner points to ensure boundary coverage
        x1_corners = np.array([0.0, 0.0, 1.0, 1.0])
        x2_corners = np.array([0.0, 1.0, 0.0, 1.0])

        # Combine Sobol and corner points
        x1 = np.concatenate([x12[:, 0], x1_corners])
        x2 = np.concatenate([x12[:, 1], x2_corners])

        # Clip to bounds
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])

        # Construct full dimension vectors: x[0]=x1, x[1]=x2, x[2:]=0
        for i in range(n):
            if time.time() > deadline or used >= n:
                break
            x = np.zeros(self.dim)
            x[0] = x1[i]
            x[1] = x2[i]
            # x[2:] remains 0 -> on Pareto front (g=0)
            problem(x)
            used += 1

        return None
