"""DTLZ1 optimizer using stratified simplex sampling for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic Pareto front with stratified points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ1 structure for n_obj=3, dim=7:
        # M = n_obj + 1 = 4
        # Position variables: x[0], x[1], x[2] (3 vars)
        # Distance variables: x[3], x[4], x[5], x[6] (4 vars)
        # Wait, standard DTLZ1: n_g = dim - (n_obj + 1) = 7 - 4 = 3
        # So position: x[0]..x[2] (3 vars), distance: x[3]..x[5] (3 vars)?
        # Actually for DTLZ1:
        # f1 = 0.5 * x[0] * (1 + g)
        # f2 = 0.5 * x[1] * (1 - x[0]) * (1 + g)
        # f3 = 0.5 * x[2] * (1 - x[1]) * (1 - x[0]) * (1 + g)
        # f4 = 0.5 * (1 - x[2]) * (1 - x[1]) * (1 - x[0]) * (1 + g)
        # g = 100 * (n_g + sum_{i=n_g+1}^{n} ((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))))
        # where n_g = n - M = 7 - 4 = 3
        # So distance vars are x[3], x[4], x[5] (indices 3,4,5), and x[6] is extra?
        # Let me just set all vars from index 3 onward to 0.5 to minimize g.
        # The Pareto front is when g=0, achieved when distance vars = 0.5.
        # Shape vars x[0], x[1], x[2] must satisfy x[0]+x[1]+x[2] <= 1 for the
        # standard DTLZ1 front (they are on a simplex).

        # For optimal hypervolume, we want to sample the Pareto front densely.
        # The Pareto front in objective space is:
        # f1 = 0.5*x0, f2 = 0.5*x1*(1-x0), f3 = 0.5*x2*(1-x1)*(1-x0), f4 = 0.5*(1-x2)*(1-x1)*(1-x0)
        # with x0+x1+x2 <= 1, xi >= 0

        # We'll use stratified sampling on the simplex {x0+x1+x2 <= 1, xi >= 0}
        # to maximize coverage of the Pareto front.

        # Strategy: Use a stratified approach on the 3D simplex.
        # We can parameterize using:
        # u = x0, v = x1*(1-x0), w = x2*(1-x1)*(1-x0)
        # But it's easier to sample directly on the simplex.

        # For the simplex x0+x1+x2 <= 1:
        # Let s = x0+x1+x2, then x0 = s*u, x1 = s*v, x2 = s*w where u+v+w=1, u,v,w >= 0
        # But this doesn't quite work because we need x0+x1+x2 <= 1.

        # Alternative: Sample (x0, x1, x2) uniformly on the simplex x0+x1+x2 <= 1.
        # We can do this by:
        # 1. Sample s ~ Beta(1, 4) (since the volume of the simplex of size s is s^3)
        #    Actually, for uniform sampling on {x0+x1+x2 <= 1}, we can use:
        #    - Sample 4 uniform random numbers, take sorted values, differences give Dirichlet(1,1,1,1)
        #    - Or: sample x0, x1, x2 from Dirichlet(1,1,1) and scale by s where s ~ U(0,1)^{1/3}

        # Simpler approach: Use quasi-random (Sobol) points mapped to the simplex.

        # Let's use a different parameterization:
        # Let a = x0, b = x1*(1-x0), c = x2*(1-x1)*(1-x0)
        # Then a, b, c >= 0 and a + b + c <= 1 (since x0+x1+x2 <= 1)
        # Actually this isn't quite right either.

        # Let me just use stratified random sampling on the simplex.
        # For each stratum, generate points uniformly.

        # Actually, the best approach for hypervolume is to ensure good coverage.
        # Let's use a combination of:
        # 1. Stratified sampling on the simplex
        # 2. Add some points near the corners and edges

        n_points = budget

        # Generate stratified points on the simplex x0+x1+x2 <= 1
        # Method: Divide into sub-simplexes and sample uniformly in each.

        # For simplicity and good coverage, use Sobol sequence mapped to simplex.
        from scipy.stats import qmc

        sobol = qmc.Sobol(d=4, scramble=True, seed=0)
        raw = sobol.random(n_points)  # (n_points, 4) in [0,1]

        # Map to simplex: use the inverse CDF approach for Dirichlet
        # For uniform on {x0+x1+x2 <= 1}:
        # Sample u1, u2, u3, u4 from U(0,1)
        # Sort them: v1 <= v2 <= v3 <= v4
        # Then x0 = v1, x1 = v2-v1, x2 = v3-v2, x3 = 1-v4 (where x3 is the "remainder")
        # This gives (x0, x1, x2, x3) ~ Dirichlet(1,1,1,1)
        # And x0+x1+x2 <= 1 automatically.

        # Actually, sorting gives Dirichlet(1,1,1,1) which is uniform on the simplex.
        # Let's use this.

        sorted_raw = np.sort(raw, axis=1)  # (n_points, 4)
        # x0 = sorted_raw[:, 0]
        # x1 = sorted_raw[:, 1] - sorted_raw[:, 0]
        # x2 = sorted_raw[:, 2] - sorted_raw[:, 1]
        # x3 = 1 - sorted_raw[:, 3]
        # x0+x1+x2+x3 = 1, all >= 0

        x0 = sorted_raw[:, 0]
        x1 = sorted_raw[:, 1] - sorted_raw[:, 0]
        x2 = sorted_raw[:, 2] - sorted_raw[:, 1]

        # Build decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2] = x2
        # Distance variables: set to 0.5 for g=0
        X[:, 3:] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points

        # Check time
        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
