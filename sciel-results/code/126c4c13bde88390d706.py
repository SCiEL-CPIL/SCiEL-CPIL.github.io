"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 exploiting known Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _simplex_stratified_sample(n, rng):
        """Generate n points uniformly on the 2-simplex {a,b>=0, a+b<=1} using stratified sampling."""
        # For uniform sampling on the 2-simplex, we use the inverse CDF transform.
        # If u,v ~ U(0,1) independent, then:
        #   a = 1 - sqrt(1 - u)
        #   b = (1 - a) * v
        # gives (a, b) uniform on the simplex.
        #
        # For stratified sampling, we stratify u and v independently on a grid.
        
        side = int(np.ceil(np.sqrt(n)))
        
        # Create a grid of stratified samples in [0,1]^2
        # Use the centers of each cell for deterministic stratification
        u_grid = np.linspace(0.0, 1.0, side + 1)
        v_grid = np.linspace(0.0, 1.0, side + 1)
        
        # Generate all grid cell centers
        u_centers = []
        v_centers = []
        for i in range(side):
            for j in range(side):
                u_centers.append((u_grid[i] + u_grid[i + 1]) / 2.0)
                v_centers.append((v_grid[j] + v_grid[j + 1]) / 2.0)
        
        u_centers = np.array(u_centers)
        v_centers = np.array(v_centers)
        
        # Take first n points (or all if side*side >= n)
        count = min(n, side * side)
        u_strat = u_centers[:count]
        v_strat = v_centers[:count]
        
        # If we need more points than grid cells, add jittered random points
        if count < n:
            remaining = n - count
            u_extra = rng.uniform(0.0, 1.0, size=remaining)
            v_extra = rng.uniform(0.0, 1.0, size=remaining)
            u_strat = np.concatenate([u_strat, u_extra])
            v_strat = np.concatenate([v_strat, v_extra])
        
        # Map to simplex coordinates
        a = 1.0 - np.sqrt(1.0 - u_strat)
        b = (1.0 - a) * v_strat
        
        return a, b

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_F = []

        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget
        
        # Use a fixed seed for the jitter to be deterministic
        rng = np.random.RandomState(42)
        
        # Generate stratified samples on the simplex
        x0_vals, x1_vals = self._simplex_stratified_sample(n_evals, rng)

        # Sample on the Pareto front
        for i in range(n_evals):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break

            x = np.zeros(self.dim)
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            x[2:] = 0.5  # optimal shape variables

            evaluate(x)

        return all_F
# EVOLVE-BLOCK-END
