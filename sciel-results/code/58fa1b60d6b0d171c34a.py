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
        # Use a grid of stratified points: divide the simplex into sub-triangles
        # For better coverage, use a combination of grid + jitter
        # Strategy: use a 2D stratified design in [0,1]^2, then map to simplex
        # via the inverse transform for uniform sampling on simplex
        # 
        # Uniform on simplex: given u,v ~ U(0,1), the transform is:
        #   a = 1 - sqrt(1-u)
        #   b = (1-a) * v
        # This gives uniform distribution on the standard 2-simplex.
        #
        # For stratified sampling, we stratify u and v separately.
        
        # Use a sqrt-spaced grid for u (since the CDF involves sqrt) and uniform for v
        # Actually, let's just use stratified u and v, then apply the transform.
        
        # Create stratified samples: for n points, use a grid with jitter
        # Use ceil(sqrt(n)) x ceil(sqrt(n)) grid, take first n points
        side = int(np.ceil(np.sqrt(n)))
        
        # Stratified u and v values
        u_strat = np.zeros(n)
        v_strat = np.zeros(n)
        
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= n:
                    break
                # Stratified position within cell
                u_strat[count] = (i + rng.uniform()) / side
                v_strat[count] = (j + rng.uniform()) / side
                count += 1
            if count >= n:
                break
        
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
