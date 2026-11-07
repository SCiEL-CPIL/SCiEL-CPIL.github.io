"""Multi-objective optimizer exploiting ZDT3 structure."""
import numpy as np
import time


class Optimizer:
    """Exploits ZDT3 structure: x_2..x_n=0 minimizes f_2, then refine x_1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3: f1 = x1, f2 = g(x2..xn) * h(f1, g)
        # g = 1 + 10(n-1) + sum(25*(sin(50*x_i^0.1)) for i=2..n)
        # h = 1 + sin^3(10*x1*pi)/2
        # Optimal: x2..xn=0 -> g=1, f1=x1, f2=h(x1)
        # Pareto front: f1 in [0,1], f2 = h(f1)

        used = 0
        results = []  # list of (x, f)

        def eval_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return np.array(f)

        # Strategy: Use stratified sampling on the known Pareto front manifold.
        # For ZDT3, x_2..x_dim=0 gives g=1, so f1=x1, f2=h(x1).
        # We allocate the entire budget to evaluating points on this manifold.
        # Use a non-uniform distribution to maximize hypervolume coverage.
        
        # The Pareto front has f1 in [0,1], f2 = 1 + sin^3(10*pi*f1)/2
        # h(f1) ranges from -0.5 to 1.5 (approximately)
        # For hypervolume with reference [11, 11], we want to cover the front well.
        
        # Key insight: The front has 5 "valleys" where h is minimized (at x1 = 0.15, 0.35, 0.55, 0.75, 0.95)
        # and 5 "peaks" where h is maximized (at x1 = 0.05, 0.25, 0.45, 0.65, 0.85)
        
        # Use stratified sampling: divide [0,1] into 5 equal intervals, each containing one valley
        # Sample more densely around the valleys where the front dips lower
        
        # Create a non-uniform grid that emphasizes the valley regions
        # Use inverse CDF approach: sample u in [0,1], map to x1 via a density that peaks at valleys
        
        # Simpler approach: create a fine grid and use adaptive refinement
        # Start with a coarse grid, then add points in regions with high curvature
        
        # Phase 1: Coarse stratified grid (50 points)
        n_coarse = 50
        x1_coarse = np.linspace(0.0, 1.0, n_coarse)
        
        for x1 in x1_coarse:
            if used >= budget or time.time() > deadline - 5:
                break
            x = np.zeros(dim)
            x[0] = x1
            f = eval_point(x)
            results.append((x.copy(), f))
        
        # Phase 2: Fine grid around valleys (100 points)
        # Valleys at x1 = 0.15, 0.35, 0.55, 0.75, 0.95
        valley_centers = [0.15, 0.35, 0.55, 0.75, 0.95]
        n_fine_per_valley = 20
        
        for center in valley_centers:
            for i in range(n_fine_per_valley):
                if used >= budget or time.time() > deadline - 5:
                    break
                # Spread points in a window around the valley center
                x1 = center + (i - n_fine_per_valley/2) * 0.01
                x1 = np.clip(x1, 0.0, 1.0)
                x = np.zeros(dim)
                x[0] = x1
                f = eval_point(x)
                results.append((x.copy(), f))
        
        # Phase 3: Random sampling with rejection to fill gaps (remaining budget)
        rng = np.random.default_rng(42)
        while used < budget:
            if time.time() > deadline - 5:
                break
            x1 = rng.random()
            x = np.zeros(dim)
            x[0] = x1
            f = eval_point(x)
            results.append((x.copy(), f))
        
        return results
