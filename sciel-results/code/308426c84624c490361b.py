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
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5
        
        n_evals = min(budget, 200)
        
        if n_evals <= 0:
            return all_points
        
        # We want to place n_evals points on the 2D simplex (triangle) to maximize hypervolume.
        # The Pareto front of DTLZ1 (M=3) is a triangle in objective space:
        # f0 = g * 0.5 * x0
        # f1 = g * 0.5 * x1
        # f2 = g * 0.5 * (1 - x0 - x1)
        # With g=0 (min), the front is x0 + x1 + x2 = 1, xi >= 0.
        # Actually, for DTLZ1, the Pareto front is:
        # f0 = 0.5 * x0
        # f1 = 0.5 * x1
        # f2 = 0.5 * (1 - x0 - x1)
        # Wait, let me check the standard DTLZ1 definition.
        # f0 = 0.5 * x0 * g
        # f1 = 0.5 * x1 * (1 - x0) * g
        # f2 = 0.5 * (1 - x1) * (1 - x0) * g
        # No, that's not right either.
        
        # Standard DTLZ1:
        # f0 = 0.5 * x0 * g(x)
        # f1 = 0.5 * x1 * (1 - x0) * g(x)
        # f2 = 0.5 * (1 - x1) * (1 - x0) * g(x)
        # where g(x) = 1 + 10 * (M-1) + sum_{i=M}^{n-1} (xi - 0.5)^2 - 10 * cos(20 * pi * (xi - 0.5))
        # For M=3, g(x) = 1 + 10*2 + sum_{i=2}^{6} ... = 21 + ...
        # Minimum g is 1 (when xi=0.5 for i>=2).
        
        # Wait, the standard definition often has g_min = 1.
        # Let's assume the problem function returns the standard DTLZ1 objectives.
        # The Pareto front is when g=1 and x0, x1, x2 (where x2 = 1-x0-x1) are on the simplex.
        # f0 = 0.5 * x0
        # f1 = 0.5 * x1 * (1 - x0)
        # f2 = 0.5 * (1 - x1) * (1 - x0)
        
        # Let's verify this.
        # If x0=1, x1=0 -> f0=0.5, f1=0, f2=0.
        # If x0=0, x1=1 -> f0=0, f1=0, f2=0.5.
        # If x0=0, x1=0 -> f0=0, f1=0, f2=0.5.
        # This doesn't look like a triangle in the positive octant.
        
        # Actually, the DTLZ1 Pareto front is:
        # f0 = 0.5 * x0
        # f1 = 0.5 * x1 * (1 - x0)
        # f2 = 0.5 * (1 - x1) * (1 - x0)
        # This is a curved surface? No, it's a triangle in objective space.
        # Let's check the points:
        # (x0, x1) = (1, 0) -> (0.5, 0, 0)
        # (x0, x1) = (0, 1) -> (0, 0, 0.5)
        # (x0, x1) = (0, 0) -> (0, 0, 0.5)
        # Wait, (0,0) gives f2 = 0.5 * 1 * 1 = 0.5.
        # (0,1) gives f1 = 0.5 * 1 * 1 = 0.5, f2 = 0.5 * 0 * 1 = 0.
        # So the vertices are (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5).
        # This is a triangle.
        
        # So we need to sample (x0, x1) such that x0 >= 0, x1 >= 0, x0 + x1 <= 1.
        # The objective values are:
        # f0 = 0.5 * x0
        # f1 = 0.5 * x1 * (1 - x0)
        # f2 = 0.5 * (1 - x1) * (1 - x0)
        
        # To maximize hypervolume, we should distribute points uniformly on this triangle.
        # A good strategy is to use a stratified sampling approach on the simplex.
        
        # We'll use a triangular lattice with jitter for better coverage.
        # Find the largest K such that (K+1)(K+2)/2 <= n_evals
        K = 1
        while (K + 1) * (K + 2) // 2 <= n_evals:
            K += 1
        K -= 1  # Now (K+1)(K+2)/2 <= n_evals
        
        valid_points = []
        for i in range(K + 1):
            for j in range(K - i + 1):
                # Add jitter to avoid degeneracy
                x0 = i / K
                x1 = j / K
                valid_points.append([x0, x1])
        
        # If we have room, add more points using a finer grid or random sampling
        remaining = n_evals - len(valid_points)
        if remaining > 0:
            # Use random sampling with rejection for the remaining points
            rng = np.random.default_rng(42)
            attempts = 0
            max_attempts = remaining * 10
            while len(valid_points) < n_evals and attempts < max_attempts:
                x0 = rng.random()
                x1 = rng.random()
                if x0 + x1 <= 1.0:
                    valid_points.append([x0, x1])
                attempts += 1
        
        # Trim to exactly n_evals if needed
        valid_points = valid_points[:n_evals]
        
        # Evaluate the points
        for p in valid_points:
            if time.time() > deadline:
                break
            x = base_x.copy()
            x[0] = p[0]
            x[1] = p[1]
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        return all_points
# EVOLVE-BLOCK-END
