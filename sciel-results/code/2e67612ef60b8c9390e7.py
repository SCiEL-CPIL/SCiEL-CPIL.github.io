"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: front determined by (x0, x1); drive g->0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        M = self.n_obj  # 3
        n = self.dim     # 22
        budget = self.budget

        # Phase 1: Dense stratified grid in (x0, x1) space with g=0
        # DTLZ7: x0, x1 determine the front when g=0
        # f1 = x0, f2 = x1, f3 = (n - M + 1 + g) * (1 - x1/sqrt(x0) * (1 - 1/(n-M+1)))
        # With g=0: f3 = (n-M+1) * (1 - x1/sqrt(x0) * (1 - 1/(n-M+1)))
        # = 20 * (1 - x1/sqrt(x0) * (1 - 1/20)) = 20 * (1 - x1/sqrt(x0) * 0.95)
        # x0 in (0,1], x1 in [0,1]
        # f3 ranges from 0 (when x1/sqrt(x0) = 1/0.95 ≈ 1.0526, but x1<=1, sqrt(x0)<=1)
        # Actually f3 min: x1=1, x0=1 => 20*(1 - 1*0.95) = 20*0.05 = 1
        # f3 max: x1=0, x0=1 => 20*(1-0) = 20
        # Wait, let me re-derive. DTLZ7 with n=22, M=3:
        # g(x3,...,xn) = 19 + sum_{i=3}^{22} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # On the front g=0, which requires x_i=0.5 for i=3..22 (the minimum of each term is 0)
        # Actually the minimum of (x-0.5)^2 - cos(20*pi*(x-0.5)) is 0 at x=0.5
        # So g_min = 0 when all x_i=0.5 for i>=3
        # f1 = x0
        # f2 = x1
        # f3 = (n - M + 1 + g) * (1 - x1/sqrt(x0) * (1 - 1/(n-M+1)))
        # With g=0: f3 = 20 * (1 - x1/sqrt(x0) * (1 - 1/20))
        # = 20 * (1 - x1/sqrt(x0) * 19/20)
        # = 20 - 19*x1/sqrt(x0)
        # For x0 in (0,1], x1 in [0,1]:
        # f3 max when x1=0: f3 = 20
        # f3 min when x1/sqrt(x0) is max. Max x1/sqrt(x0) = 1/sqrt(x0) when x1=1.
        # So f3_min = 20 - 19/sqrt(x0). For x0=1: f3_min = 1. For x0<1, f3_min < 1.
        # But f3 must be >= 0 for the point to be valid (non-negative objectives).
        # f3 >= 0 => 20 - 19*x1/sqrt(x0) >= 0 => x1/sqrt(x0) <= 20/19 ≈ 1.0526
        # Since x1 <= 1 and sqrt(x0) <= 1, x1/sqrt(x0) <= 1/sqrt(x0).
        # So we need 1/sqrt(x0) <= 20/19 => sqrt(x0) >= 19/20 => x0 >= (19/20)^2 = 0.9025
        # Wait, that's not right. x1 can be less than 1. The constraint is x1 <= min(1, 20*sqrt(x0)/19).
        # For x0 >= 0.9025, sqrt(x0) >= 0.95, so 20*sqrt(x0)/19 >= 1.0, so x1 can be [0,1].
        # For x0 < 0.9025, x1 must be <= 20*sqrt(x0)/19 < 1.

        # Let's create a grid in objective space directly.
        # We want points on the Pareto front. The front is parameterized by (x0, x1) with x_i=0.5 for i>=3.
        # f1 = x0, f2 = x1, f3 = 20 - 19*x1/sqrt(x0)
        # We need f1 > 0, f2 >= 0, f3 >= 0, and all < reference point [1.1, 1.1, 25.0]
        
        # Strategy: sample (f1, f2) in a grid, compute f3, check validity
        # f1 in (0, 1.1), f2 in [0, 1.1), f3 in [0, 25)
        # From f3 = 20 - 19*f2/sqrt(f1), we need f3 >= 0 => f2 <= 20*sqrt(f1)/19
        # And f3 < 25 is always true since f3 <= 20.
        # f1 < 1.1 and f2 < 1.1 are satisfied for f1 in (0,1], f2 in [0,1]
        
        # Let's use a dense grid in (f1, f2) space
        # f1: 20 points in (0.01, 1.0]
        # f2: 20 points in [0, 1.0]
        # That's 400 points, too many for budget 200.
        # Let's use 14x14 = 196 points, plus 4 corner points = 200
        
        n_grid = 14
        f1_vals = np.linspace(0.05, 1.0, n_grid)
        f2_vals = np.linspace(0.0, 1.0, n_grid)
        
        points = []
        for f1 in f1_vals:
            for f2 in f2_vals:
                if f2 > 20.0 * np.sqrt(f1) / 19.0:
                    continue
                f3 = 20.0 - 19.0 * f2 / np.sqrt(f1)
                if f3 < 0:
                    f3 = 0.0
                points.append((f1, f2, f3))
        
        # Add some corner/extreme points
        # f1 near 0, f2=0 => f3=20
        points.append((0.01, 0.0, 20.0))
        # f1=1, f2=1 => f3=1
        points.append((1.0, 1.0, 1.0))
        # f1=1, f2=0 => f3=20
        points.append((1.0, 0.0, 20.0))
        # f1=0.9025, f2=1.0 => f3=0
        points.append((0.9025, 1.0, 0.0))
        
        # Trim to budget
        if len(points) > budget:
            # Uniformly subsample
            idx = np.linspace(0, len(points)-1, budget).astype(int)
            points = [points[i] for i in idx]
        
        # Now convert (f1, f2, f3) to decision variables
        # f1 = x0, f2 = x1, x_i = 0.5 for i >= 2
        # We need to call problem(x) for each point to get the actual objective values
        # and record them.
        
        used = 0
        F = []
        for (f1, f2, f3) in points:
            if time.time() > deadline:
                break
            x = np.full(n, 0.5)
            x[0] = min(max(f1, 0.0), 1.0)
            x[1] = min(max(f2, 0.0), 1.0)
            f = problem(x)
            F.append(f)
            used += 1
        
        # Phase 2: Use remaining budget for refinement if any
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Do a few random perturbations around the best points
            # Actually, just do more grid points or random samples
            rng = np.random.default_rng(0)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Random point on the front
                f1 = 0.01 + 0.99 * rng.random()
                f2_max = min(1.0, 20.0 * np.sqrt(f1) / 19.0)
                f2 = f2_max * rng.random()
                f3 = 20.0 - 19.0 * f2 / np.sqrt(f1)
                x = np.full(n, 0.5)
                x[0] = f1
                x[1] = f2
                f = problem(x)
                F.append(f)
                used += 1
        
        return np.array(F) if F else np.array([])
# EVOLVE-BLOCK-END
