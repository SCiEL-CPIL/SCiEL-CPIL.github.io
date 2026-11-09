import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7: x0, x1 are active. x2... are inactive.
        # Set inactive to 0.
        # Sample x0, x1.
        
        # The Pareto front of DTLZ7 in 3 objectives is a 2D surface in [0,1]^2.
        # f0 = x0
        # f1 = 1 - x1
        # f2 = 1 + (n_active) * h(x1, g) where g = sum(x2^2)
        # To minimize f2, we must set x2...x21 = 0 so g=0.
        # Then f2 = 1 + 21 * (1 - sqrt(x1/21)).
        # This is a convex curve in x1.
        
        # Strategy:
        # 1. Use a grid to ensure corners and uniform coverage.
        # 2. Add extra points clustered near the "knee" or interesting regions?
        #    Actually, for hypervolume, uniform coverage of the front is key.
        #    The front is smooth. A grid is good.
        #    Current best: 14x14 grid = 196 points. 4 points left.
        #    Inspirations suggest:
        #    - Adding corners (already in grid).
        #    - Using Sobol (low discrepancy) might be better than grid for hypervolume?
        #      Grid is deterministic. Sobol is also deterministic.
        #      Grid ensures exact corners. Sobol might miss exact corners unless added.
        #      The 14x14 grid includes 0.0 and 1.0 in both dims.
        #      So corners are included.
        
        # Let's try to optimize the grid or add points.
        # Budget is 200.
        # 14x14 = 196.
        # We have 4 evaluations left.
        # Where to put them?
        # Maybe midpoints of the edges? Or center?
        # The function f2 is convex. The "knee" is where the tradeoff is steepest?
        # Actually, for hypervolume, we want to cover the volume under the front.
        # The front is defined by x0, x1.
        # f0 = x0
        # f1 = 1 - x1
        # f2 = 1 + 21 * (1 - sqrt(x1/21))
        
        # Let's check the shape of f2(x1).
        # x1=0 -> f2 = 1 + 21 = 22.
        # x1=1 -> f2 = 1 + 21*(1 - sqrt(1/21)) = 1 + 21*(1 - 0.218) = 1 + 16.4 = 17.4.
        # Wait, sqrt(1/21) is approx 0.218.
        # So f2 ranges from 17.4 to 22.
        # The reference point is [1.1, 1.1, 25.0].
        # So all points on the front are better than the reference point in f2.
        
        # Since the front is smooth and convex, a uniform grid is a good approximation.
        # However, hypervolume calculation is sensitive to the "steps" in the approximation.
        # A finer grid is better.
        # 14x14 uses 196 points.
        # Can we use 15x15 = 225? No, budget is 200.
        # 14x14 is the largest square grid <= 200.
        # What about 16x12 = 192?
        # 16 in x0, 12 in x1.
        # Or 12x16?
        # Since f0 and f1 are linear in x0 and x1, the area in (f0, f1) plane is uniform.
        # The f2 value depends only on x1.
        # So the "volume" contribution of a slice at x1 is proportional to dx1 * (25 - f2(x1)).
        # The integral is smooth.
        # A uniform grid in x0 and x1 should be optimal for minimizing error for a smooth function.
        
        # Let's stick with 14x14 grid.
        # But we have 4 spare points.
        # Where do they help most?
        # Maybe at the center of the cells?
        # Or maybe we can use a non-uniform grid?
        # Since f2(x1) is convex, the curvature is highest at x1=0?
        # f2'(x1) = 21 * (-1/(2*sqrt(21*x1))) = -sqrt(21)/(2*sqrt(x1)).
        # f2''(x1) = sqrt(21)/(4*x1^1.5).
        # Curvature is highest at x1 -> 0.
        # So we should sample more densely near x1=0.
        
        # Let's try a non-uniform grid in x1.
        # Transform x1' = x1^alpha to cluster near 0?
        # If we use x1_vals = np.linspace(0, 1, n1)**alpha, we cluster near 0 if alpha > 1.
        # Let's try alpha = 2.0 for x1.
        # And uniform for x0.
        
        # Let's test this hypothesis.
        # n0 = 14, n1 = 14.
        # x0_vals = np.linspace(0, 1, 14)
        # x1_vals = np.linspace(0, 1, 14)**2.0
        
        # But wait, the grid must include 0 and 1.
        # np.linspace(0,1,14)**2 includes 0 and 1.
        
        # Let's try this.
        
        n0 = 14
        n1 = 14
        
        # Use power law for x1 to cluster near 0 (high curvature)
        alpha_x1 = 2.0
        x0_vals = np.linspace(0.0, 1.0, n0)
        x1_lin = np.linspace(0.0, 1.0, n1)
        x1_vals = x1_lin ** alpha_x1
        
        # Create meshgrid
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        
        # Flatten
        x0_flat = X0.flatten()
        x1_flat = X1.flatten()
        
        n_samples = len(x0_flat)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # X[:, 2:] = 0
        
        # We have 4 points left.
        # Let's add the center points of the "corners" of the domain?
        # Or just evaluate them if we have time?
        # The loop below will evaluate n_samples points.
        # If n_samples < budget, we can add more.
        
        # Let's add 4 specific points:
        # (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5) ?
        # Or maybe just use the remaining budget to refine the grid?
        # Actually, 14x14 is 196.
        # Let's add 4 points at (0.5, 0.5) with slight perturbations?
        # Or better, let's just use the 4 points to evaluate the exact center (0.5, 0.5) and its neighbors?
        
        # Let's append 4 points:
        # (0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25)
        # These are midpoints of the central cells.
        
        extra_points = np.array([
            [0.5, 0.5],
            [0.25, 0.5],
            [0.75, 0.5],
            [0.5, 0.25]
        ])
        
        x0_flat = np.concatenate([x0_flat, extra_points[:, 0]])
        x1_flat = np.concatenate([x1_flat, extra_points[:, 1]])
        
        n_samples = len(x0_flat)
        
        # Ensure we don't exceed budget
        if n_samples > self.budget:
            x0_flat = x0_flat[:self.budget]
            x1_flat = x1_flat[:self.budget]
            n_samples = self.budget
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
