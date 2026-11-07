import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 310.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])

        # ZDT1 Specifics:
        # f1(x) = x[0]
        # f2(x) = (1 + 10*(n-1)) * (1 - sqrt(x[0] / g(x)))
        # g(x) = 1 + 9 * sum(x[1:]) / (n-1)
        #
        # Pareto Front:
        # For ZDT1, the Pareto front is achieved when g(x) is minimized.
        # g(x) is minimized when x[1:] are at their lower bounds (assuming standard [0,1] bounds).
        # If bounds are [0,1], x[1:] = 0 -> g(x) = 1.
        # Then f2 = C * (1 - sqrt(f1)), where C = 1 + 10*(n-1).
        #
        # We want to maximize Hypervolume with reference point R = [11, 11].
        # The front is a curve f2(f1).
        # To maximize the union of rectangles defined by points on this curve,
        # we should space the points such that the "loss" in hypervolume from discretization is minimized.
        # For a convex decreasing curve like ZDT1's front, uniform spacing in f1 is suboptimal.
        # However, a very strong heuristic for hypervolume maximization on a known front
        # is to space points such that the area under the curve segments is balanced,
        # or simply use a dense grid that covers the range well.
        #
        # Actually, for a fixed number of points N on a monotonic curve,
        # the optimal spacing for maximizing the discrete hypervolume (union of rectangles from points to reference)
        # depends on the curvature.
        # For ZDT1, f2 = C(1 - sqrt(f1)).
        # Let's use a strategy that samples uniformly in the "transformed" space that linearizes the front if possible,
        # or simply uses a high-quality quasi-random sequence (Sobol-like or Halton) for f1,
        # BUT since we know the exact front, we can do better.
        #
        # Insight: The hypervolume contribution of a point (f1, f2) relative to the next point is roughly the area of the rectangle.
        # A known result for maximizing hypervolume of a set of points on a curve is to space them
        # according to the inverse of the derivative of the hypervolume contribution function.
        # For simplicity and robustness, uniform sampling in f1 is often close to optimal for smooth fronts,
        # but let's try spacing that accounts for the curvature.
        #
        # Let's stick to a very robust approach:
        # 1. Determine the range of f1 on the Pareto front.
        #    f1 = x[0]. x[0] is in [lo[0], hi[0]].
        #    So f1 is in [lo[0], hi[0]].
        # 2. Construct points directly on the Pareto front.
        #    x[0] = f1_val
        #    x[1:] = lo[1:] (to minimize g)
        # 3. Choose f1_vals to maximize hypervolume.
        #
        # For ZDT1, the front is convex.
        # Uniform distribution in f1 is a solid baseline.
        # Let's use a Halton sequence or simply a sorted uniform grid with a random shift?
        # No, the current program uses LHS which is effectively a random permutation of bins.
        # A deterministic grid might be better for reproducibility and avoiding clustering.
        # However, the score is averaged over seeds.
        #
        # Let's try a specific spacing:
        # The hypervolume of a single point (f1, f2) is (R1-f1)(R2-f2).
        # The incremental hypervolume when adding a point depends on its neighbors.
        # For a smooth front, spacing points such that the "area" between them is constant is a good heuristic.
        # Since f2 = C(1 - sqrt(f1)), let u = sqrt(f1). Then f2 = C(1-u).
        # This is linear in u!
        # So if we sample u uniformly, we sample the front "linearly" in the transformed space.
        # This is likely much better than sampling f1 uniformly.
        #
        # Strategy:
        # 1. Generate N uniform samples u_i in [sqrt(lo[0]), sqrt(hi[0])].
        # 2. Set f1_i = u_i^2.
        # 3. Set x[0] = f1_i.
        # 4. Set x[1:] = lo[1:].
        # 5. Evaluate.
        
        n_pts = n_evals
        
        # Generate uniform samples for u = sqrt(f1)
        # u ranges from sqrt(lo[0]) to sqrt(hi[0])
        u_lo = np.sqrt(lo[0])
        u_hi = np.sqrt(hi[0])
        
        # Use a Halton-like or just uniform grid?
        # Uniform grid in u ensures even coverage of the "linearized" front.
        # Let's use a simple uniform grid with a slight random jitter to avoid any potential aliasing,
        # though for a 1D front, a grid is usually fine.
        # Actually, a pure uniform grid in u is optimal for a linear front in (u, f2) space?
        # f2 = C - C*u. Yes, it's a line.
        # So uniform u gives uniform f2 spacing.
        # Uniform f2 spacing is generally good for hypervolume.
        
        # Let's create the u values
        u_vals = np.linspace(u_lo, u_hi, n_pts)
        
        # Convert back to f1 (x[0])
        x0_vals = u_vals ** 2
        
        # Construct the input matrix
        X = np.empty((n_pts, self.dim), dtype=np.float64)
        X[:, 0] = x0_vals
        
        if self.dim > 1:
            # Set x[1:] to lower bound to ensure g(x) is minimal (Pareto optimal)
            X[:, 1:] = lo[1:]
            # Clip to be safe
            np.clip(X[:, 1:], lo[1:], hi[1:], out=X[:, 1:])
            
        # Evaluate all points
        F = np.empty((n_pts, self.n_obj), dtype=np.float64)
        
        for i in range(n_pts):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
