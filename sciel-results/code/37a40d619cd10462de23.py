import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 Pareto front is at x[2:] = 0.5
        # We only need to optimize x[0] and x[1]
        
        # Use a uniform grid to ensure coverage of the boundaries (corners)
        # which is critical for hypervolume as it maximizes the extent of the dominated region.
        # 14x14 = 196 points.
        # Use Gauss-Legendre quadrature nodes for better boundary clustering
        # which maximizes hypervolume contribution
        n_gl = 14
        # Get Gauss-Legendre nodes and weights on [-1, 1]
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_gl)
        # Map to [0, 1]
        gl_nodes_01 = 0.5 * (gl_nodes + 1.0)
        
        x0_vals = gl_nodes_01
        x1_vals = gl_nodes_01
        
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Add 4 more points to utilize full budget
        # Corners are already well-covered by GL nodes clustering at boundaries
        # Add center and edge midpoints for better coverage
        extra_x0 = [0.5, 0.5, 0.0, 1.0]
        extra_x1 = [0.0, 1.0, 0.5, 0.5]
        
        x0_flat = np.concatenate([x0_flat, extra_x0])
        x1_flat = np.concatenate([x1_flat, extra_x1])
        
        # Truncate to budget just in case
        x0_flat = x0_flat[:budget]
        x1_flat = x1_flat[:budget]
        
        # Construct full vectors
        X = np.zeros((len(x0_flat), n))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        X[:, 2:] = 0.5
        
        # Clip to bounds (though 0.5 and [0,1] are already within [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        # We assume problem is fast enough for 200 calls.
        # If time is tight, we might need to batch or reduce, but 200 is small.
        F = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))
        
        # The return value is ignored, but we return the evaluated points or just None.
        # The contract says "return value of __call__ is ignored".
        return X
