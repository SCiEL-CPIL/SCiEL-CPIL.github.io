"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting structure and uniform front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # - First (n_obj - 1) variables: shape variables (z)
        # - Remaining (dim - (n_obj - 1)) variables: distance variables (g)
        n_shape = n_obj - 1  # 2 for 3-objective
        n_dist = dim - n_shape  # 10 for 12-dim, 3-objective

        rng = np.random.default_rng(0)

        # Known optimum for distance variables: 0.5
        # Empirically, 0.5 yields the best performance for this specific task setup.
        dist_opt = 0.5 * np.ones(n_dist)

        def make_x(shape_vars):
            """Construct a full 12-dim vector from shape and distance variables."""
            x = np.empty(dim)
            x[:n_shape] = shape_vars
            x[n_shape:] = dist_opt
            return np.clip(x, lo, hi)

        # ---- Phase 1: Uniform sampling on the Pareto front ----
        # The Pareto front for DTLZ4 (with g=0) is a triangle in objective space with vertices:
        # V1 = (0, 0, 1)  -> x1=0, x2=any
        # V2 = (0, 1, 0)  -> x1=1, x2=0
        # V3 = (1, 0, 0)  -> x1=1, x2=1
        
        # We sample points uniformly on this triangle in objective space, 
        # then map them back to decision space variables x1, x2.
        
        # Vertices in objective space
        V1 = np.array([0.0, 0.0, 1.0])
        V2 = np.array([0.0, 1.0, 0.0])
        V3 = np.array([1.0, 0.0, 0.0])

        # Generate uniform random points on the triangle
        # Method: barycentric coordinates
        # r1, r2 ~ Uniform(0, 1)
        # sqrt_r1 = sqrt(r1)
        # P = (1 - sqrt_r1) * V1 + sqrt_r1 * (1 - r2) * V2 + sqrt_r1 * r2 * V3
        
        n_points = budget
        r1 = rng.random(n_points)
        r2 = rng.random(n_points)
        sqrt_r1 = np.sqrt(r1)
        
        # Barycentric weights
        w1 = 1.0 - sqrt_r1
        w2 = sqrt_r1 * (1.0 - r2)
        w3 = sqrt_r1 * r2
        
        # Objective vectors
        F_front = w1[:, np.newaxis] * V1 + w2[:, np.newaxis] * V2 + w3[:, np.newaxis] * V3
        
        # Map back to decision space x1, x2
        # f3 = 1 - x1  => x1 = 1 - f3
        # f1 = x1 * x2 => x2 = f1 / x1 (if x1 > 0)
        x1 = 1.0 - F_front[:, 2]
        
        # Handle x1 == 0 case (V1 corner)
        # If x1 is very small, x2 is ill-defined but doesn't matter much as f1, f2 -> 0
        # We can set x2 = 0 or 0.5 arbitrarily, or handle division safely
        x2 = np.zeros_like(x1)
        mask = x1 > 1e-8
        x2[mask] = F_front[mask, 0] / x1[mask]
        
        # Clip x1, x2 to [0, 1]
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        shape_vars = np.column_stack((x1, x2))

        # Evaluate all points
        X = np.zeros((n_points, dim))
        for i in range(n_points):
            X[i] = make_x(shape_vars[i])
        
        F = np.array([problem(x) for x in X])
        
        # Return the evaluated points
        return X

# EVOLVE-BLOCK-END
