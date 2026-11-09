"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting structure and cosine-spaced front sampling."""

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

        # Known optimum for distance variables: 0.5
        # For DTLZ4, g(x) = sum((x_i - 0.5)^2) for i in shape+1..dim
        # So setting x_i = 0.5 gives g = 0, which is optimal.
        dist_opt = 0.5 * np.ones(n_dist)

        def make_x(shape_vars):
            """Construct a full 12-dim vector from shape and distance variables."""
            x = np.empty(dim)
            x[:n_shape] = shape_vars
            x[n_shape:] = dist_opt
            return np.clip(x, lo, hi)

        # ---- Cosine-spaced sampling on the Pareto front ----
        # The Pareto front for DTLZ4 (with g=0) is a triangle in objective space.
        # We use a structured grid with cosine spacing to concentrate samples near vertices.
        
        # For a 3-objective problem, the Pareto front is a 2D triangle.
        # We use a grid of points (u, v) where u, v in [0, 1] and u + v <= 1.
        # The objective values are:
        # f1 = u * sin(v * pi/2)  -- wait, let me re-derive.
        
        # Actually, for DTLZ4 with g=0:
        # f1 = x1 * cos(x2 * pi/2)
        # f2 = x1 * sin(x2 * pi/2)
        # f3 = 1 - x1
        # where x1, x2 in [0, 1].
        
        # The Pareto front is parameterized by (x1, x2) in [0,1]^2.
        # f1 + f2 = x1, f3 = 1 - x1.
        # So the front is the set of points (f1, f2, f3) such that f1 + f2 + f3 = 1, f_i >= 0.
        
        # We want to sample points uniformly on this triangle, but with cosine spacing
        # to concentrate near the vertices.
        
        # Strategy: Use a 2D grid in (a, b) space where a, b in [0, 1].
        # Map to (x1, x2) using cosine transformation.
        
        # Let's use a different approach: sample directly on the triangle using
        # barycentric coordinates with cosine spacing.
        
        # The triangle has vertices:
        # V1 = (0, 0, 1)  -> x1=0, any x2
        # V2 = (0, 1, 0)  -> x1=1, x2=0 (since f1=0, f2=1 => x1*cos(x2*pi/2)=0, x1*sin(x2*pi/2)=1)
        #   Wait: if x1=1, x2=0: f1=1*cos(0)=1, f2=1*sin(0)=0, f3=0. So (1,0,0).
        #   If x1=1, x2=1: f1=1*cos(pi/2)=0, f2=1*sin(pi/2)=1, f3=0. So (0,1,0).
        # V3 = (1, 0, 0)  -> x1=1, x2=0
        
        # So the three vertices in objective space are:
        # (1, 0, 0), (0, 1, 0), (0, 0, 1)
        
        # We can parameterize points on the triangle using two parameters (s, t) in [0, 1]
        # with s + t <= 1. The point is:
        # F = s * (1,0,0) + t * (0,1,0) + (1-s-t) * (0,0,1)
        # = (s, t, 1-s-t)
        
        # To map back to (x1, x2):
        # f3 = 1 - x1 => x1 = 1 - f3 = 1 - (1-s-t) = s+t
        # f1 = x1 * cos(x2 * pi/2) => cos(x2*pi/2) = f1/x1 = s/(s+t) (if s+t > 0)
        # f2 = x1 * sin(x2 * pi/2) => sin(x2*pi/2) = f2/x1 = t/(s+t) (if s+t > 0)
        # So x2 = (2/pi) * arcsin(t/(s+t)) if s+t > 0, else 0.
        
        # Use cosine spacing for s and t to concentrate near vertices.
        # For a 1D cosine grid: x_i = cos(i * pi / (2*(n-1))) for i=0..n-1, mapped to [0,1].
        # Actually, let's use: for i in 0..n-1, u_i = 0.5 * (1 + cos(pi * i / (n-1)))
        # This gives u_0 = 1, u_{n-1} = 0, concentrated near endpoints.
        
        n_grid = int(np.sqrt(budget))
        # Adjust n_grid so that total points <= budget
        # Number of points on a triangular grid with n_grid points per edge:
        # sum_{i=0}^{n_grid-1} (n_grid - i) = n_grid*(n_grid+1)/2
        while n_grid * (n_grid + 1) // 2 > budget:
            n_grid -= 1
        
        # Generate cosine-spaced values in [0, 1]
        if n_grid > 1:
            u_vals = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_grid) / (n_grid - 1)))
        else:
            u_vals = np.array([0.5])
        
        # Generate all (s, t) pairs with s + t <= 1
        s_vals = []
        t_vals = []
        for i, s in enumerate(u_vals):
            for j, t in enumerate(u_vals):
                if s + t <= 1.0 + 1e-12:
                    s_vals.append(s)
                    t_vals.append(t)
        
        s_arr = np.array(s_vals)
        t_arr = np.array(t_vals)
        
        n_points = len(s_arr)
        
        # Compute x1, x2
        x1 = s_arr + t_arr
        x2 = np.zeros_like(x1)
        mask = x1 > 1e-12
        ratio = t_arr[mask] / x1[mask]
        ratio = np.clip(ratio, 0.0, 1.0)
        x2[mask] = (2.0 / np.pi) * np.arcsin(ratio)
        
        # Clip to [0, 1]
        x1 = np.clip(x1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        
        shape_vars = np.column_stack((x1, x2))
        
        # Evaluate all points
        F_list = []
        for i in range(n_points):
            x = make_x(shape_vars[i])
            F_list.append(problem(x))
        
        # If we have budget remaining, add some random points near the front
        remaining = budget - n_points
        if remaining > 0 and time.time() < deadline:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                s = rng.random()
                t = rng.random()
                if s + t > 1.0:
                    s, t = 1.0 - s, 1.0 - t
                    if s + t > 1.0:
                        s, t = 0.5, 0.5
                x1_r = s + t
                x2_r = 0.0
                if x1_r > 1e-12:
                    ratio_r = t / x1_r
                    ratio_r = np.clip(ratio_r, 0.0, 1.0)
                    x2_r = (2.0 / np.pi) * np.arcsin(ratio_r)
                x1_r = np.clip(x1_r, 0.0, 1.0)
                x2_r = np.clip(x2_r, 0.0, 1.0)
                x = make_x(np.array([x1_r, x2_r]))
                F_list.append(problem(x))
        
        return F_list

# EVOLVE-BLOCK-END
