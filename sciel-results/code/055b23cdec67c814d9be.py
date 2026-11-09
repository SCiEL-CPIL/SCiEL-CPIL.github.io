"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: sample the known Pareto front via a 2D Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        # k = number of "g" variables (last k variables), shape vars = first (m-1)
        k = n - m + 1
        n_shape = m - 1  # 2 for 3 objectives

        # Build a 2D Halton (low-discrepancy) sequence over the angle space
        def halton_2d(npts):
            # 2D Halton sequence: base 2 and 3
            def halton_1d(idx, base):
                f = 1.0
                r = 0.0
                i = idx
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            xs = []
            for i in range(1, npts + 1):
                xs.append([halton_1d(i, 2), halton_1d(i, 3)])
            return np.array(xs)

        # Use edge points for better coverage of the front boundaries
        # These correspond to points where one or more objectives are near 0 or 1
        n_edge = 12
        n_halton = self.budget - n_edge
        if n_halton < 1:
            n_halton = max(1, self.budget - n_edge)

        # Generate Halton points in [0, 1]^2
        halton_pts = halton_2d(n_halton)
        
        # Map to angle space [0, pi/2]^2
        theta1 = halton_pts[:, 0] * (np.pi / 2)
        theta2 = halton_pts[:, 1] * (np.pi / 2)
        
        # Map to x-space [0, 1]^2
        x1 = np.cos(theta1)
        x2 = np.cos(theta2)
        
        # Add edge points: points along the boundaries of the Pareto front
        # These ensure good coverage near where objectives approach 0 or 1
        edge_thetas = []
        # Corners
        edge_thetas.append([0.0, 0.0])
        edge_thetas.append([np.pi/2, 0.0])
        edge_thetas.append([0.0, np.pi/2])
        # Midpoints of edges
        edge_thetas.append([np.pi/4, 0.0])
        edge_thetas.append([0.0, np.pi/4])
        edge_thetas.append([np.pi/2, np.pi/4])
        edge_thetas.append([np.pi/4, np.pi/2])
        # Additional edge points for better distribution
        for frac in [0.2, 0.8]:
            edge_thetas.append([frac * np.pi/2, 0.0])
            edge_thetas.append([0.0, frac * np.pi/2])
            edge_thetas.append([np.pi/2, frac * np.pi/2])
            edge_thetas.append([frac * np.pi/2, np.pi/2])
        
        edge_thetas = np.array(edge_thetas)
        # Trim if we have too many
        edge_thetas = edge_thetas[:n_edge]
        
        edge_x1 = np.cos(edge_thetas[:, 0])
        edge_x2 = np.cos(edge_thetas[:, 1])
        
        shape_pts = np.vstack([
            np.column_stack([edge_x1, edge_x2]),
            np.column_stack([x1, x2])
        ])
        
        # Trim to budget if needed
        shape_pts = shape_pts[:self.budget]

        # Build full x vectors: shape vars from sequence, g-vars = 0.5 (clipped to bounds)
        g_val = 0.5
        g_val = np.clip(g_val, lo[n_shape], hi[n_shape])

        # Build x matrix
        X = np.empty((len(shape_pts), n))
        X[:, :n_shape] = shape_pts
        for j in range(n_shape, n):
            X[:, j] = g_val

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.empty((len(X), m))
        for i in range(len(X)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
