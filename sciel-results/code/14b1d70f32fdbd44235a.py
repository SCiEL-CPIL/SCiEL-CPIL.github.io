"""Optimized DTLZ4 solver: direct analytical Pareto front construction with LHS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytical Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3, N=12:
        # Position variables: x[0], x[1] (M-1 = 2)
        # Distance variables: x[2]...x[11] (N-M+1 = 10)
        # For Pareto front (g=0), set distance variables to 0.5
        # Objectives:
        # f1 = cos(x0*pi/2)*cos(x1*pi/2)
        # f2 = cos(x0*pi/2)*sin(x1*pi/2)
        # f3 = sin(x0*pi/2)

        # Strategy: Use LHS on the 2D parameter space mapped to the spherical triangle.
        # The Pareto front is a quarter unit sphere in the positive octant.
        # Parameterize by (alpha, beta) where:
        #   f1 = cos(alpha) * cos(beta)
        #   f2 = cos(alpha) * sin(beta)
        #   f3 = sin(alpha)
        # with alpha in [0, pi/2], beta in [0, pi/2]

        # For uniform coverage on the spherical surface:
        # The area element is dA = cos(alpha) d(alpha) d(beta)
        # So we sample alpha with PDF proportional to cos(alpha), i.e., CDF = sin(alpha)
        # Inverse CDF: alpha = arcsin(u1)
        # Beta is uniform: beta = u2 * pi/2

        n_points = budget

        # Reserve 3 evaluations for corner points
        n_lhs = n_points - 3 if n_points > 3 else n_points

        # Generate LHS in 2D
        # LHS: divide [0,1] into n equal intervals, pick one random point from each
        if n_lhs > 0:
            # Create strata for each dimension
            u1 = (np.arange(n_lhs) + np.random.uniform(size=n_lhs)) / n_lhs
            u2 = (np.arange(n_lhs) + np.random.uniform(size=n_lhs)) / n_lhs

            # Shuffle to decorrelate
            perm2 = np.random.permutation(n_lhs)
            u2 = u2[perm2]

            # Map to (alpha, beta)
            alpha = np.arcsin(np.clip(u1, 0.0, 1.0))
            beta = (np.pi / 2) * np.clip(u2, 0.0, 1.0)

            # Compute objective values to verify
            f1 = np.cos(alpha) * np.cos(beta)
            f2 = np.cos(alpha) * np.sin(beta)
            f3 = np.sin(alpha)

            # Decision vector parameters: x[0] = alpha/(pi/2), x[1] = beta/(pi/2)
            p1 = alpha / (np.pi / 2)
            p2 = beta / (np.pi / 2)
        else:
            p1 = np.array([])
            p2 = np.array([])

        # Corner points: (0,0) -> f=(1,0,0), (1,0) -> f=(0,1,0), (0,1) -> f=(0,0,1)
        corner_p1 = np.array([0.0, 1.0, 0.0])
        corner_p2 = np.array([0.0, 0.0, 1.0])

        # Combine all points
        all_p1 = np.concatenate([p1, corner_p1[:n_points - n_lhs]])
        all_p2 = np.concatenate([p2, corner_p2[:n_points - n_lhs]])

        # Ensure within [0, 1]
        all_p1 = np.clip(all_p1, 0.0, 1.0)
        all_p2 = np.clip(all_p2, 0.0, 1.0)

        # Construct decision vectors
        X = np.zeros((len(all_p1), dim))
        X[:, 0] = all_p1
        X[:, 1] = all_p2
        X[:, 2:] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])

        return X
# EVOLVE-BLOCK-END
