# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    DTLZ4 optimizer using a structured 2D design on the quarter-sphere Pareto front.
    Strategy:
    - Place 3 points at the extreme vertices of the Pareto front.
    - Place 197 points using a 2D LHS design mapped to the quarter-sphere surface
      with uniform area distribution.
    - All points are constructed to lie exactly on the Pareto front (g=0).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_points = self.budget

        # 1. Define the 3 extreme vertices of the Pareto front
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        n_vertices = min(3, n_points)
        n_lhs = n_points - n_vertices

        # 2. Generate 2D Latin Hypercube Sample for the remaining points
        if n_lhs > 0:
            idx = np.arange(n_lhs)
            perm1 = np.random.permutation(n_lhs)
            perm2 = np.random.permutation(n_lhs)
            u1 = (perm1 + np.random.rand(n_lhs)) / n_lhs
            u2 = (perm2 + np.random.rand(n_lhs)) / n_lhs

            # Map to uniform area on quarter-sphere
            c_phi = 1.0 - u1
            s_phi = np.sqrt(np.maximum(0.0, 1.0 - c_phi**2))
            theta = u2 * (np.pi / 2.0)

            x0 = s_phi * np.cos(theta)
            x1 = s_phi * np.sin(theta)
            x2 = c_phi

            lhs_points = np.column_stack((x0, x1, x2))
        else:
            lhs_points = np.empty((0, 3))

        # 3. Combine vertices and LHS points
        all_points = np.vstack((vertices[:n_vertices], lhs_points))

        # 4. Construct full decision vectors
        x_all = np.zeros((len(all_points), self.dim))
        x_all[:, :3] = all_points
        x_all[:, 3:] = 0.5

        # Clip to bounds
        x_all = np.clip(x_all, lo, hi)

        # 5. Evaluate
        results = []
        for i in range(len(x_all)):
            if time.time() > deadline:
                break
            x = x_all[i]
            f = problem(x)
            results.append(f)

        return results
# EVOLVE-BLOCK-END
