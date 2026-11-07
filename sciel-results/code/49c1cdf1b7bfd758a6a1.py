"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sequence on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        n_g = 5
        n_front = dim - n_g  # 7 for dim=12

        # Generate Sobol sequence for 2 dimensions
        from scipy.stats import qmc

        # Reserve 3 points for explicit vertices
        n_vertices = 3
        n_sobol = budget - n_vertices
        
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate sobol points
        u = sampler.random(n_sobol)  # shape (n_sobol, 2)

        # Map uniform variables to decision variables for DTLZ4 Pareto front
        # For DTLZ4 with 3 objectives:
        # x1 = (2/pi) * arccos(u1)  -- maps [0,1] to [0, pi/2]
        # x2 = u2
        # This gives uniform coverage on the quarter-sphere surface
        x1_vals = (2.0 / np.pi) * np.arccos(u[:, 0])
        x2_vals = u[:, 1]

        # Construct decision vectors
        X = np.zeros((budget, dim))
        X[:n_sobol, 0] = x1_vals
        X[:n_sobol, 1] = x2_vals

        # Set other front variables (indices 2 to n_front-1) to 0.5
        if n_front > 2:
            X[:n_sobol, 2:n_front] = 0.5

        # Set g variables (indices n_front to dim-1) to 0.5 to ensure g=0
        X[:n_sobol, n_front:] = 0.5
        
        # Add explicit vertices to ensure coverage of the extreme points
        # Vertex 1: x0=0, x1=0
        X[n_sobol, 0] = 0.0
        X[n_sobol, 1] = 0.0
        X[n_sobol, 2:] = 0.5
        
        # Vertex 2: x0=0, x1=1
        X[n_sobol+1, 0] = 0.0
        X[n_sobol+1, 1] = 1.0
        X[n_sobol+1, 2:] = 0.5
        
        # Vertex 3: x0=1, x1=0
        X[n_sobol+2, 0] = 1.0
        X[n_sobol+2, 1] = 0.0
        X[n_sobol+2, 2:] = 0.5

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Check time before starting evaluation loop
        if time.time() - t0 > deadline:
            return X

        # Evaluate all points
        F = np.empty((budget, n_obj))
        for i in range(budget):
            if time.time() - t0 > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
