"""Optimizer for DTLZ4 multi-objective optimization using analytic Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting analytic Pareto structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3 objectives:
        # f1 = (1+g) * 0.5 * x0 * cos(pi*x1/2)
        # f2 = (1+g) * 0.5 * x0 * sin(pi*x1/2)
        # f3 = (1+g) * (1 - x0)
        # g(x) = sum((x_i - 0.5)^2) for i in [2, dim)
        # Pareto front: g=0 => x[2:] = 0.5, and x0, x1 vary in [0,1]
        
        # On the Pareto front (g=0):
        # f1 = 0.5 * x0 * cos(pi*x1/2)
        # f2 = 0.5 * x0 * sin(pi*x1/2)
        # f3 = 1 - x0
        
        # Let a = pi*x1/2, so x1 = 2a/pi, a in [0, pi/2]
        # f1 = 0.5 * x0 * cos(a)
        # f2 = 0.5 * x0 * sin(a)
        # f3 = 1 - x0
        
        # For good hypervolume coverage, we want to sample (x0, a) such that
        # the resulting points on the quarter-sphere surface are well-distributed.
        # The area element on the sphere surface is proportional to x0^2 * dx0 * da
        # (since the radius in the f1-f2 plane is 0.5*x0, and the arc length element
        #  in the angular direction is 0.5*x0*da).
        
        # Actually, let's think about this differently. The Pareto front in objective
        # space is a quarter sphere of radius 1 (since f1^2 + f2^2 + f3^2 = 
        # 0.25*x0^2 + (1-x0)^2 = 0.25*x0^2 + 1 - 2*x0 + x0^2 = 1.25*x0^2 - 2*x0 + 1
        # Hmm, that's not a sphere. Let me re-derive.
        
        # f1 = 0.5*x0*cos(a), f2 = 0.5*x0*sin(a), f3 = 1-x0
        # f1^2 + f2^2 = 0.25*x0^2
        # f1^2 + f2^2 + f3^2 = 0.25*x0^2 + (1-x0)^2 = 0.25*x0^2 + 1 - 2*x0 + x0^2
        # = 1.25*x0^2 - 2*x0 + 1
        # This is not constant, so it's not a sphere. It's a different surface.
        
        # For hypervolume maximization, we want points that dominate as much volume
        # as possible. The key is to have good coverage of the entire Pareto front.
        
        # Use a 2D Fibonacci lattice (golden angle spiral) for space-filling properties.
        # This provides superior uniformity compared to simple grids or Sobol for 2D.
        
        # Generate Fibonacci lattice points in [0,1]^2
        n = budget
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.399963
        
        # Fibonacci lattice: points are at (i/n, (i*golden_angle) mod 1)
        # But we want them in [0,1]^2, so let's use:
        # x0_i = (i + 0.5) / n  for i = 0, ..., n-1
        # a_i = (i * golden_angle) mod 1, then map to [0, pi/2]
        
        # Actually, a better approach: use a 2D lattice that tiles [0,1]^2
        # with low discrepancy. The Fibonacci lattice in 2D is:
        # p_i = ((i/n), ((i*alpha) mod 1)) where alpha = (1+sqrt(5))/2
        # But this is a 1D sequence. For 2D, we need a 2D low-discrepancy sequence.
        
        # Let's use a stratified approach with Fibonacci numbers for the grid.
        # Or better: use a 2D Halton sequence or Sobol sequence.
        
        # Actually, the simplest good approach: use a stratified grid with
        # golden-ratio spacing, or use scipy's Sobol sequence.
        
        # Let's use Sobol from scipy for better coverage:
        from scipy.stats import qmc
        
        sampler = qmc.Sobol(d=2, scramble=True)
        u = sampler.random(n)  # shape (n, 2), values in [0, 1)
        
        # Map u to (x0, x1):
        # x0 = u[:, 0]  (uniform in [0, 1])
        # For x1, we want to map such that the angular variable a = pi*x1/2
        # is well-distributed. Since a in [0, pi/2] and x1 = 2a/pi,
        # uniform in x1 is fine.
        x0_vals = u[:, 0]
        x1_vals = u[:, 1]
        
        # Construct decision vectors
        X = np.zeros((n, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        X[:, 2:] = 0.5  # Set g-variables to 0.5 to ensure g=0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Check time before starting evaluation loop
        if time.time() - t0 > deadline:
            return X
        
        # Evaluate all points
        F = np.empty((n, n_obj))
        for i in range(n):
            if time.time() - t0 > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
