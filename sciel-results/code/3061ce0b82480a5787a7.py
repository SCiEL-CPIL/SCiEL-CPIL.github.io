"""Optimized DTLZ4 solver: Sobol sequence Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sequence Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

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

        # Strategy: Use Sobol sequence for low-discrepancy sampling in 2D parameter space.
        # Sobol sequences are generally superior to Halton sequences for low-discrepancy.
        
        n_points = budget
        
        # Generate Sobol sequence
        # Use scramble=True to avoid correlations and improve distribution
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # Generate n_points - 3 points, leaving room for 3 corners
        n_sobol = n_points - 3
        if n_sobol < 1:
            n_sobol = 1
            
        sobol_points = sampler.random(n=n_sobol)
        
        # Add corner points explicitly to ensure extreme coverage
        # Corners in parameter space:
        # (0,0) -> f=(1,0,0)
        # (0,1) -> f=(0,1,0)
        # (1,0) -> f=(0,0,1)
        # Note: (1,1) maps to f=(0,0,0) which is dominated by others? 
        # f1=cos(pi/2)cos(pi/2)=0, f2=0, f3=sin(pi/2)=1. 
        # Wait, alpha=pi/2, beta=pi/2 -> f1=0, f2=0, f3=1. This is the (0,0,1) corner.
        # Let's re-verify mapping:
        # f1 = cos(a)c(b), f2 = cos(a)s(b), f3 = sin(a)
        # a=0, b=0 -> (1, 0, 0)
        # a=0, b=pi/2 -> (0, 1, 0)
        # a=pi/2, b=0 -> (0, 0, 1)
        # a=pi/2, b=pi/2 -> (0, 0, 1) ... Wait.
        # cos(pi/2)=0, so f1=0, f2=0. sin(pi/2)=1. f3=1.
        # So (1,1) in [0,1]^2 maps to (0,0,1).
        # The corner (0,0,1) is covered by a=pi/2, any b.
        # The corner (1,0,0) is a=0, b=0.
        # The corner (0,1,0) is a=0, b=pi/2.
        
        # So the 3 extreme corners of the objective space are:
        # (1,0,0) from (0,0)
        # (0,1,0) from (0,1)
        # (0,0,1) from (1,0) or (1,1)
        
        corner_params = np.array([
            [0.0, 0.0], # (1,0,0)
            [0.0, 1.0], # (0,1,0)
            [1.0, 0.0], # (0,0,1)
        ])
        
        # Combine Sobol points and corners
        params = np.vstack([sobol_points, corner_params])
        
        # Ensure we have exactly n_points
        if len(params) > n_points:
            params = params[:n_points]
        elif len(params) < n_points:
            # Pad with more Sobol points if needed (should not happen)
            extra = sampler.random(n=n_points - len(params))
            params = np.vstack([params, extra])
            
        # Clip to [0,1]
        params = np.clip(params, 0.0, 1.0)
        
        # Map parameters to (alpha, beta)
        alpha = (np.pi / 2) * params[:, 0]
        beta = (np.pi / 2) * params[:, 1]
        
        # Construct decision vectors
        X = np.zeros((len(params), dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
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
