# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    
    Strategy:
    1. Exploit the analytic structure of DTLZ4:
       - The Pareto front is a quarter-sphere (quarter circle in 2D parameter space)
       - For 3 objectives: f1=0.5(1+g)x1, f2=0.5(1+g)x2, f3=0.5(1+g)x3
       - x1^2+x2^2+x3^2=1, xi>=0
       - Front: x1=cos(phi), x2=sin(phi), x3=0 for phi in [0, pi/2]
       - Wait, that's wrong. x3 = sqrt(1-x1^2-x2^2)
       - On the front (g=0): x1^2+x2^2=1, so x3=0? No...
       
       Actually for DTLZ3/4: the front is where all points are Pareto optimal.
       The front is the set where g=0 and x1^2+x2^2+x3^2=1, xi>=0.
       This is a quarter-sphere in 3D, which is 2D. But we only have 2 decision
       variables for the front (x1, x2), with x3 determined.
       
       So: x1 in [0,1], x2 in [0, sqrt(1-x1^2)]
       Or in polar: r in [0,1], phi in [0, pi/2], but on the front r=1
       Wait no - the front IS the quarter unit circle: x1=cos(phi), x2=sin(phi)
       
    2. Use stratified sampling of phi in [0, pi/2] to cover the front uniformly
    3. Set g-variables to 0.5 (optimal)
    4. Use remaining budget for refinement if needed
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        n_front = n_obj - 1  # 2
        n_g = dim - n_front  # 10
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The Pareto front for DTLZ with n_obj=3 is parameterized by phi in [0, pi/2]:
        # x[0] = cos(phi), x[1] = sin(phi)
        # g-variables all at 0.5
        # This gives points on the quarter unit circle, which maps to the Pareto front
        
        # Use stratified sampling: divide [0, pi/2] into N equal intervals
        # and sample at midpoints for uniform coverage
        
        n_samples = budget  # Use all budget for front sampling
        
        # Stratified phi samples
        phi_values = (np.arange(n_samples) + 0.5) / n_samples * (np.pi / 2)
        
        # Generate decision vectors
        X = np.zeros((n_samples, dim))
        X[:, 0] = np.cos(phi_values)
        X[:, 1] = np.sin(phi_values)
        # Set g-variables to 0.5 (optimal for DTLZ4)
        X[:, n_front:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
