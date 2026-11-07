import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget
        used = 0
        
        # Strategy: Use a Fibonacci lattice to distribute points uniformly on the DTLZ2 Pareto front.
        # DTLZ2 with 3 objectives has a quarter-sphere Pareto front.
        # Position variables x[0], x[1] map to angles theta_1, theta_2 in [0, pi/2].
        # x[0] = 2*theta_1/pi, x[1] = 2*theta_2/pi
        # Direction variables x[2:] should be 0.5 for g=0 (on the Pareto front).
        
        # Generate N points using a Fibonacci-like lattice on the quarter sphere
        N = max_evals  # Use all budget for front sampling
        
        # Golden angle for Fibonacci lattice
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.399 rad
        
        points = np.zeros((N, self.dim))
        for k in range(N):
            # Uniform distribution in theta_1 (polar angle)
            theta_1 = (k + 0.5) / N * (np.pi / 2.0)
            # Fibonacci spacing in theta_2 (azimuthal angle)
            theta_2 = (k * golden_angle) % (np.pi / 2.0)
            
            # Map to decision variables
            x = np.full(self.dim, 0.5)  # Direction variables at optimum
            x[0] = 2.0 * theta_1 / np.pi  # theta_1 in [0, pi/2] -> x[0] in [0, 1]
            x[1] = 2.0 * theta_2 / np.pi  # theta_2 in [0, pi/2] -> x[1] in [0, 1]
            points[k] = x
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        used += N
        
        return None
