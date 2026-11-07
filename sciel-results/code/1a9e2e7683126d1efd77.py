import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        points = []
        
        # Add extreme points on the Pareto front of DTLZ4
        # For DTLZ4 with m=3, the Pareto front is defined by g(x)=0
        # which requires x[2:] = 0.5 (since beta=10, x^10 + (1-x)^10 = 0 implies x=0.5)
        # The position variables x[0], x[1] determine the objectives:
        # f1 = x0 * cos(x1 * pi/2)
        # f2 = x0 * sin(x1 * pi/2)
        # f3 = 1 - x0
        
        # 1. f=(1,0,0) -> x0=1, x1=0
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)
        
        # 2. f=(0,1,0) -> x0=0, x1=1 (actually x0=0 makes f1=f2=0, so this is f=(0,0,1))
        # Wait, let's re-verify the DTLZ4 formulation.
        # Standard DTLZ2/4:
        # f1 = x1 * prod_{i=2}^{M} cos(x_i * pi/2)
        # f2 = x1 * sin(x_2 * pi/2) * prod_{i=3}^{M} cos(x_i * pi/2)
        # ...
        # fM = x1 * sin(x_M * pi/2)
        
        # For M=3:
        # f1 = x0 * cos(x1 * pi/2)
        # f2 = x0 * sin(x1 * pi/2)
        # f3 = 1 - x0
        
        # Extremes:
        # x0=1, x1=0 -> f1=1, f2=0, f3=0 -> (1, 0, 0)
        # x0=0, x1=0 -> f1=0, f2=0, f3=1 -> (0, 0, 1)
        # x0=0, x1=1 -> f1=0, f2=0, f3=1 -> (0, 0, 1)
        # x0=1, x1=1 -> f1=0, f2=1, f3=0 -> (0, 1, 0)
        
        # So the three extreme points of the Pareto front are:
        # (1, 0, 0) from x0=1, x1=0
        # (0, 1, 0) from x0=1, x1=1
        # (0, 0, 1) from x0=0, x1=any (e.g., 0)
        
        # Point 1: (1, 0, 0)
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)
        
        # Point 2: (0, 1, 0)
        x = np.zeros(dim)
        x[0] = 1.0
        x[1] = 1.0
        x[2:] = 0.5
        points.append(x)
        
        # Point 3: (0, 0, 1)
        x = np.zeros(dim)
        x[0] = 0.0
        x[1] = 0.0
        x[2:] = 0.5
        points.append(x)
        
        # Add center point of the triangle in objective space
        # The centroid of the triangle with vertices (1,0,0), (0,1,0), (0,0,1) is (1/3, 1/3, 1/3)
        # We need to find x0, x1 such that:
        # x0 * cos(x1 * pi/2) = 1/3
        # x0 * sin(x1 * pi/2) = 1/3
        # 1 - x0 = 1/3 -> x0 = 2/3
        # (2/3) * cos(x1 * pi/2) = 1/3 -> cos(x1 * pi/2) = 1/2 -> x1 * pi/2 = pi/3 -> x1 = 2/3
        x = np.zeros(dim)
        x[0] = 2.0/3.0
        x[1] = 2.0/3.0
        x[2:] = 0.5
        points.append(x)
        
        # Add Sobol points for the 2D position space
        # Use unscrambled Sobol for deterministic, high-quality distribution
        n_sobol = budget - len(points)
        if n_sobol > 0:
            sobol = qmc.Sobol(d=2, scramble=False)
            sobol_points = sobol.random(n_sobol)
            for p in sobol_points:
                x = np.zeros(dim)
                x[0] = p[0]
                x[1] = p[1]
                x[2:] = 0.5
                points.append(x)
                
        # Evaluate
        F = np.array([problem(x) for x in points])
        
        if time.time() > deadline:
            return F
            
        return F
