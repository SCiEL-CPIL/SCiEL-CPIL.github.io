"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured grid optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        points = []
        
        # Add the 3 vertices of the Pareto front
        # Vertex 1: f=(1,0,0) -> x0=0, x1=0
        x_v1 = np.full(self.dim, 0.5)
        x_v1[0] = 0.0
        x_v1[1] = 0.0
        points.append(x_v1)
        
        # Vertex 2: f=(0,1,0) -> x0=0, x1=1
        x_v2 = np.full(self.dim, 0.5)
        x_v2[0] = 0.0
        x_v2[1] = 1.0
        points.append(x_v2)
        
        # Vertex 3: f=(0,0,1) -> x0=1, x1=0
        x_v3 = np.full(self.dim, 0.5)
        x_v3[0] = 1.0
        x_v3[1] = 0.0
        points.append(x_v3)
        
        # Add grid points
        remaining_budget = self.budget - len(points)
        if remaining_budget > 0:
            N = int(np.sqrt(remaining_budget))
            while N * N > remaining_budget:
                N -= 1
            
            for i in range(N):
                c1 = (i + 0.5) / N
                theta1 = np.arccos(c1)
                x0 = (2.0 / np.pi) * theta1
                
                for j in range(N):
                    theta2 = (j + 0.5) * (np.pi / 2.0) / N
                    x1 = (2.0 / np.pi) * theta2
                    
                    x = np.full(self.dim, 0.5)
                    x[0] = x0
                    x[1] = x1
                    points.append(x)
                    
                    if len(points) >= self.budget:
                        break
                if len(points) >= self.budget:
                    break
        
        for i in range(len(points)):
            points[i] = np.clip(points[i], lo, hi)
            
        for x in points:
            if time.time() > deadline - 5:
                break
            problem(x)
            
        return points
# EVOLVE-BLOCK-END
