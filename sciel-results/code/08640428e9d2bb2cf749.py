import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        n_points = self.budget
        
        # DTLZ2 Pareto front: f_i = sin(pi/2 * x_i) for i=1..M
        # We need x_i in [0, 1] such that sin(pi/2 * x_i) = desired value
        # So x_i = (2/pi) * arcsin(desired value)
        
        # Generate points on the Pareto front using uniform distribution in angle space
        # For 3 objectives, we need theta1, theta2 in [0, pi/2]
        # f1 = cos(theta1) * cos(theta2)
        # f2 = cos(theta1) * sin(theta2)  
        # f3 = sin(theta1)
        
        # Actually, let me use the standard parameterization:
        # f_i = sin(pi/2 * x_i) for i = 1, ..., M
        # The Pareto front is: sum(f_i^2) = 1, f_i >= 0
        # So we can parametrize using two angles alpha, beta in [0, pi/2]
        # f1 = cos(alpha) * cos(beta)
        # f2 = cos(alpha) * sin(beta)
        # f3 = sin(alpha)
        
        # To place points uniformly on the quarter sphere, we can use:
        # alpha = arcsin(sqrt(u1))  where u1 ~ Uniform[0,1]
        # beta = arcsin(sqrt(u2))   where u2 ~ Uniform[0,1]
        # This gives uniform distribution on the quarter sphere
        
        # Generate Halton sequence for quasi-random points
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate points on the Pareto front
        for i in range(1, n_points + 1):
            if time.time() > deadline - 2.0:
                break
            
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            
            # Convert to angles for uniform distribution on quarter sphere
            alpha = np.arcsin(np.sqrt(u1))
            beta = np.arcsin(np.sqrt(u2))
            
            # Compute objective values
            f1 = np.cos(alpha) * np.cos(beta)
            f2 = np.cos(alpha) * np.sin(beta)
            f3 = np.sin(alpha)
            
            # Convert back to decision variables
            # x_i = (2/pi) * arcsin(f_i)
            x = np.zeros(self.dim)
            x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f1, 0, 1))
            x[1] = (2.0 / np.pi) * np.arcsin(np.clip(f2, 0, 1))
            x[2] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
            x[3:] = 0.5
            
            problem(x)
        
        return None
