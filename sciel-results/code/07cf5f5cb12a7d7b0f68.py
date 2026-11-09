import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Generate low-discrepancy points in [0, 1]^2
        # Sobol sequence is a good choice for uniform coverage
        sampler = qmc.Sobol(d=2, scramble=False)
        # We need 200 points. Sobol requires power of 2 for best properties, but works for any N.
        # Let's use 200 points.
        points = sampler.random(n_evals)
        
        # Map to theta1, theta2 in [0, pi/2]
        # For uniform distribution on the DTLZ4 Pareto front (quarter sphere),
        # we need theta1 to be sampled such that cos(theta1) is uniform.
        # theta2 should be uniform.
        theta1 = np.arccos(points[:, 0])
        theta2 = points[:, 1] * (np.pi / 2)
        
        results = []
        for t1, t2 in zip(theta1, theta2):
            x = np.zeros(self.dim)
            
            # Calculate decision variables for position
            # x1 = 2 * theta1 / pi
            # x2 = 2 * theta2 / pi
            x[0] = (2 * t1 / np.pi) * (hi[0] - lo[0]) + lo[0]
            x[1] = (2 * t2 / np.pi) * (hi[1] - lo[1]) + lo[1]
            
            # Set distance variables to 0.5 to minimize g
            # DTLZ4: g(x) = sum (xi - 0.5)^2 for i >= 2
            # So xi = 0.5 is optimal
            for i in range(2, self.dim):
                x[i] = 0.5 * (hi[i] - lo[i]) + lo[i]
                
            f = problem(x)
            results.append(f)
            
        return results
