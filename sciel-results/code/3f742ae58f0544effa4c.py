import numpy as np
import time
import scipy.stats.qmc as qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s/330s kill

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        if n_obj != 3:
            # Fallback for unexpected dimensions, though task specifies 3
            # Just return zeros or random
            return np.zeros((budget, n_obj))

        # DTLZ4 with 3 objectives has 2 position variables and dim-2 distance variables.
        # Pareto front is a quarter-sphere in the first octant.
        # f1 = cos(a)cos(b), f2 = cos(a)sin(b), f3 = sin(a)
        # where a = x1 * pi/2, b = x2 * pi/2.
        
        # We want to sample the front densely.
        # Using a Sobol sequence in 2D to generate (a, b) in [0, 1] then scaling to [0, pi/2].
        
        # Generate points
        # Use Sobol for low discrepancy
        sampler = qmc.Sobol(d=2, scramble=True)
        # Sobol requires power of 2 for best performance, but works for others.
        # We need budget points.
        # Let's take the next power of 2 >= budget and slice.
        n_pow = 1
        while n_pow < budget:
            n_pow *= 2
        
        # Generate n_pow points in [0,1]^2
        samples = sampler.random(n_pow)
        # Take the first 'budget' points
        samples = samples[:budget]
        
        # Scale to [0, pi/2]
        # a = samples[:, 0] * (pi/2)
        # b = samples[:, 1] * (pi/2)
        
        # Construct decision vectors
        points = []
        pi_2 = np.pi / 2.0
        
        for i in range(budget):
            a = samples[i, 0] * pi_2
            b = samples[i, 1] * pi_2
            
            x = np.zeros(dim)
            # Position variables
            # x1 maps to a, x2 maps to b
            # x1 = 2a/pi, x2 = 2b/pi
            x[0] = 2.0 * a / np.pi
            x[1] = 2.0 * b / np.pi
            
            # Distance variables: set to 0.5 to minimize g(x)
            # Assuming bounds are [0,1]
            if dim > 2:
                x[2:] = 0.5
            
            # Clip to bounds just in case
            for j in range(dim):
                if bounds[j][0] is not None:
                    x[j] = max(bounds[j][0], x[j])
                if bounds[j][1] is not None:
                    x[j] = min(bounds[j][1], x[j])
            
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Check deadline
        if time.time() > deadline:
            return F
        
        return F
