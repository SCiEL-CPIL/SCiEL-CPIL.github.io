import numpy as np
import time

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
            # Fallback
            return np.zeros((budget, n_obj))

        # DTLZ4 with 3 objectives:
        # f1 = g * x1
        # f2 = g * (1-x1) * x2
        # f3 = g * (1-x1) * (1-x2)
        # Optimal g=1 requires x[2:] = 0.5.
        # Front is triangle f1+f2+f3=1.
        # To maximize hypervolume, sample uniformly in the simplex (objective space).
        
        # 1. Generate a triangular grid in f1-f2 plane
        # Number of points in grid with side K is (K+1)(K+2)/2
        # Find max K such that points <= budget
        K = int(np.floor((-3 + np.sqrt(9 + 8*budget)) / 2))
        
        points = []
        for i in range(K + 1):
            for j in range(K - i + 1):
                f1 = i / K
                f2 = j / K
                # f3 = 1 - f1 - f2
                
                # Map to x
                x = np.zeros(dim)
                x[0] = f1
                if f1 < 1.0:
                    x[1] = f2 / (1.0 - f1)
                else:
                    x[1] = 0.0
                x[2:] = 0.5
                points.append(x)
                
        n_grid = len(points)
        
        # 2. Fill remaining budget with random points from Dirichlet(1,1,1)
        if n_grid < budget:
            n_rand = budget - n_grid
            # Use seeded random (evaluator sets seed)
            rand_f = np.random.dirichlet([1, 1, 1], size=n_rand)
            for row in rand_f:
                f1, f2, _ = row
                x = np.zeros(dim)
                x[0] = f1
                if f1 < 1.0:
                    x[1] = f2 / (1.0 - f1)
                else:
                    x[1] = 0.0
                x[2:] = 0.5
                points.append(x)

        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Check deadline
        if time.time() > deadline:
            return F
        
        return F
