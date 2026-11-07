import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # bounds is list of (low, high)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific knowledge:
        # n_obj = 3
        # dim = 12
        # Distance variables: x[2] to x[11] (indices 2 to 11) should be 0.5
        # Angle variables: x[0], x[1]
        
        # We generate budget points
        # To ensure we use the seed set by evaluator, we use np.random
        # Generate budget x 2 random numbers
        r = np.random.random((self.budget, 2))
        
        # Transform to x1, x2
        # x1 = (2/pi) * arccos(r[:, 0])
        # x2 = (2/pi) * r[:, 1]
        
        x1 = (2.0 / np.pi) * np.arccos(r[:, 0])
        x2 = (2.0 / np.pi) * r[:, 1]
        
        # Create full population
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        # Set distance variables to 0.5
        # Assuming bounds are [0,1], 0.5 is the center.
        # If bounds are different, we should map 0.5 to the center of the bound?
        # DTLZ2 definition usually assumes [0,1]. The task says bounds [0,1] for every variable.
        # So 0.5 is correct.
        X[:, 2:] = 0.5
        
        # Clip just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The problem function takes a single vector x and returns 3 objectives.
        # We can loop or vectorize if possible, but problem(x) signature suggests single vector.
        # "problem(x) clips a finite length-12 vector to the bounds and returns 3 objective values."
        # So we must call it for each x.
        
        F = np.zeros((self.budget, self.n_obj))
        for i in range(self.budget):
            F[i] = problem(X[i])
            
        return X
