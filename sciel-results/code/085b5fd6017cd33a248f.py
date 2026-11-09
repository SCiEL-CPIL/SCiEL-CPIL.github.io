import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size and generations
        # Budget = 200, Dim = 4
        # Let's use N=20, G=9. Total evals = 20 + 9*20 = 200.
        N = 20
        if self.budget < N:
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # DE Parameters
        F_scale = 0.8  # Differential weight
        CR = 0.9        # Crossover rate (though we are using vectorized replacement, 
                        # standard DE uses binomial crossover per dimension.
                        # For simplicity and robustness in low dim, we can use 
                        # the entire vector v if it passes selection, or 
                        # perform binomial crossover.
                        # Let's use binomial crossover for proper DE behavior.
        
        def dominates(a, b):
            """Returns True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)
        
        def is_dominated_by_any(point, population_F, exclude_idx=None):
            """Check if point is dominated by any point in population_F."""
            for j in range(len(population_F)):
                if j == exclude_idx:
                    continue
                if dominates(population_F[j], point):
                    return True
            return False

        while used + N <= self.budget:
            for i in range(N):
                # Select distinct r1, r2, r3
                while True:
                    r1 = np.random.randint(N)
                    r2 = np.random.randint(N)
                    r3 = np.random.randint(N)
                    if r1 != r2 and r2 != r3 and r1 != r3 and r1 != i and r2 != i and r3 != i:
                        break
                
                # Generate trial vector v
                v = X[r1] + F_scale * (X[r2] - X[r3])
                
                # Binomial Crossover
                trial = X[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() <= CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = np.array(problem(trial))
                used += 1
                
                # Selection
                if dominates(f_trial, F[i]):
                    X[i] = trial
                    F[i] = f_trial
                elif dominates(F[i], f_trial):
                    pass # Keep original
                else:
                    # Non-dominated with respect to F[i]
                    # Check if dominated by any other point in population
                    if not is_dominated_by_any(f_trial, F, exclude_idx=i):
                        X[i] = trial
                        F[i] = f_trial
                    # Else keep original to preserve diversity of current front

        return X
