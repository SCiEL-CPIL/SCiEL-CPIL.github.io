import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 20
        if self.budget < N:
            N = self.budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Parameters
        F_mut = 0.8
        CR = 0.5
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a random individual i
            i = np.random.randint(N)
            
            # Select three distinct random indices r1, r2, r3 != i
            # Simple sampling
            r1, r2, r3 = np.random.choice([j for j in range(N) if j != i], size=3, replace=False)
            
            # Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Crossover
            j_rand = np.random.randint(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Selection
            # Check if trial dominates i
            if np.all(f_trial <= F[i]) and np.any(f_trial < F[i]):
                X[i] = trial
                F[i] = f_trial
            elif not (np.all(F[i] <= f_trial) and np.any(F[i] < f_trial)):
                # If they are non-dominated, we might want to keep the one with better crowding or just random
                # For simplicity, if non-dominated, we can replace with probability 0.5 or use a simple archive
                # Here, let's just keep the current one to maintain stability, or replace if random
                # A better strategy: keep both in a larger archive, but we are limited to N.
                # Let's replace with 50% chance to allow exploration
                if np.random.random() < 0.5:
                    X[i] = trial
                    F[i] = f_trial

        return X
