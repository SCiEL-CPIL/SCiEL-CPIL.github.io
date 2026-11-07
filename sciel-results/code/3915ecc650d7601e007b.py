import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Differential Evolution Parameters
        NP = 10
        F = 0.6
        CR = 0.9
        
        # 1. Initialization
        population = np.zeros((NP, dim))
        fitness = np.zeros(NP)
        
        for i in range(NP):
            x = rng.uniform(-4.0, 4.0, size=dim)
            population[i] = x
            fitness[i] = eval_func(x)
            
        idx_best = np.argmin(fitness)
        best_f = fitness[idx_best]
        best_x = population[idx_best].copy()

        # 2. Main Loop
        while calls < budget - 1 and time_ok():
            for i in range(NP):
                if calls >= budget or not time_ok():
                    break
                
                # Select 3 distinct random indices from [0, NP) excluding i
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # Mutation: DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Clip to search space
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover: Binomial
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                        
                # Evaluate and Select
                f_trial = eval_func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
