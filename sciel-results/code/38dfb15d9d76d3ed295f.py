"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0
        max_calls = self.budget

        def time_check():
            return time.time() < deadline

        def de_run(seed, budget_for_run):
            nonlocal total_calls, best_x, best_f

            rng = np.random.default_rng(seed)
            n = self.dim
            # Population size
            npop = max(10, 10 * n) 
            # DE parameters
            F = 0.8
            CR = 0.9

            # Initialize population
            pop = rng.uniform(-5, 5, (npop, n))
            fitness = np.full(npop, np.inf)

            # Evaluate initial population
            for i in range(npop):
                if not time_check() or total_calls >= max_calls:
                    break
                f_val = func(pop[i])
                total_calls += 1
                fitness[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = pop[i].copy()

            if best_x is None:
                return

            # Main loop
            while total_calls < max_calls and time_check():
                new_pop = np.copy(pop)
                new_fitness = np.copy(fitness)
                
                # Track best in this generation
                gen_best_idx = np.argmin(fitness)
                gen_best_f = fitness[gen_best_idx]
                gen_best_x = pop[gen_best_idx].copy()
                
                # Stagnation counter
                stagnant = 0
                max_stagnant = 100 # Generations without improvement
                
                while total_calls < max_calls and time_check() and stagnant < max_stagnant:
                    improved = False
                    for i in range(npop):
                        if not time_check() or total_calls >= max_calls:
                            break
                        
                        # Select 3 distinct indices
                        r1, r2, r3 = rng.choice(npop, 3, replace=False)
                        
                        # Differential variation
                        v = pop[r1] + F * (pop[r2] - pop[r3])
                        
                        # Binomial crossover
                        trial = np.copy(pop[i])
                        j_rand = rng.integers(n)
                        for j in range(n):
                            if rng.random() < CR or j == j_rand:
                                trial[j] = v[j]
                        
                        # Clip to box
                        trial = np.clip(trial, -5, 5)
                        
                        # Evaluate trial
                        f_trial = func(trial)
                        total_calls += 1
                        
                        # Selection
                        if f_trial <= fitness[i]:
                            new_pop[i] = trial
                            new_fitness[i] = f_trial
                            improved = True
                            
                            if f_trial < gen_best_f:
                                gen_best_f = f_trial
                                gen_best_x = trial.copy()
                                
                            if f_trial < best_f:
                                best_f = f_val
                                best_x = trial.copy()

                    # Update population
                    pop = new_pop
                    fitness = new_fitness
                    
                    if not improved:
                        stagnant += 1
                    else:
                        stagnant = 0
                        
                    # If we improved, reset stagnant logic for the outer loop check
                    # But we need to check if we should break the outer loop (restart)
                    # For now, we just continue. The outer loop in __call__ handles restarts.

            return

        # Multi-start strategy
        rng_global = np.random.default_rng(42)
        n_starts = min(5, max(2, max_calls // 100))

        for i in range(n_starts):
            if not time_check() or total_calls >= max_calls:
                break

            # Budget for this run
            remaining = max_calls - total_calls
            if remaining <= 0:
                break
            
            # Give each run a reasonable budget
            # If this is the last run, give it all remaining
            if i == n_starts - 1:
                budget_for_run = remaining
            else:
                budget_for_run = remaining // (n_starts - i)
                budget_for_run = max(budget_for_run, 50) # Min 50 calls per run

            de_run(seed=1000 + i, budget_for_run=budget_for_run)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
