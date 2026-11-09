import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280.0
        rng = np.random.default_rng(123)
        n = self.dim
        budget = self.budget

        def evaluate(x):
            if self.calls >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x_clipped.copy()
            return val

        # 1. Initial exploration: diverse set of points
        initial_candidates = []
        initial_candidates.append(np.zeros(n))
        for _ in range(20):
            initial_candidates.append(rng.uniform(-4, 4, n))
        for _ in range(10):
            initial_candidates.append(rng.uniform(0.5, 1.5, n))
            initial_candidates.append(rng.uniform(-1.5, -0.5, n))

        for x in initial_candidates:
            if self.calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        if self.best_x is None:
            self.best_x = np.zeros(n)
            evaluate(self.best_x)
            return self.best_x, self.best_f

        # 2. Differential Evolution with interleaved local polishing
        # Population size
        pop_size = max(10, min(50, budget // 10))
        # Initialize population around current best and random points
        population = [self.best_x.copy()]
        for _ in range(pop_size - 1):
            if rng.random() < 0.5:
                pop_member = self.best_x + rng.normal(0, 1.0, n)
            else:
                pop_member = rng.uniform(-4, 4, n)
            pop_member = np.clip(pop_member, -5, 5)
            population.append(pop_member)
        
        # Evaluate initial population
        for i in range(pop_size):
            if self.calls >= budget or time.time() > deadline:
                break
            f_val = evaluate(population[i])
            if f_val is None:
                break
        
        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.5  # Crossover rate
        no_improve = 0
        max_no_improve = 30
        
        # Main DE loop with interleaved local polishing
        while self.calls < budget and time.time() < deadline:
            # Sort population by fitness (keep track of best)
            # For simplicity, just track global best
            
            # Generate offspring for each member
            for i in range(pop_size):
                if self.calls >= budget or time.time() > deadline:
                    break
                
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Differential evolution crossover
                trial = population[r1] + F * (population[r2] - population[r3])
                
                # Binary crossover
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = population[r1][j] + F * (population[r2][j] - population[r3][j])
                
                trial = np.clip(trial, -5, 5)
                
                f_trial = evaluate(trial)
                if f_trial is None:
                    break
                
                # Compare with parent (we need parent's fitness, but we don't store it)
                # For simplicity, replace if trial is better than current global best or random replacement
                # Better: keep a simple strategy - replace if better than worst in population
                # For now, just replace if it improves global best
                if f_trial < self.best_f:
                    population[i] = trial
                    no_improve = 0
                else:
                    no_improve += 1
            
            # Interleaved local polishing every few iterations
            if no_improve > 5 and self.calls < budget - 20 and time.time() < deadline:
                # Perform local search from current best
                local_calls = [0]
                local_best_f = [self.best_f]
                
                def nm_wrapper(z):
                    if self.calls >= budget or time.time() > deadline or local_calls[0] > 10:
                        return local_best_f[0]
                    z_clipped = np.clip(z, -5, 5)
                    val = func(z_clipped)
                    self.calls += 1
                    local_calls[0] += 1
                    if val < local_best_f[0]:
                        local_best_f[0] = val
                        self.best_f = val
                        self.best_x = z_clipped.copy()
                    return val
                
                try:
                    scipy.optimize.minimize(
                        nm_wrapper,
                        self.best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 10, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                    )
                except Exception:
                    pass
                
                # Update population with improved best
                population[0] = self.best_x.copy()
                no_improve = 0
        
        # 3. Final polish with Nelder-Mead if budget remains
        remaining = budget - self.calls
        if remaining > 20 and time.time() < deadline:
            local_calls = [0]
            local_best_f = [self.best_f]

            def nm_wrapper(z):
                if self.calls >= budget or time.time() > deadline or local_calls[0] > remaining:
                    return local_best_f[0]
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                self.calls += 1
                local_calls[0] += 1
                if val < local_best_f[0]:
                    local_best_f[0] = val
                    self.best_f = val
                    self.best_x = z_clipped.copy()
                return val

            try:
                scipy.optimize.minimize(
                    nm_wrapper,
                    self.best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining // 5, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        return self.best_x, self.best_f
