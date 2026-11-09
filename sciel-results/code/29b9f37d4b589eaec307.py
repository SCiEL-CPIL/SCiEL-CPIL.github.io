"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid DE + (1+1)-ES for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        
        best_x = None
        best_f = float('inf')
        
        # Strategy: 5 Restarts
        # Each restart: 
        #   1. Initialize small population (DE)
        #   2. Run DE for ~50% of restart budget
        #   3. Run (1+1)-ES on best DE point for remaining ~50%
        
        num_restarts = 5
        budget_per_restart = total_budget // num_restarts
        
        rng = np.random.default_rng(42)

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
                
            # Determine budget for this restart
            remaining_budget = total_budget - calls_made
            curr_budget = min(budget_per_restart, remaining_budget)
            if curr_budget < 10:
                break

            calls_at_start = calls_made

            # --- Phase 1: Differential Evolution (Exploration) ---
            pop_size = 6 
            F = 0.8
            CR = 0.9
            
            # Initialization
            if restart_idx == 0:
                # Start near 0 (local min) but with some spread
                pop = np.zeros((pop_size, n))
                for i in range(pop_size):
                    if i == 0:
                        pop[i] = 0.0
                    else:
                        pop[i] = rng.uniform(-2.0, 2.0, n)
            elif restart_idx == 1:
                pop = rng.uniform(-3.0, 3.0, (pop_size, n))
            elif restart_idx == 2:
                pop = rng.uniform(-5.0, 5.0, (pop_size, n))
            elif restart_idx == 3:
                pop = rng.uniform(-5.0, 5.0, (pop_size, n))
            else:
                if best_x is not None:
                    pop = np.tile(best_x, (pop_size, 1))
                    for i in range(pop_size):
                        pop[i] += rng.normal(0, 2.0, n)
                    pop = np.clip(pop, -5, 5)
                else:
                    pop = rng.uniform(-5, 5, (pop_size, n))

            # Evaluate initial population
            pop_f = np.array([func(x) for x in pop])
            calls_made += pop_size
            
            # Update global best
            min_idx = np.argmin(pop_f)
            if pop_f[min_idx] < best_f:
                best_f = pop_f[min_idx]
                best_x = pop[min_idx].copy()

            # DE Loop: Run for 50% of curr_budget
            de_target_calls = int(curr_budget * 0.5)
            
            while calls_made - calls_at_start < de_target_calls and calls_made < total_budget:
                if time.time() > deadline:
                    return best_x, best_f
                
                # Generate one generation (pop_size evaluations)
                # If we don't have enough budget for a full generation, break
                if calls_made + pop_size > total_budget:
                    break
                if time.time() > deadline:
                    break

                new_pop = np.copy(pop)
                new_f = np.copy(pop_f)
                
                for i in range(pop_size):
                    # Select 3 distinct indices
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    # Ensure they are not i
                    while r1 == i or r2 == i or r3 == i:
                        r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover
                    trial = np.copy(pop[i])
                    cross_point = rng.integers(0, n)
                    for j in range(n):
                        if rng.random() < CR or j == cross_point:
                            trial[j] = v[j]
                    
                    trial = np.clip(trial, -5, 5)
                    
                    trial_f = func(trial)
                    calls_made += 1
                    
                    if trial_f <= new_f[i]:
                        new_pop[i] = trial
                        new_f[i] = trial_f
                        
                pop = new_pop
                pop_f = new_f
                
                # Update global best
                min_idx = np.argmin(pop_f)
                if pop_f[min_idx] < best_f:
                    best_f = pop_f[min_idx]
                    best_x = pop[min_idx].copy()

            # --- Phase 2: (1+1)-ES (Exploitation) ---
            # Start from best in current population
            local_best_idx = np.argmin(pop_f)
            x = pop[local_best_idx].copy()
            fx = pop_f[local_best_idx]
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with 1/5 success rule
            sigma = 1.0
            window = 10
            successes = 0
            window_count = 0
            
            # Run until we've used the rest of the restart budget
            # Total budget for this restart is curr_budget.
            # We have used (calls_made - calls_at_start).
            # Stop when (calls_made - calls_at_start) >= curr_budget
            
            while calls_made - calls_at_start < curr_budget and calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                window_count += 1

                # Adapt sigma
                if window_count >= window:
                    rate = successes / window_count
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.15:
                        sigma *= 0.8
                    successes = 0
                    window_count = 0
                    sigma = np.clip(sigma, 1e-8, 10.0)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
