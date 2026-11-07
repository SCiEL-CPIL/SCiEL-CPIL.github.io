"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Differential Evolution with adaptive parameters for BBOB f11 (Discus).
    
    Strategy:
    1. Use a moderate population size (lambda=12) for good coverage.
    2. Adaptive step size (F) and crossover rate (CR) based on success rate.
    3. Strict budget management to stay within 500 calls.
    4. Time guard to prevent timeout.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_f = float('inf')

        # DE Parameters
        pop_size = 12  # Lambda
        F_min, F_max = 0.3, 0.8
        CR_min, CR_max = 0.1, 0.9
        
        # Adaptive state
        success_count = 0
        total_trials = 0
        F = 0.5  # Initial scaling factor
        CR = 0.5 # Initial crossover rate

        def should_stop():
            nonlocal calls_made
            if calls_made >= budget:
                return True
            if time.time() > deadline:
                return True
            return False

        # Initialize population
        # We need pop_size points. Check if we have budget for them.
        if budget < pop_size:
            # Fallback to random search if budget is too small
            for _ in range(budget):
                if should_stop():
                    break
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        # Generate initial population
        pop = []
        for i in range(pop_size):
            x = rng.uniform(-5, 5, dim)
            pop.append(x)
        
        # Evaluate initial population
        pop_f = []
        for x in pop:
            fx = func(x)
            calls_made += 1
            pop_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort initial population
        order = np.argsort(pop_f)
        pop = [pop[i] for i in order]
        pop_f = [pop_f[i] for i in order]

        # Main DE Loop
        while not should_stop():
            # Check if we have enough budget for a full generation
            # Each generation requires pop_size evaluations
            if calls_made + pop_size > budget:
                # Try to use remaining budget for a few more random or local steps?
                # For simplicity and safety, we stop here to avoid partial generations
                # which complicate state management.
                break

            # Adapt F and CR based on recent success rate
            if total_trials > 0:
                success_rate = success_count / total_trials
                # If success rate is high, decrease F and CR to refine
                # If success rate is low, increase F and CR to explore
                if success_rate > 0.5:
                    F = max(F_min, F * 0.9)
                    CR = max(CR_min, CR * 0.95)
                elif success_rate < 0.2:
                    F = min(F_max, F * 1.1)
                    CR = min(CR_max, CR * 1.05)
                # Reset counters
                success_count = 0
                total_trials = 0

            new_pop = []
            new_pop_f = []
            
            for i in range(pop_size):
                if should_stop():
                    # Fill remaining with current best to maintain population size?
                    # Or just break and handle partial population?
                    # Breaking here means we might have fewer than pop_size points in new_pop.
                    # It's safer to break out of the main loop entirely if we can't finish a gen.
                    # But we are inside the inner loop. Let's just break and then check.
                    break
                
                # Select three distinct random individuals other than i
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                
                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover: u[j] = v[j] if rand < CR or j == j_rand, else pop[i][j]
                j_rand = rng.integers(0, dim)
                u = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Evaluate trial vector
                fu = func(u)
                calls_made += 1
                total_trials += 1
                
                if fu < pop_f[i]:
                    new_pop.append(u)
                    new_pop_f.append(fu)
                    success_count += 1
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                else:
                    new_pop.append(pop[i])
                    new_pop_f.append(pop_f[i])

            # If we broke out of the inner loop early due to stop condition
            if len(new_pop) < pop_size:
                # This shouldn't happen if we checked budget before, but just in case
                # We can't update population properly. Stop.
                break

            # Update population
            pop = new_pop
            pop_f = new_pop_f
            
            # Sort population for next iteration (DE doesn't strictly require sorting,
            # but it helps for adaptive strategies if we used rank-based selection)
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

        return best_x, best_f

# EVOLVE-BLOCK-END
