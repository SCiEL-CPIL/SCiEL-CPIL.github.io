"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Island Differential Evolution optimizer.
    
    Uses 5 independent DE populations (islands) that evolve in parallel.
    Periodic migration occurs where the best individual from one island
    replaces the worst in another. This balances global exploration (multiple
    islands in different regions) with local exploitation (DE within each island).
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        def check_deadline():
            return n_calls >= budget or time.time() > deadline

        # ---- Multi-Island DE Implementation ------------------------------
        n_islands = 5
        # Each island gets a portion of the budget
        # We will simulate sequential execution of islands, but maintain state for all.
        
        # DE Parameters
        # Population size per island: 10
        pop_size = 10
        # Total population = 5 * 10 = 50
        # Migration every 10 generations
        
        # Initialize islands
        islands = []
        for i in range(n_islands):
            # Start each island in a different region of the box
            # Use deterministic offsets to ensure coverage
            center = rng.uniform(-4.0, 4.0, dim)
            # Initial population around center
            pop = center + rng.standard_normal((pop_size, dim)) * 2.0
            pop = np.clip(pop, -5.0, 5.0)
            
            # Evaluate initial population
            fvals = np.zeros(pop_size)
            for j in range(pop_size):
                if check_deadline():
                    return x_best, f_best
                fvals[j] = func(pop[j])
                record(pop[j], fvals[j])
            
            islands.append({
                'pop': pop,
                'fvals': fvals,
                'gen': 0
            })

        # Main loop
        # We process islands in a round-robin fashion to keep it fair
        # Each "step" is one generation for one island (pop_size evaluations)
        
        # To make it efficient, we run one generation for island 0, then 1, etc.
        # But 500 calls / 10 per gen = 50 gens per island if sequential.
        # Actually, we have 500 calls total.
        # 5 islands * 10 pop = 50 initial calls.
        # Remaining 450 calls.
        # 450 / 10 = 45 generations per island if we cycle.
        
        while n_calls < budget and time.time() < deadline:
            for island_idx in range(n_islands):
                if check_deadline():
                    break
                
                island = islands[island_idx]
                pop = island['pop']
                fvals = island['fvals']
                
                # DE Mutation: Best/2/Current
                # Select best individual in this island
                best_idx = np.argmin(fvals)
                best_x = pop[best_idx]
                
                new_pop = np.copy(pop)
                new_fvals = np.copy(fvals)
                
                for i in range(pop_size):
                    if check_deadline():
                        break
                    
                    # Select 3 distinct random individuals
                    candidates = [x for x in range(pop_size) if x != i]
                    a, b, c = rng.choice(candidates, 3, replace=False)
                    
                    # Mutation: x_i + F * (x_a - x_b) + F * (x_best - x_c) ? 
                    # Standard DE/rand/1: x_i + F * (x_a - x_b)
                    # DE/best/1: x_best + F * (x_a - x_b)
                    # Let's use DE/best/2 for stronger exploitation or DE/rand/1 for exploration.
                    # Given multimodality, DE/rand/1 is safer, but DE/best/1 converges faster.
                    # Let's use a mix: 50% DE/rand/1, 50% DE/best/1
                    if rng.random() < 0.5:
                        mutant = pop[a] + 0.8 * (pop[b] - pop[c])
                    else:
                        mutant = best_x + 0.8 * (pop[a] - pop[b])
                    
                    mutant = np.clip(mutant, -5.0, 5.0)
                    
                    # Evaluate mutant
                    f_mutant = func(mutant)
                    record(mutant, f_mutant)
                    
                    # Selection
                    if f_mutant < fvals[i]:
                        new_pop[i] = mutant
                        new_fvals[i] = f_mutant
                        # Update island best if this is better
                        if f_mutant < fvals[best_idx]:
                            best_idx = i
                            best_x = mutant
                
                island['pop'] = new_pop
                island['fvals'] = new_fvals
                island['gen'] += 1
                
                # Migration
                # Every 5 generations, migrate best from this island to worst in next island
                if island['gen'] % 5 == 0:
                    next_island_idx = (island_idx + 1) % n_islands
                    next_island = islands[next_island_idx]
                    
                    # Find best in current island
                    curr_best_idx = np.argmin(island['fvals'])
                    curr_best_x = island['pop'][curr_best_idx]
                    curr_best_f = island['fvals'][curr_best_idx]
                    
                    # Find worst in next island
                    next_worst_idx = np.argmax(next_island['fvals'])
                    next_worst_f = next_island['fvals'][next_worst_idx]
                    
                    # If current best is better than next worst, swap
                    if curr_best_f < next_worst_f:
                        next_island['pop'][next_worst_idx] = curr_best_x
                        next_island['fvals'][next_worst_idx] = curr_best_f
                        # Note: We don't record this as a new call, it's just state transfer
                        # But wait, the evaluator counts calls to func.
                        # We are NOT calling func here. We are just copying the value.
                        # This is valid because we already evaluated this point in the current island.
                        # The value is known.

        return x_best, f_best
# EVOLVE-BLOCK-END
