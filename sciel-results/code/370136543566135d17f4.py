"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Island Differential Evolution optimizer.
    
    Uses 5 independent DE populations (islands) that compete and migrate.
    Each island uses a 'Best/1' mutation strategy biased towards the global best.
    This balances global exploration (via islands) and local exploitation (via DE).
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

        # ---- Multi-Island DE Setup ---------------------------------------
        n_islands = 5
        pop_size = 10
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize islands with diverse starting points
        # To ensure good coverage, we mix random points and points near a few deterministic anchors
        anchors = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        ]
        
        islands = []
        for i in range(n_islands):
            population = []
            for j in range(pop_size):
                # Mix of random and anchor-based initialization
                if j == 0:
                    # One member starts near the anchor
                    x = anchors[i] + rng.standard_normal(dim) * 0.5
                else:
                    # Others start randomly but clipped to box
                    x = rng.uniform(-5.0, 5.0, dim)
                x = np.clip(x, -5.0, 5.0)
                f = func(x)
                record(x, f)
                if n_calls >= budget or time.time() > deadline:
                    return x_best, f_best
                population.append((x, f))
            
            # Sort by fitness
            population.sort(key=lambda item: item[1])
            islands.append({
                "pop": population,
                "stall": 0
            })

        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        # Main search loop
        # We cycle through islands. Each island step consumes pop_size evaluations (roughly)
        # But we process one individual at a time to respect the budget strictly.
        # Actually, standard DE evaluates pop_size new points per generation.
        # To keep it simple and budget-safe, we will process islands in a round-robin fashion,
        # generating one new candidate per island per "tick".
        
        island_indices = list(range(n_islands))
        rng.shuffle(island_indices)
        
        # Track which individual in the population we are replacing for each island
        # In standard DE, we replace the worst individual if the new one is better.
        # Here, we will just generate a mutant for the current "best" or a random individual?
        # Let's stick to standard DE: For each generation, for each individual i, create mutant.
        # Since we have a strict call budget, we will iterate through islands and generate mutants.
        
        # To maximize AOCC, early progress is key. 
        # DE is population-based. 
        # Let's do a "streaming" DE where we keep a population per island.
        
        while n_calls < budget and time.time() < deadline:
            # Pick an island (round robin or random)
            # Random selection of island helps if one is stuck
            island_idx = rng.integers(n_islands)
            island = islands[island_idx]
            pop = island["pop"]
            
            # Check if island is stuck
            # If the best individual in this island hasn't improved in a while, restart it
            # For simplicity, let's just run DE.
            
            # Pick a target individual from the population (random or worst?)
            # Standard DE: iterate through all individuals.
            # Here, we pick a random individual index to target.
            target_idx = rng.integers(pop_size)
            target_x, target_f = pop[target_idx]
            
            # Find global best from all islands (for 'Best/1' or 'Current-to-Best')
            # We already have x_best, f_best globally.
            
            # Mutation: Best/1
            # v = x_best + F * (x_r1 - x_r2)
            # Select two random distinct individuals from the SAME island to maintain local diversity
            # But using x_best (global) helps convergence.
            
            r1, r2 = rng.choice(pop_size, size=2, replace=False)
            x_r1 = pop[r1][0]
            x_r2 = pop[r2][0]
            
            # Use global best for the "Best" part to accelerate convergence
            # v = x_best + F * (x_r1 - x_r2)
            v = x_best + F * (x_r1 - x_r2)
            
            # Crossover
            y = np.empty(dim)
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    y[j] = v[j]
                else:
                    y[j] = target_x[j]
            
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            record(y, fy)
            
            if n_calls >= budget or time.time() > deadline:
                break
            
            # Selection
            if fy < target_f:
                # Replace target
                pop[target_idx] = (y, fy)
                # Resort population to keep best at top? 
                # For efficiency, we can just keep it unsorted and find best when needed,
                # but 'Best/1' uses global best, so local sorting is less critical for mutation,
                # but useful for identifying local best if we switch strategy.
                # Let's keep it sorted for simplicity, pop_size is small (10).
                pop.sort(key=lambda item: item[1])
                
                # Check if this island improved its local best
                # If the new best is better than the previous best of the island
                # (We don't track previous island best separately, but pop[0] is the best)
                # If pop[0][1] < old_best, it improved.
                # We don't have old_best stored, but we can infer if the global best changed?
                # Simpler: just let DE run.
            else:
                # No improvement for this specific target
                # Increment stall counter for the island?
                island["stall"] += 1
                # If stalled too much, restart the island around global best
                if island["stall"] > 50:
                    # Restart island population around global best
                    new_pop = []
                    for j in range(pop_size):
                        x_new = x_best + rng.standard_normal(dim) * 1.0
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = func(x_new)
                        record(x_new, f_new)
                        if n_calls >= budget or time.time() > deadline:
                            return x_best, f_best
                        new_pop.append((x_new, f_new))
                    new_pop.sort(key=lambda item: item[1])
                    island["pop"] = new_pop
                    island["stall"] = 0
                    break # Break out of current island loop iteration to handle budget
            # If improved, reset stall? 
            # Actually, stall should be per-island improvement.
            # If fy < target_f, it's a local improvement.
            # Let's say if the island's best (pop[0][1]) improves, reset stall.
            # We don't easily know if pop[0] changed without storing prev_best.
            # Let's just reset stall if any improvement happens in the island.
            if fy < target_f:
                island["stall"] = 0

        return x_best, f_best
# EVOLVE-BLOCK-END
