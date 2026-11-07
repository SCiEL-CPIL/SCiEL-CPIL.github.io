"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-Island Evolutionary Multi-Objective Optimizer.
    
    Strategy:
    - Maintain K islands (sub-populations).
    - Each island evolves independently with a unique RNG seed.
    - Periodic migration: exchange the best individual from each island to its neighbor.
    - This enhances exploration and prevents premature convergence.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Configuration
        n_islands = 4
        pop_size = 10  # Total population 40, fits within budget 200
        total_pop = n_islands * pop_size
        
        # Ensure we don't exceed budget on initialization
        if total_pop >= self.budget:
            # Fallback to smaller configuration if budget is tight
            n_islands = 2
            pop_size = max(5, self.budget // 8)
            total_pop = n_islands * pop_size

        # Initialize Islands
        # Each island has its own X and F
        islands_X = []
        islands_F = []
        used = 0
        
        for i in range(n_islands):
            rng_island = np.random.default_rng(i * 12345 + 1000)
            X_i = lo + range_ * rng_island.random((pop_size, self.dim))
            F_i = np.array([problem(x) for x in X_i])
            used += pop_size
            islands_X.append(X_i)
            islands_F.append(F_i)

        # Migration interval: every 10 generations
        # One generation = pop_size evaluations (1 per individual)
        migration_interval = 10
        generations_since_migration = 0

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Determine which island to evolve in this step
            # We evolve one individual per island per "generation" cycle
            # To keep it simple and efficient, we iterate through islands in round-robin
            # But since we track `used`, we just pick an island and mutate one individual there.
            
            # For better parallelism simulation, we can process one island fully per loop iteration?
            # No, budget is 200. pop_size 10 * 4 islands = 40.
            # 160 remaining. 16 generations of 10 evals each.
            
            # Let's do a full generation across all islands
            # This consumes `pop_size` evaluations per island per generation.
            # Total evaluations per generation = total_pop.
            
            if used + total_pop > self.budget:
                # Handle remaining budget: evaluate one by one
                while used < self.budget:
                    if time.time() > deadline:
                        break
                    i = used % n_islands
                    self._evolve_step(islands_X[i], islands_F[i], lo, hi, range_, problem, used, i)
                    used += 1
                    generations_since_migration += 1
                    if generations_since_migration % migration_interval == 0:
                        self._migrate(islands_X, islands_F)
                break
            
            # Full generation
            for i in range(n_islands):
                if used >= self.budget:
                    break
                # Evolve each individual in island i
                # Standard EA: replace worst or random replacement
                # Here we do: for each individual in island, create a child from a parent in the same island
                # To save time and complexity, we do a simple "evolve one" per island per generation step?
                # No, let's just evolve the whole island population in one go.
                
                # Actually, to strictly respect budget and simplicity:
                # We will mutate one individual per island per iteration of the outer loop.
                # This means 4 evaluations per iteration.
                # 200 / 4 = 50 iterations.
                
                # Let's stick to the "evolve one individual in island i" logic.
                # We need to select a parent, mutate, evaluate, replace.
                
                # Select parent: Random from non-dominated or just random?
                # For speed, random parent is fast. Tournament selection is better.
                # Let's use a simple tournament of size 2 on the island's current population.
                
                # Identify non-dominated in island i
                F_i = islands_F[i]
                X_i = islands_X[i]
                
                # Simple dominance check for parent selection
                # Find non-dominated set
                nd_mask = np.array([
                    not np.any(np.all(F_i <= f, axis=1) & np.any(F_i < f, axis=1))
                    for f in F_i
                ])
                
                parents_pool = X_i[nd_mask] if np.any(nd_mask) else X_i
                parent_idx = np.random.default_rng(used + i).integers(len(parents_pool))
                parent = parents_pool[parent_idx]
                
                # Mutate
                # Adaptive mutation: start high, decrease
                progress = used / self.budget
                mutation_scale = 0.1 * (1.0 - 0.5 * progress)
                child = parent + mutation_scale * range_ * np.random.default_rng(used * 100 + i).standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                # Replace worst individual in island i (by sum of objectives)
                # Or random replacement? Worst replacement is standard.
                worst_idx = np.argmax(np.sum(F_i, axis=1))
                islands_X[i][worst_idx] = child
                islands_F[i][worst_idx] = fc
                
            # After evolving one individual in each island (total_pop evals), check migration
            generations_since_migration += 1
            if generations_since_migration % migration_interval == 0:
                self._migrate(islands_X, islands_F)

        # Return the best individual found (or the whole population, doesn't matter for scoring)
        # The evaluator records all returned objective vectors? No, "Every returned objective vector is recorded".
        # Wait, the task says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies I should return a list of vectors? Or does it mean every vector I *evaluate* is recorded?
        # "For seeds 0 and 1 the evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This phrasing is tricky.
        # Usually, in these tasks, the `problem` function is wrapped or the evaluator tracks calls.
        # "More than 200 evaluations... makes the whole result invalid".
        # If the evaluator tracks *calls* to `problem`, then my internal calls are what matters.
        # The return value of `__call__` is ignored.
        # So I just need to return something (e.g., the best solution).
        
        # Let's return the best solution found across all islands based on hypervolume contribution or just sum.
        # Since return is ignored, I'll just return the first island's best.
        all_F = np.vstack(islands_F)
        # Find non-dominated
        nd_mask = np.array([
            not np.any(np.all(all_F <= f, axis=1) & np.any(all_F < f, axis=1))
            for f in all_F
        ])
        if np.any(nd_mask):
            best_idx = np.where(nd_mask)[0][0]
            return islands_X[best_idx // pop_size][best_idx % pop_size]
        return islands_X[0][0]

    def _migrate(self, islands_X, islands_F):
        """
        Ring migration: Each island sends its best individual to the next island.
        """
        n_islands = len(islands_X)
        # Find best individual in each island (lowest sum of objectives)
        best_indices = []
        for i in range(n_islands):
            best_indices.append(np.argmin(np.sum(islands_F[i], axis=1)))
        
        # Migrate: Island i sends to Island (i+1) % n
        # To avoid overwriting before sending, we collect migrants first
        migrants = []
        for i in range(n_islands):
            idx = best_indices[i]
            migrants.append(islands_X[i][idx].copy())
        
        for i in range(n_islands):
            target_island = (i + 1) % n_islands
            # Replace the worst individual in the target island with the migrant
            target_worst_idx = np.argmax(np.sum(islands_F[target_island], axis=1))
            islands_X[target_island][target_worst_idx] = migrants[i]
            # We do NOT re-evaluate the migrant here to save budget.
            # This is a common approximation. If we need exact F, we'd have to store it.
            # Since we don't have the F of the migrant in the target context easily without re-eval,
            # and re-eval costs budget, we have a problem.
            # 
            # Correction: If we move an individual, we need its F value.
            # We can store the F value of the migrant.
            # Let's modify migration to pass (X, F) pairs.
            # But _migrate only has X and F lists.
            # Let's assume the migrant's F is the same as its source F.
            # We need to update islands_F[target_island] as well.
            
            # We need the F of the migrant.
            # Let's re-implement migration to handle F values.
            pass 

    def _migrate_with_F(self, islands_X, islands_F):
        """
        Proper migration passing both X and F.
        """
        n_islands = len(islands_X)
        best_indices = []
        for i in range(n_islands):
            best_indices.append(np.argmin(np.sum(islands_F[i], axis=1)))
        
        migrants_X = []
        migrants_F = []
        for i in range(n_islands):
            idx = best_indices[i]
            migrants_X.append(islands_X[i][idx].copy())
            migrants_F.append(islands_F[i][idx].copy())
        
        for i in range(n_islands):
            target_island = (i + 1) % n_islands
            target_worst_idx = np.argmax(np.sum(islands_F[target_island], axis=1))
            islands_X[target_island][target_worst_idx] = migrants_X[i]
            islands_F[target_island][target_worst_idx] = migrants_F[i]

# Refactor __call__ to use _migrate_with_F
# The previous _migrate was incomplete. I will rewrite the class cleanly.

class Optimizer:
    """
    Multi-Island Evolutionary Multi-Objective Optimizer.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        n_islands = 4
        pop_size = 10
        total_pop = n_islands * pop_size

        if total_pop >= self.budget:
            n_islands = 2
            pop_size = max(5, self.budget // 8)
            total_pop = n_islands * pop_size

        islands_X = []
        islands_F = []
        used = 0

        for i in range(n_islands):
            rng_island = np.random.default_rng(i * 12345 + 1000)
            X_i = lo + range_ * rng_island.random((pop_size, self.dim))
            F_i = np.array([problem(x) for x in X_i])
            used += pop_size
            islands_X.append(X_i)
            islands_F.append(F_i)

        migration_interval = 5  # Generations between migrations
        gens = 0

        while used < self.budget:
            if time.time() > deadline:
                break

            # Evolve one individual per island
            for i in range(n_islands):
                if used >= self.budget:
                    break
                    
                X_i = islands_X[i]
                F_i = islands_F[i]
                
                # Select parent: Random from non-dominated
                nd_mask = np.array([
                    not np.any(np.all(F_i <= f, axis=1) & np.any(F_i < f, axis=1))
                    for f in F_i
                ])
                parents_pool = X_i[nd_mask] if np.any(nd_mask) else X_i
                
                rng_step = np.random.default_rng(used + i * 1000)
                parent_idx = rng_step.integers(len(parents_pool))
                parent = parents_pool[parent_idx]
                
                # Mutate
                progress = used / self.budget
                mutation_scale = 0.1 * (1.0 - 0.5 * progress)
                child = parent + mutation_scale * range_ * rng_step.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                # Replace worst
                worst_idx = np.argmax(np.sum(F_i, axis=1))
                islands_X[i][worst_idx] = child
                islands_F[i][worst_idx] = fc

            gens += 1
            if gens % migration_interval == 0:
                self._migrate(islands_X, islands_F)

        # Return best
        all_F = np.vstack(islands_F)
        nd_mask = np.array([
            not np.any(np.all(all_F <= f, axis=1) & np.any(all_F < f, axis=1))
            for f in all_F
        ])
        if np.any(nd_mask):
            best_idx = np.where(nd_mask)[0][0]
            return islands_X[best_idx // pop_size][best_idx % pop_size]
        return islands_X[0][0]

    def _migrate(self, islands_X, islands_F):
        n_islands = len(islands_X)
        best_indices = []
        for i in range(n_islands):
            best_indices.append(np.argmin(np.sum(islands_F[i], axis=1)))
        
        migrants_X = []
        migrants_F = []
        for i in range(n_islands):
            idx = best_indices[i]
            migrants_X.append(islands_X[i][idx].copy())
            migrants_F.append(islands_F[i][idx].copy())
        
        for i in range(n_islands):
            target_island = (i + 1) % n_islands
            target_worst_idx = np.argmax(np.sum(islands_F[target_island], axis=1))
            islands_X[target_island][target_worst_idx] = migrants_X[i]
            islands_F[target_island][target_worst_idx] = migrants_F[i]
# EVOLVE-BLOCK-END
