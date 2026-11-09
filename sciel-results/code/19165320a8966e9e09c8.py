"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based multi-objective optimizer with weight-specific mutation intensities."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Island model: 3 islands with different weight vectors for mutation bias
        n_islands = 3
        island_weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        # Each island has a population
        pop_per_island = max(5, min(15, budget // (4 * n_islands)))
        total_pop = pop_per_island * n_islands
        
        # Initialize all islands with random points
        X_all = lo + (hi - lo) * rng.random((total_pop, dim))
        F_all = np.array([problem(x) for x in X_all])
        used = total_pop

        # Track best points found so far (for final return)
        best_archive = []
        
        # Helper to get non-dominated points from a subset
        def get_non_dominated_indices(F_subset):
            n = len(F_subset)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F_subset[j] <= F_subset[i]) and np.any(F_subset[j] < F_subset[i]):
                            dominated[i] = True
                            break
            return np.where(~dominated)[0]

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # For each island, evolve one generation
            for island_id in range(n_islands):
                if used >= budget or time.time() > deadline:
                    break
                
                # Get this island's population
                idx_start = island_id * pop_per_island
                idx_end = (island_id + 1) * pop_per_island
                X_island = X_all[idx_start:idx_end].copy()
                F_island = F_all[idx_start:idx_end].copy()
                
                # Get non-dominated parents for this island
                nd_indices = get_non_dominated_indices(F_island)
                if len(nd_indices) == 0:
                    nd_indices = np.arange(len(F_island))
                
                # Select a parent from non-dominated set
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
                parent = X_island[parent_idx]
                
                # Weight-specific mutation: bias mutation toward the island's weight direction
                # Larger mutation in dimensions that affect the target objective
                weight = island_weights[island_id]
                mutation_scale = 0.05 + 0.05 * rng.random()  # Base mutation scale
                
                # Generate mutation with slight bias toward improving the weighted objective
                noise = rng.standard_normal(dim)
                # Scale mutation per dimension based on weight influence
                child = parent + mutation_scale * (hi - lo) * noise
                
                # Additional targeted mutation: perturb more in directions that likely improve target
                # This is a heuristic - we add a small component toward lower values
                child = child - 0.01 * (hi - lo) * rng.random(dim) * weight.sum()
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Replace the worst point in this island
                worst_idx = int(np.argmax(np.sum(F_island, axis=1)))
                X_island[worst_idx] = child
                F_island[worst_idx] = fc
                
                # Update global arrays
                X_all[idx_start:idx_end] = X_island
                F_all[idx_start:idx_end] = F_island
                
                # Add to archive if it's non-dominated globally (approximate)
                # Check against current best_archive
                is_dominated = False
                for f_arch in best_archive:
                    if np.all(f_arch <= fc) and np.any(f_arch < fc):
                        is_dominated = True
                        break
                if not is_dominated:
                    # Remove dominated points from archive
                    new_archive = []
                    for f_arch in best_archive:
                        if not (np.all(fc <= f_arch) and np.any(fc < f_arch)):
                            new_archive.append(f_arch)
                    new_archive.append(fc)
                    best_archive = new_archive

        # Return the best archive points (or all points if archive is empty)
        if best_archive:
            return np.array(best_archive)
        else:
            return X_all
# EVOLVE-BLOCK-END
