"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight evolutionary multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated(self, F):
        """Return indices of non-dominated points (lightweight version)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        # A point is non-dominated if no other point is better in all objectives
        # and at least as good in all
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        # Initial population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random_sample((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best known non-dominated points
        all_F = F.copy()

        # Reserve budget for local search
        local_search_budget = min(40, self.budget // 5)
        ea_budget = self.budget - local_search_budget

        # Adaptive mutation scale: start large, shrink
        base_scale = 0.3
        min_scale = 0.02

        # Main Evolutionary Loop
        while used < ea_budget:
            if time.time() > deadline:
                break

            # Get non-dominated front
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])

            # Select parent from non-dominated front with some randomness
            if len(nd_indices) >= 2:
                i = nd_indices[rng.randint(len(nd_indices))]
                j = nd_indices[rng.randint(len(nd_indices))]
                if np.sum(F[i]) <= np.sum(F[j]):
                    parent_idx = i
                else:
                    parent_idx = j
            else:
                parent_idx = nd_indices[0]

            # Adaptive mutation: scale decreases as we use more budget
            progress = used / ea_budget
            scale = base_scale * (1 - progress) + min_scale * progress

            # Gaussian mutation with adaptive scale
            mutation = scale * (hi - lo) * rng.standard_normal(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace the worst individual in the population
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Also maintain a global archive of non-dominated points
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

        # Local Polish Phase: High-precision refinement
        if used < self.budget and time.time() < deadline:
            # Get the current non-dominated points from the population
            nd_indices = self._fast_non_dominated(F)
            if len(nd_indices) > 0:
                # Sort non-dominated points by sum of objectives (center of front first)
                nd_sums = np.sum(F[nd_indices], axis=1)
                sorted_nd = nd_indices[np.argsort(nd_sums)]
                
                # Pick top candidates for local search
                n_polish = min(10, len(sorted_nd))
                polish_indices = sorted_nd[:n_polish]
                
                # Distribute remaining budget among these points
                remaining_budget = self.budget - used
                steps_per_point = max(1, remaining_budget // n_polish)
                
                for pi in polish_indices:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    current_x = X[pi].copy()
                    current_F = F[pi].copy()
                    
                    # Local search: random hill climbing with decreasing step size
                    base_step = 0.1 * (hi - lo)
                    for step in range(steps_per_point):
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Decrease step size
                        step_scale = base_step * (0.8 ** step)
                        perturbation = step_scale * rng.standard_normal(self.dim)
                        new_x = current_x + perturbation
                        new_x = np.clip(new_x, lo, hi)
                        
                        new_F = np.array(problem(new_x), dtype=np.float64)
                        used += 1
                        
                        # Accept if it dominates current best in this local search
                        if np.all(new_F <= current_F) and np.any(new_F < current_F):
                            current_x = new_x
                            current_F = new_F
                            # Update population and archive
                            X[pi] = current_x
                            F[pi] = current_F
                            all_F = np.vstack([all_F, current_F.reshape(1, -1)])
                        else:
                            # Even if it doesn't dominate, add to archive if it's potentially non-dominated?
                            # For simplicity, only add if it dominates. 
                            # But we can also just record it to the archive to ensure coverage.
                            # Let's add it to the archive regardless, the hypervolume calc will filter dominated points.
                            all_F = np.vstack([all_F, new_F.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END
