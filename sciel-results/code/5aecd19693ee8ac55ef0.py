"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol initialization + DE-based multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use Sobol sequence for initial population
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True)
        pop_size = min(64, budget // 3)
        sobol_samples = sampler.random(pop_size)
        X = lo + (hi - lo) * sobol_samples
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: check non-dominated mask
        def non_dominated_mask(F):
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Helper: compute hypervolume contribution (simplified for 3 objectives)
        # We'll use a simple indicator: sum of (ref - f) for each point, but only for non-dominated
        # Actually, for selection we can use a simpler approach: keep diverse non-dominated points

        # Main loop: DE-style mutation
        CR = 0.8
        F_mut = 0.8

        while used < budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_mask = non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) < 3:
                # If too few non-dominated, use all points
                parent_pool = np.arange(len(F))
            else:
                parent_pool = nd_indices

            n_parents = len(parent_pool)

            # DE mutation: pick 3 distinct parents
            for _ in range(min(1, budget - used)):  # one child per iteration
                if time.time() > deadline:
                    break

                idx1, idx2, idx3 = rng.choice(n_parents, size=3, replace=False)
                v = F[parent_pool[idx1]] + F_mut * (F[parent_pool[idx2]] - F[parent_pool[idx3]])
                # This is in objective space, not decision space. Let's do it in decision space.

                # Actually, DE should be in decision space
                x1 = X[parent_pool[idx1]]
                x2 = X[parent_pool[idx2]]
                x3 = X[parent_pool[idx3]]
                v = x1 + F_mut * (x2 - x3)

                # Binomial crossover
                trial = X[parent_pool[idx1]].copy()
                crossover_mask = rng.random(dim) < CR
                if not np.any(crossover_mask):
                    crossover_mask[rng.integers(dim)] = True
                trial[crossover_mask] = v[crossover_mask]
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1

                # Replace worst point in terms of some fitness indicator
                # Use a simple crowding-like measure: replace the point that is dominated
                # or has the worst "hypervolume-like" score
                # Simple approach: replace the point with the worst sum of normalized objectives
                # Or better: replace a dominated point if exists, else replace worst non-dominated

                # Check if trial is dominated by any existing point
                dominated_by_existing = False
                for i in range(len(F)):
                    if np.all(F[i] <= ft) and np.any(F[i] < ft):
                        dominated_by_existing = True
                        break

                if dominated_by_existing:
                    # Still evaluate, but don't replace unless it's better than worst
                    # Find the worst point to replace
                    # Use a simple metric: for each point, compute how much it contributes to HV
                    # Simplified: replace the point that is most dominated
                    domination_counts = np.zeros(len(F))
                    for i in range(len(F)):
                        for j in range(len(F)):
                            if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                domination_counts[i] += 1
                    worst_idx = np.argmax(domination_counts)
                    if domination_counts[worst_idx] > 0:
                        X[worst_idx] = trial
                        F[worst_idx] = ft
                else:
                    # Trial is non-dominated, replace the worst non-dominated point
                    # or the most dominated point
                    # Find non-dominated points
                    nd_mask_existing = non_dominated_mask(F)
                    if np.any(~nd_mask_existing):
                        # Replace a dominated point
                        dominated_indices = np.where(~nd_mask_existing)[0]
                        # Replace the most dominated one
                        domination_counts = np.zeros(len(F))
                        for i in dominated_indices:
                            for j in range(len(F)):
                                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                    domination_counts[i] += 1
                        worst_idx = dominated_indices[np.argmax(domination_counts[dominated_indices])]
                        X[worst_idx] = trial
                        F[worst_idx] = ft
                    else:
                        # All non-dominated, replace the one with worst "spread"
                        # Use a simple metric: replace the point closest to another point
                        # Or just replace a random non-dominated point
                        nd_indices_existing = np.where(nd_mask_existing)[0]
                        worst_idx = nd_indices_existing[rng.integers(len(nd_indices_existing))]
                        X[worst_idx] = trial
                        F[worst_idx] = ft

        return X

# EVOLVE-BLOCK-END
