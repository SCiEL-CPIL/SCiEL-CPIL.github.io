import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Local RNG seeded from global np.random state (which evaluator has seeded)
        rng = np.random

        used = 0

        def dominates(a, b):
            """Returns True if a dominates b (a <= b in all, a < b in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def dominates_any(f, archive_F):
            """Check if f is dominated by any point in archive_F (list of arrays)"""
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def remove_dominated(f, archive_X, archive_F):
            """Remove points dominated by f from archive. Returns new lists."""
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            return new_X, new_F

        def crowding_distance(F_arr):
            """Calculate crowding distance for a set of objective vectors F_arr (N, n_obj)."""
            n = F_arr.shape[0]
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_indices = np.argsort(F_arr[:, m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                f_range = F_arr[sorted_indices[-1], m] - F_arr[sorted_indices[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    cd[idx] += (F_arr[sorted_indices[i + 1], m] - F_arr[sorted_indices[i - 1], m]) / f_range
            return cd

        # Archive as lists of numpy arrays
        archive_X = []
        archive_F = []

        def evaluate_and_add(x):
            nonlocal used, archive_X, archive_F
            if used >= budget:
                return None
            if time.time() - start_time > deadline:
                return None

            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1

            if np.any(~np.isfinite(f)):
                return None

            # Check if dominated by existing archive
            if dominates_any(f, archive_F):
                return None

            # Remove points dominated by f
            archive_X, archive_F = remove_dominated(f, archive_X, archive_F)

            archive_X.append(x)
            archive_F.append(f)
            return x, f

        # --- Initialization: diverse starting points ---
        # Center
        x0 = (lo + hi) / 2.0
        evaluate_and_add(x0)

        # Vertices of the hypercube
        for mask in range(2 ** dim):
            if used >= budget:
                break
            x = np.array([hi[j] if (mask >> j) & 1 else lo[j] for j in range(dim)])
            evaluate_and_add(x)

        # Additional random points for diversity
        while used < 15:
            x = lo + (hi - lo) * rng.random(dim)
            evaluate_and_add(x)

        # --- Main search loop: NSGA-II style with DE mutation ---
        # We maintain a population for DE crossover, but the archive is the non-dominated set.
        # Strategy: 
        #   - Keep a "population" of recent good points (including archive) for mutation
        #   - Use DE/rand/1 mutation
        #   - Use crowding distance for selection when archive is full
        #   - In later stages, do local search around archive points

        pop_size = min(30, max(10, budget // 10))
        
        # Population for DE: start with archive points + random
        pop_X = list(archive_X)
        pop_F = list(archive_F)

        # Fill population with random points
        while len(pop_X) < pop_size and used < budget:
            x = lo + (hi - lo) * rng.random(dim)
            res = evaluate_and_add(x)
            if res is not None:
                pop_X.append(x)
                pop_F.append(res[1])

        # If archive is empty (shouldn't happen), just do random search
        if len(archive_X) == 0:
            while used < budget:
                if time.time() - start_time > deadline:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                evaluate_and_add(x)
            return None

        # DE parameters
        F_factor = 0.7
        CR = 0.8

        # Track best hypervolume-ish indicator for stagnation detection
        # We'll use the number of archive points and their spread as a proxy

        last_archive_size = len(archive_X)
        stagnation_count = 0

        while used < budget:
            if time.time() - start_time > deadline:
                break

            progress = used / budget

            # Switch to local polishing mode in the last 25% of budget
            if progress > 0.75:
                # Local search: perturb archive points with small steps
                if len(archive_X) == 0:
                    x = lo + (hi - lo) * rng.random(dim)
                    evaluate_and_add(x)
                    continue

                # Select point with highest crowding distance
                if len(archive_X) > 1:
                    F_arr = np.array(archive_F)
                    cd = crowding_distance(F_arr)
                    idx = np.argmax(cd)
                else:
                    idx = 0

                base_x = archive_X[idx]

                # Decreasing step size
                remaining_frac = (budget - used) / (0.25 * budget)
                step_size = (0.02 + 0.05 * remaining_frac) * (hi - lo)
                x = base_x + rng.normal(0, 1, dim) * step_size
                evaluate_and_add(x)

                # Also try a few random points occasionally
                if rng.random() < 0.2:
                    x = lo + (hi - lo) * rng.random(dim)
                    evaluate_and_add(x)
            else:
                # DE-based exploration
                if len(pop_X) < 3:
                    x = lo + (hi - lo) * rng.random(dim)
                    evaluate_and_add(x)
                    if len(pop_X) < pop_size:
                        # Re-evaluate to get f
                        x_clip = np.clip(x, lo, hi)
                        f = np.array(problem(x_clip), dtype=float)
                        used += 1
                        if np.all(np.isfinite(f)):
                            pop_X.append(x_clip)
                            pop_F.append(f)
                    continue

                # DE/rand/1 mutation
                n_pop = len(pop_X)
                i1, i2, i3 = rng.choice(n_pop, 3, replace=False)
                x1, x2, x3 = pop_X[i1], pop_X[i2], pop_X[i3]

                # Adaptive F: decrease as we progress
                F_current = F_factor * (1.0 - 0.4 * progress)

                mutant = x1 + F_current * (x2 - x3)

                # Binomial crossover with x1 as target
                j_rand = rng.randint(dim)
                cross_mask = rng.random(dim) < CR
                cross_mask[j_rand] = True

                trial = np.where(cross_mask, mutant, x1)

                res = evaluate_and_add(trial)

                # Update population: replace worst point (by crowding distance)
                if len(pop_X) >= pop_size:
                    # Compute crowding distance for population
                    pop_F_arr = np.array(pop_F)
                    cd_pop = crowding_distance(pop_F_arr)
                    worst_idx = np.argmin(cd_pop)
                    if res is not None:
                        pop_X[worst_idx] = trial
                        pop_F[worst_idx] = res[1]
                elif res is not None:
                    pop_X.append(trial)
                    pop_F.append(res[1])

                # Stagnation check
                if len(archive_X) > last_archive_size:
                    last_archive_size = len(archive_X)
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # If stagnating, inject random diversity
                if stagnation_count > 20:
                    for _ in range(min(5, budget - used)):
                        x = lo + (hi - lo) * rng.random(dim)
                        res = evaluate_and_add(x)
                        if res is not None and len(pop_X) < pop_size:
                            pop_X.append(res[0])
                            pop_F.append(res[1])
                    stagnation_count = 0

                # Prune archive if it grows too large
                if len(archive_X) > 50:
                    F_arr = np.array(archive_F)
                    cd = crowding_distance(F_arr)
                    sort_idx = np.argsort(-cd)
                    keep_idx = sort_idx[:40]
                    archive_X = [archive_X[i] for i in keep_idx]
                    archive_F = [archive_F[i] for i in keep_idx]

        return None
