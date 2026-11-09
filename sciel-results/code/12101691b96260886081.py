"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 325.0  # safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = min(40, max(10, budget // 5))
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            used += 1
            return np.asarray(f, dtype=np.float64)

        def non_dominated_sort(F):
            """Return list of fronts (each a list of indices), sorted by dominance level."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # nd[i] = number of points that dominate i
            fronts = []
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1
            current = [i for i in range(n) if nd[i] == 0]
            while current:
                fronts.append(current[:])
                next_front = []
                for i in current:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices in F."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            cd = np.zeros(len(indices))
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([eval_point(x) for x in X])

        # Main evolutionary loop
        while used < budget and time_ok():
            # Non-dominated sorting
            fronts = non_dominated_sort(F)

            # Build next generation
            next_X = []
            next_F = []
            n_children = min(pop_size, budget - used)

            # Tournament selection with crowding distance
            def tournament_select():
                # Select from first few fronts
                candidates = []
                for front in fronts[:2]:
                    candidates.extend(front)
                if not candidates:
                    candidates = list(range(len(F)))
                # Random pair
                i = rng.integers(len(candidates))
                j = rng.integers(len(candidates))
                # Compute crowding for comparison
                cd_vals = crowding_distance(F, candidates)
                # Map back
                cd_i = cd_vals[candidates.index(candidates[i])]
                cd_j = cd_vals[candidates.index(candidates[j])]
                if cd_i > cd_j:
                    return candidates[i]
                elif cd_j > cd_i:
                    return candidates[j]
                else:
                    return candidates[i]

            for _ in range(n_children):
                if not time_ok() or used >= budget:
                    break
                p1 = tournament_select()
                p2 = tournament_select()
                # SBX-like crossover (simple blend)
                beta = rng.random()
                c1 = beta * X[p1] + (1 - beta) * X[p2]
                c2 = (1 - beta) * X[p1] + beta * X[p2]
                # Mutation
                sigma = 0.1 * (hi - lo)
                c1 = c1 + sigma * rng.standard_normal(dim)
                c2 = c2 + sigma * rng.standard_normal(dim)
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)

                fc1 = eval_point(c1)
                next_X.append(c1)
                next_F.append(fc1)
                if used < budget and time_ok():
                    fc2 = eval_point(c2)
                    next_X.append(c2)
                    next_F.append(fc2)

            if not next_X:
                break

            # Combine parent and child, select
            all_X = np.vstack([X, np.array(next_X)])
            all_F = np.vstack([F, np.array(next_F)])
            fronts = non_dominated_sort(all_F)

            # Select pop_size from fronts
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    # Use crowding distance to pick remaining
                    remaining = pop_size - len(new_indices)
                    cd_vals = crowding_distance(all_F, front)
                    order = np.argsort(-cd_vals)
                    new_indices.extend(front[order[:remaining]])
                    break

            new_indices = new_indices[:pop_size]
            X = all_X[new_indices]
            F = all_F[new_indices]

        # Final local polish: use remaining budget for local search around best points
        if time_ok() and used < budget:
            # Find non-dominated points
            fronts = non_dominated_sort(F)
            elite_indices = []
            for front in fronts[:1]:
                elite_indices.extend(front)
            if not elite_indices:
                elite_indices = [int(np.argmin(np.sum(F, axis=1)))]

            # Local refinement around elites
            remaining = budget - used
            n_elite = len(elite_indices)
            steps_per_elite = max(1, remaining // max(1, n_elite))

            for idx in elite_indices:
                if used >= budget or not time_ok():
                    break
                x = X[idx].copy()
                f = F[idx].copy()
                best_x = x.copy()
                best_f = f.copy()
                # Simple local search with decreasing step
                step = 0.05 * (hi - lo)
                for _ in range(steps_per_elite):
                    if used >= budget or not time_ok():
                        break
                    for d in range(dim):
                        for direction in [-1, 1]:
                            if used >= budget or not time_ok():
                                break
                            x_trial = best_x.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, lo, hi)
                            f_trial = eval_point(x_trial)
                            # Check if it dominates or is non-dominated with current best
                            if np.all(f_trial <= best_f) and np.any(f_trial < best_f):
                                best_x = x_trial.copy()
                                best_f = f_trial.copy()
                            elif np.all(best_f <= f_trial) and np.any(best_f < f_trial):
                                pass  # keep best
                            else:
                                # Non-dominated: replace if sum is lower
                                if np.sum(f_trial) < np.sum(best_f):
                                    best_x = x_trial.copy()
                                    best_f = f_trial.copy()
                    step *= 0.7

                # Update population
                X[idx] = best_x
                F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
