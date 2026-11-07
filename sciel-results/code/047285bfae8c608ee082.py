"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast evolutionary multi-objective optimizer for WFG7 with NSGA-II-style selection and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 40

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: compute dominance relationships
        def compute_dominance(F):
            """Returns dominated_mask (True if point i is dominated by some other point)"""
            N = F.shape[0]
            if N == 0:
                return np.zeros(0, dtype=bool)
            # F[i,j] <= F[k,j] for all j and F[i,j] < F[k,j] for some j => k dominates i
            le = F[:, np.newaxis, :] <= F[np.newaxis, :, :]  # (N, N, M): le[i,k,j] = F[k,j] <= F[i,j]
            lt = F[:, np.newaxis, :] < F[np.newaxis, :, :]
            all_le = np.all(le, axis=2)  # (N, N): all_le[i,k] = F[k] <= F[i]
            any_lt = np.any(lt, axis=2)  # (N, N): any_lt[i,k] = F[k] < F[i] in at least one obj
            # dom[i,k] = True means k dominates i
            dom = all_le & any_lt
            np.fill_diagonal(dom, False)
            dominated_mask = np.any(dom, axis=1)  # dominated_mask[i] = True if some k dominates i
            return dominated_mask

        def crowding_distance(F):
            """Compute crowding distance for each point in F."""
            N = F.shape[0]
            M = F.shape[1]
            if N <= 2:
                return np.full(N, np.inf)
            cd = np.zeros(N)
            for m in range(M):
                obj = F[:, m]
                sorted_idx = np.argsort(obj)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
                if obj_range > 1e-12:
                    for i in range(1, N - 1):
                        cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
            return cd

        def non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            N = F.shape[0]
            if N == 0:
                return []
            dominated_mask = compute_dominance(F)
            front0 = np.where(~dominated_mask)[0]
            fronts = [front0]
            remaining = set(range(N)) - set(front0.tolist())
            
            while remaining:
                # Find next non-dominated set from remaining
                rem_arr = np.array(sorted(remaining))
                F_rem = F[rem_arr]
                dom_rem = compute_dominance(F_rem)
                nd_in_rem = np.where(~dom_rem)[0]
                if len(nd_in_rem) == 0:
                    # Should not happen, but safety
                    nd_in_rem = np.array([0])
                front = rem_arr[nd_in_rem]
                fronts.append(front)
                remaining -= set(front.tolist())
            return fronts

        def tournament_select(F, X, size=2):
            """Tournament selection based on dominance and crowding distance."""
            N = len(F)
            # Quick non-dominated sort
            fronts = non_dominated_sort(F)
            # Assign ranks
            rank = np.zeros(N, dtype=int)
            for r, front in enumerate(fronts):
                rank[front] = r
            # Crowding distance within the best front
            cd = crowding_distance(F)
            
            winners = []
            for _ in range(size):
                candidates = rng.integers(0, N, size=size)
                best = candidates[0]
                for c in candidates[1:]:
                    if rank[c] < rank[best]:
                        best = c
                    elif rank[c] == rank[best] and cd[c] > cd[best]:
                        best = c
                winners.append(best)
            return winners

        # Main loop
        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # Select parents via tournament
            p1, p2 = tournament_select(F, X, size=2)
            parent1 = X[p1]
            parent2 = X[p2]

            # Crossover: uniform with probability 0.6, otherwise mutation only
            if rng.random() < 0.6:
                mask = rng.random(dim) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()

            # Adaptive mutation
            progress = used / self.budget
            # Larger steps early, smaller late
            base_step = 0.12 * (1.0 - 0.6 * progress)
            # Gaussian perturbation
            noise = rng.standard_normal(dim)
            child = child + base_step * span * noise

            # Occasionally do a larger exploratory jump
            if rng.random() < 0.05:
                child = lo + span * rng.random(dim)

            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Environmental selection: add child, then select best pop_size
            X_new = np.vstack([X, child[None, :]])
            F_new = np.vstack([F, fc[None, :]])

            # Non-dominated sort
            fronts = non_dominated_sort(F_new)
            
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front.tolist())
                else:
                    # Need to truncate this front using crowding distance
                    F_front = F_new[front]
                    cd_front = crowding_distance(F_front)
                    # Sort by crowding distance descending
                    order = np.argsort(cd_front)[::-1]
                    needed = pop_size - len(selected_idx)
                    selected_idx.extend(front[order[:needed]].tolist())
                    break

            X = X_new[selected_idx]
            F = F_new[selected_idx]

        # Final local polish: try to improve each point with small perturbations
        polish_budget = self.budget - used
        if polish_budget > 5 and time.time() < deadline - 5:
            # Sort by sum of objectives, polish the best few
            F_sums = np.sum(F, axis=1)
            top_k = min(5, pop_size)
            top_indices = np.argsort(F_sums)[:top_k]
            
            for idx in top_indices:
                if time.time() > deadline - 2:
                    break
                # Try small perturbations around this point
                best_x = X[idx]
                best_f = F[idx]
                best_sum = np.sum(best_f)
                
                for _ in range(3):
                    if time.time() > deadline - 2:
                        break
                    # Small step
                    step = 0.02 * span
                    trial = best_x + step * rng.standard_normal(dim)
                    trial = np.clip(trial, lo, hi)
                    tf = problem(trial)
                    used += 1
                    t_sum = np.sum(tf)
                    # Check if trial dominates best_f or is just better in sum
                    le_t = best_f <= tf
                    lt_t = best_f < tf
                    best_dominates_trial = np.all(le_t) and np.any(lt_t)
                    trial_dominates_best = np.all(tf <= best_f) and np.any(tf < best_f)
                    
                    if trial_dominates_best or (not best_dominates_trial and t_sum < best_sum):
                        best_x = trial
                        best_f = tf
                        best_sum = t_sum
                
                X[idx] = best_x
                F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
