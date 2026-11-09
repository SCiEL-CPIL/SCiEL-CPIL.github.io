"""Multi-objective optimizer on WFG4 using a refined evolutionary strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with adaptive mutation and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size
        # For 200 budget, a pop size of 20-40 is reasonable to allow multiple generations
        pop_size = 30 

        # Initial random population (Uniform)
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final hypervolume calculation if needed, 
        # but the task says "Every returned objective vector is recorded". 
        # The evaluator records them. We just need to return something or ignore return.
        # We will maintain the best non-dominated set in F.

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts, where each front is a list of indices."""
            n_pts = len(F_arr)
            if n_pts == 0:
                return []
            
            S = [[] for _ in range(n_pts)]  # S[i] = indices dominated by i
            n_dominating = np.zeros(n_pts, dtype=int)
            fronts = []
            
            # Calculate dominance
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dominating[i] += 1
            
            # First front
            front = [i for i in range(n_pts) if n_dominating[i] == 0]
            if not front:
                return []
            fronts.append(front)
            
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                front = next_front
                k += 1
            return fronts

        def crowding_distance(F_arr, nd_indices):
            """Compute crowding distance for non-dominated points."""
            if len(nd_indices) <= 2:
                return np.ones(len(nd_indices))
            distances = np.zeros(len(nd_indices))
            for k in range(m):
                sorted_idx = np.argsort(F_arr[nd_indices, k])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = F_arr[nd_indices, k].max() - F_arr[nd_indices, k].min()
                if f_range > 1e-15:
                    for i in range(1, len(nd_indices) - 1):
                        distances[sorted_idx[i]] += (
                            F_arr[nd_indices[sorted_idx[i + 1]], k] -
                            F_arr[nd_indices[sorted_idx[i - 1]], k]
                        ) / f_range
            return distances

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # 1. Create offspring
            n_offspring = pop_size
            X_off = np.zeros((n_offspring, n))
            F_off = np.zeros((n_offspring, m))
            
            for i in range(n_offspring):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Binary Tournament Selection
                # Sort population by non-dominated fronts and crowding distance
                fronts = fast_non_dominated_sort(F)
                if not fronts:
                    # Fallback if something went wrong
                    break
                    
                # Build selection pool with priorities
                # We need to select parents from the current population X, F
                # Standard NSGA-II: Select from population using tournament on (front_id, crowding)
                
                # To speed up, we can just pick 2 random and compare their rank/crowding
                # But we need ranks. Let's compute ranks once per generation.
                # This is expensive. Let's use a simpler heuristic for small pop:
                # Pick 2 random, if one is in a better front, pick it. If same front, pick higher crowding.
                
                # Pre-compute front IDs and crowding for current X, F
                front_ids = np.zeros(len(F), dtype=int)
                all_cd = np.zeros(len(F))
                for f_id, front in enumerate(fronts):
                    for idx in front:
                        front_ids[idx] = f_id
                    cd_vals = crowding_distance(F, np.array(front))
                    for j, idx in enumerate(front):
                        all_cd[idx] = cd_vals[j]

                # Tournament Selection
                idx1, idx2 = rng.integers(pop_size, size=2)
                if front_ids[idx1] < front_ids[idx2]:
                    p1_idx = idx1
                elif front_ids[idx2] < front_ids[idx1]:
                    p1_idx = idx2
                else:
                    if all_cd[idx1] > all_cd[idx2]:
                        p1_idx = idx1
                    else:
                        p1_idx = idx2
                p1 = X[p1_idx]
                
                idx1, idx2 = rng.integers(pop_size, size=2)
                if front_ids[idx1] < front_ids[idx2]:
                    p2_idx = idx1
                elif front_ids[idx2] < front_ids[idx1]:
                    p2_idx = idx2
                else:
                    if all_cd[idx1] > all_cd[idx2]:
                        p2_idx = idx1
                    else:
                        p2_idx = idx2
                p2 = X[p2_idx]

                # SBX Crossover
                if rng.random() < 0.9:
                    child = np.zeros(n)
                    for d in range(n):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 20)
                        child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                else:
                    child = p1.copy()

                # Polynomial Mutation
                # Adaptive step size
                progress = used / self.budget
                # Start with larger steps, shrink later
                # WFG4 bounds are usually [0,1] or similar.
                # Let's use a relative step.
                step_size = 0.1 * (1.0 - 0.5 * progress)
                
                for d in range(n):
                    if rng.random() < 0.3: # Mutation rate
                        child[d] += step_size * (hi[d] - lo[d]) * rng.standard_normal()
                
                child = np.clip(child, lo, hi)
                F_off[i] = problem(child)
                X_off[i] = child
                used += 1

            # 2. Environmental Selection
            # Combine parent and offspring
            X_comb = np.vstack([X, X_off[:n_offspring]])
            F_comb = np.vstack([F, F_off[:n_offspring]])
            
            # Sort combined population
            fronts = fast_non_dominated_sort(F_comb)
            
            X_new = np.zeros((pop_size, n))
            F_new = np.zeros((pop_size, m))
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    # Add entire front
                    for idx in front:
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                else:
                    # Need to select subset of this front
                    # Calculate crowding distance for this front
                    front_arr = np.array(front)
                    cd_vals = crowding_distance(F_comb, front_arr)
                    # Sort by crowding distance descending
                    sorted_cd = np.argsort(cd_vals)[::-1]
                    needed = pop_size - count
                    for k in range(needed):
                        idx = front_arr[sorted_cd[k]]
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                    break
                if count >= pop_size:
                    break

            X = X_new
            F = F_new

        return X
# EVOLVE-BLOCK-END
