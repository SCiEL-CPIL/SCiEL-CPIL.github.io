"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search (NSGA-II) with deadline guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = min(20, self.budget // 10)
        if pop_size < 2:
            pop_size = 2
            
        # Initial population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
            
            I = [i for i in range(n) if rank[i] == 0]
            k = 0
            while len(I) > 0:
                k += 1
                Q = []
                for i in I:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k
                            Q.append(j)
                I = Q
            return rank

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            distances = np.zeros(n)
            front_F = F[indices]
            m = self.n_obj
            for m_idx in range(m):
                sorted_idx = np.argsort(front_F[:, m_idx])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_min = front_F[sorted_idx[0], m_idx]
                f_max = front_F[sorted_idx[-1], m_idx]
                if f_max - f_min == 0:
                    continue
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (front_F[sorted_idx[k+1], m_idx] - front_F[sorted_idx[k-1], m_idx]) / (f_max - f_min)
            return distances

        # Helper: Selection
        def select_parents(F, rank, pop_indices):
            # Calculate crowding distance for each front
            fronts = {}
            for i in pop_indices:
                r = rank[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            cd = np.zeros(len(pop_indices))
            # Map original indices to positions in pop_indices for CD calculation?
            # Actually, let's just calculate CD on the full population based on rank groups
            # Simpler: Calculate CD for each front and store in a dict
            cd_dict = {}
            for r, idxs in fronts.items():
                cd_vals = crowding_distance(F, idxs)
                for i, val in zip(idxs, cd_vals):
                    cd_dict[i] = val
            
            # Tournament Selection
            parents = []
            for _ in range(pop_size):
                i1 = pop_indices[rng.integers(len(pop_indices))]
                i2 = pop_indices[rng.integers(len(pop_indices))]
                
                r1, r2 = rank[i1], rank[i2]
                if r1 < r2:
                    parents.append(i1)
                elif r2 < r1:
                    parents.append(i2)
                else:
                    if cd_dict[i1] > cd_dict[i2]:
                        parents.append(i1)
                    else:
                        parents.append(i2)
            return parents

        # Helper: Crossover (SBX)
        def sbx_crossover(a, b, eta=20.0):
            if rng.random() < 0.5:
                return a.copy(), b.copy()
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    child1[i] = a[i]
                    child2[i] = b[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
                    child2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
            return child1, child2

        # Helper: Mutation
        def mutation(x, eta=20.0):
            y = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    y[i] = x[i] + delta_q * range_[i]
            y = np.clip(y, lo, hi)
            return y

        # Main Evolutionary Loop (NSGA-II)
        # Use 80% of budget for evolution
        evo_budget = int(self.budget * 0.8)
        
        while used < evo_budget:
            if time.time() > deadline:
                break
                
            rank = fast_non_dominated_sort(F)
            parents = select_parents(F, rank, np.arange(pop_size))
            
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(pop_size, evo_budget - used)
            
            for _ in range(n_offspring):
                p1_idx = parents[rng.integers(pop_size)]
                p2_idx = parents[rng.integers(pop_size)]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(c2) # Fixed typo: c2 not c1
                
                if used >= evo_budget:
                    break
            
            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            combined_X = np.vstack((X, offspring_X))
            combined_F = np.vstack((F, offspring_F))
            
            # Environmental Selection
            combined_rank = fast_non_dominated_sort(combined_F)
            
            # Group by rank
            fronts = {}
            for i in range(len(combined_F)):
                r = combined_rank[i]
                if r not in fronts:
                    fronts[r] = []
                fronts[r].append(i)
            
            # Sort fronts by rank
            sorted_ranks = sorted(fronts.keys())
            
            new_X = []
            new_F = []
            
            for r in sorted_ranks:
                idxs = fronts[r]
                if len(new_X) + len(idxs) <= pop_size:
                    for i in idxs:
                        new_X.append(combined_X[i])
                        new_F.append(combined_F[i])
                else:
                    # Need to select from this front
                    needed = pop_size - len(new_X)
                    cd_vals = crowding_distance(combined_F, idxs)
                    # Sort by CD descending
                    sorted_idxs = [idxs[i] for i in np.argsort(-cd_vals)]
                    for i in range(needed):
                        new_X.append(combined_X[sorted_idxs[i]])
                        new_F.append(combined_F[sorted_idxs[i]])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)

        # Local Polish Phase
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Get non-dominated points
            rank = fast_non_dominated_sort(F)
            non_dom_idx = np.where(rank == 0)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))
            
            # Polish a few points
            max_polish_points = min(5, len(non_dom_idx), remaining // 10)
            for idx in non_dom_idx[:max_polish_points]:
                if used >= self.budget or time.time() > deadline:
                    break
                
                best_x = X[idx].copy()
                best_f = F[idx].copy()
                
                # Simple coordinate search
                for dim_i in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        step = 0.05 * range_[dim_i]
                        trial = best_x.copy()
                        trial[dim_i] += direction * step
                        trial = np.clip(trial, lo, hi)
                        
                        if np.array_equal(trial, best_x):
                            continue
                        
                        trial_f = problem(trial)
                        used += 1
                        
                        # Accept if dominates
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial
                            best_f = trial_f
                            X[idx] = best_x
                            F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
