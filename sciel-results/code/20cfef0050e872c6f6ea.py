"""Rewritten multi-objective optimizer for WFG4 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Determine population size
        # We need N + G*N <= budget. 
        # Let's aim for N=20, G=9 -> 200 evals.
        N = 20
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # NSGA-II parameters
        eta_c = 20.0
        eta_m = 20.0
        p_c = 1.0  # Crossover probability
        p_m = 1.0 / dim  # Mutation probability per gene

        while used + N <= budget:
            # Create offspring
            offspring_X = np.empty((N, dim))
            offspring_F = np.empty((N, n_obj))
            
            # Assign ranks and crowding distances to current population
            ranks, crowding = self._fast_non_dominated_sorting(F)
            
            for i in range(0, N, 2):
                # Selection: Binary Tournament
                p1_idx = self._tournament_select(rng, N, ranks, crowding)
                p2_idx = self._tournament_select(rng, N, ranks, crowding)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover
                if rng.random() < p_c:
                    c1, c2 = self._sbx_crossover(rng, p1, p2, lo, hi, eta_c)
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                
                # Mutation
                c1 = self._poly_mutation(rng, c1, lo, hi, eta_m, p_m)
                c2 = self._poly_mutation(rng, c2, lo, hi, eta_m, p_m)
                
                # Evaluate
                f1 = np.array(problem(c1), dtype=float)
                used += 1
                f2 = np.array(problem(c2), dtype=float)
                used += 1
                
                offspring_X[i] = c1
                offspring_F[i] = f1
                
                if i + 1 < N:
                    offspring_X[i+1] = c2
                    offspring_F[i+1] = f2

            # Combine parent and offspring
            X_comb = np.vstack((X, offspring_X))
            F_comb = np.vstack((F, offspring_F))
            
            # Fast non-dominated sorting on combined
            ranks_comb, crowding_comb = self._fast_non_dominated_sorting(F_comb)
            
            # Select next generation
            next_X = np.empty((N, dim))
            next_F = np.empty((N, n_obj))
            
            selected = []
            # Max rank is usually small, but we iterate until selected
            max_rank = np.max(ranks_comb)
            for rank in range(max_rank + 1):
                # Get indices of individuals with this rank
                indices = np.where(ranks_comb == rank)[0]
                if len(indices) == 0:
                    continue
                
                if len(selected) + len(indices) <= N:
                    selected.extend(indices)
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - len(selected)
                    # Sort indices by crowding distance descending
                    crowding_vals = crowding_comb[indices]
                    sorted_indices = indices[np.argsort(crowding_vals)[::-1]]
                    selected.extend(sorted_indices[:remaining])
                    break
            
            # Ensure we have exactly N selected
            if len(selected) < N:
                while len(selected) < N:
                    selected.append(rng.integers(2*N))
            
            X = X_comb[selected]
            F = F_comb[selected]

        return X

    def _tournament_select(self, rng, N, ranks, crowding):
        i1 = rng.integers(N)
        i2 = rng.integers(N)
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if crowding[i1] > crowding[i2]:
                return i1
            elif crowding[i2] > crowding[i1]:
                return i2
            else:
                return i1

    def _sbx_crossover(self, rng, p1, p2, lo, hi, eta_c):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() <= 1.0 / (eta_c + 1.0):
                        beta = 2.0 * rng.random()
                    else:
                        beta = 2.0 * (1.0 - rng.random())
                    
                    beta = 1.0 / (beta * eta_c + 1.0)
                    
                    c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                    
                    if c1[i] > hi[i]:
                        c1[i] = hi[i]
                    if c1[i] < lo[i]:
                        c1[i] = lo[i]
                    if c2[i] > hi[i]:
                        c2[i] = hi[i]
                    if c2[i] < lo[i]:
                        c2[i] = lo[i]
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        return c1, c2

    def _poly_mutation(self, rng, x, lo, hi, eta_m, p_m):
        x_mut = x.copy()
        for i in range(len(x)):
            if rng.random() < p_m:
                if abs(hi[i] - lo[i]) < 1e-14:
                    continue
                
                r = rng.random()
                if r < 0.5:
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                    mut_pow = 2.0 / (eta_m + 1.0) * (delta1 + 1.0)
                    if r <= 1.0 / mut_pow:
                        delta = (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta = (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] - delta * (hi[i] - lo[i])
                else:
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                    mut_pow = 2.0 / (eta_m + 1.0) * (delta2 + 1.0)
                    if (1.0 - r) <= 1.0 / mut_pow:
                        delta = (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta = (2.0 * r) ** (1.0 / (eta_m + 1.0))
                    x_mut[i] = x_mut[i] + delta * (hi[i] - lo[i])
                    
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def _fast_non_dominated_sorting(self, F):
        n = len(F)
        m = F.shape[1]
        
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [list() for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n, dtype=float)
        
        current_front = [i for i in range(n) if domination_count[i] == 0]
        rank = 0
        
        while len(current_front) > 0:
            for i in current_front:
                ranks[i] = rank
            rank += 1
            
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            # Calculate crowding distance for current front
            if len(current_front) > 1:
                for k in range(m):
                    front_F = F[current_front, k]
                    min_val = np.min(front_F)
                    max_val = np.max(front_F)
                    range_val = max_val - min_val
                    if range_val == 0:
                        continue
                    # Sort front by k-th objective
                    sorted_indices = np.argsort(front_F)
                    sorted_front = [current_front[i] for i in sorted_indices]
                    crowding[sorted_front[0]] = float('inf')
                    crowding[sorted_front[-1]] = float('inf')
                    for idx in range(1, len(sorted_front) - 1):
                        i = sorted_front[idx]
                        prev_i = sorted_front[idx - 1]
                        next_i = sorted_front[idx + 1]
                        crowding[i] += (F[next_i, k] - F[prev_i, k]) / range_val
            
            current_front = next_front
            
        return ranks, crowding
# EVOLVE-BLOCK-END
