"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT4 using NSGA-II inspired approach with x1 bias."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # 300s limit, leave 10s safety margin
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: 20 is a good balance for 200 budget
        pop_size = 20
        
        # Initialize population
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            # Bias x1 towards 0, which is the global optimum for g(x) in ZDT4
            if np.random.rand() < 0.5:
                X[i, 0] = 0.0
            else:
                X[i, 0] = np.random.uniform(lo[0], hi[0])
            # Other dimensions uniform
            X[i, 1:] = np.random.uniform(lo[1:], hi[1:])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort and crowding distance
            ranks, fronts = self._non_dominated_sort(F)
            cd = self._crowding_distance(F, fronts)
            
            # Select two parents using tournament selection
            p1_idx = self._tournament_select(ranks, cd, 2)
            p2_idx = self._tournament_select(ranks, cd, 2)
            
            # Crossover (SBX)
            child = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi)
            
            # Mutation (Polynomial)
            child = self._polynomial_mutation(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Combine and select next generation
            X_new = np.vstack([X, child])
            F_new = np.vstack([F, fc])
            
            ranks_new, fronts_new = self._non_dominated_sort(F_new)
            cd_new = self._crowding_distance(F_new, fronts_new)
            
            # Sort by rank (ascending) and crowding distance (descending)
            # lexsort uses last key as primary, so we put cd (negated for desc) last?
            # No, lexsort((key1, key2)) sorts by key2 then key1.
            # We want primary: rank (asc), secondary: cd (desc)
            # So key1 = rank, key2 = -cd
            sort_keys = (-cd_new, ranks_new)
            keep_indices = np.lexsort(sort_keys)[:pop_size]
            
            X = X_new[keep_indices]
            F = F_new[keep_indices]
            
        return X

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[j] += 1
        
        front = []
        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while len(front) > 0:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
            
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        n = len(F)
        cd = np.zeros(n)
        for front in fronts:
            if len(front) <= 2:
                cd[front] = np.inf
                continue
            for k in range(self.n_obj):
                sorted_idx = sorted(front, key=lambda i: F[i, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], k]
                f_max = F[sorted_idx[-1], k]
                if f_max > f_min:
                    for j in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[j]] += (F[sorted_idx[j+1], k] - F[sorted_idx[j-1], k]) / (f_max - f_min)
        return cd

    def _tournament_select(self, ranks, cd, size=2):
        n = len(ranks)
        indices = np.random.randint(0, n, size)
        # Select best: min rank, max cd
        best_idx = None
        best_rank = np.inf
        best_cd = -np.inf
        for idx in indices:
            if ranks[idx] < best_rank or (ranks[idx] == best_rank and cd[idx] > best_cd):
                best_rank = ranks[idx]
                best_cd = cd[idx]
                best_idx = idx
        return best_idx

    def _sbx_crossover(self, p1, p2, lo, hi):
        child = np.zeros_like(p1)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                x1, x2 = p1[i], p2[i]
            else:
                x1, x2 = p2[i], p1[i]
            
            if abs(x1 - x2) < 1e-9:
                child[i] = x1
            else:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (2 + 1)) # eta_c = 20, so 1/(20+1)? No, standard SBX
                    # Let's use eta_c = 20
                    beta = (2 * u) ** (1 / 21)
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / 21)
                
                child[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            
            # Clip to bounds
            child[i] = np.clip(child[i], lo[i], hi[i])
        return child

    def _polynomial_mutation(self, x, lo, hi):
        eta_m = 20
        for i in range(self.dim):
            if np.random.rand() < 0.5: # 50% mutation rate? Or 1/dim?
                # Standard is 1/dim
                pass # We'll do 1/dim below
        
        # 1/dim mutation rate
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                x_l = (x[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                x_u = (hi[i] - x[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                
                if u < 0.5:
                    delta = (2 * u + 1 - x_l) ** (1 / (eta_m + 1))
                    delta_q = 1 - delta ** (eta_m + 1)
                else:
                    delta = (2 * (1 - u) + x_u - 1) ** (1 / (eta_m + 1))
                    delta_q = 1 - delta ** (eta_m + 1)
                
                x[i] += delta_q * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
# EVOLVE-BLOCK-END
