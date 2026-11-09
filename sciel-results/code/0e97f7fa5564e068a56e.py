"""Rewritten optimizer for RE21 using Sobol initialization and NSGA-II refinement."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 300  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Global Exploration with Sobol Sequence
        # Use 40% of budget for initial sampling
        n_init = max(10, int(self.budget * 0.4))
        
        # Generate Sobol points
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2 for best quality, but we can just take n_init
        # We need to ensure we don't exceed budget
        n_init = min(n_init, self.budget)
        
        # Sobol generates points in [0, 1]^d
        # We need to map to [lo, hi]
        # Note: qmc.Sobol.random(n) returns n points
        # For small n, we might just use random if Sobol setup is too slow, but it's fast.
        try:
            # Sobol sequence
            points = sobol.random(n_init)
        except Exception:
            points = np.random.rand(n_init, self.dim)
            
        X_init = lo + points * (hi - lo)
        F_init = np.array([problem(x) for x in X_init])
        
        used_evals = n_init
        if used_evals >= self.budget:
            return X_init
            
        # 2. Local Exploitation with NSGA-II
        # Population size
        pop_size = 20
        # Ensure we have enough budget for at least one generation
        # Each generation creates pop_size offspring
        # So max generations = (budget - used_evals) // pop_size
        
        # Initialize population from best non-dominated points of initial sampling
        # If we have fewer than pop_size, fill with random
        X = X_init.copy()
        F = F_init.copy()
        
        if len(X) < pop_size:
            n_fill = pop_size - len(X)
            X_fill = lo + np.random.rand(n_fill, self.dim) * (hi - lo)
            F_fill = np.array([problem(x) for x in X_fill])
            used_evals += n_fill
            X = np.vstack([X, X_fill])
            F = np.vstack([F, F_fill])
        elif len(X) > pop_size:
            # Select best pop_size non-dominated points
            # Simple selection: just take first pop_size, or sort by hypervolume contribution?
            # For simplicity and speed, let's just take a random subset or the first ones.
            # Better: keep the non-dominated ones if possible.
            nd_indices = self._find_non_dominated_indices(F)
            if len(nd_indices) >= pop_size:
                X = X[nd_indices[:pop_size]]
                F = F[nd_indices[:pop_size]]
            else:
                # If not enough ND, fill with random from initial
                remaining = np.setdiff1d(np.arange(len(X)), nd_indices)
                need = pop_size - len(nd_indices)
                fill_idx = np.random.choice(remaining, need, replace=False)
                idx = np.concatenate([nd_indices, fill_idx])
                X = X[idx]
                F = F[idx]

        # NSGA-II Loop
        while used_evals + pop_size <= self.budget:
            # Check time
            if time.time() > deadline - 5:
                break
                
            # Generate offspring
            X_off = np.zeros((pop_size, self.dim))
            F_off = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Select parents using tournament selection on ranks
                # We need ranks for current population
                ranks = self._fast_non_dominated_sort(F)
                
                # Tournament selection
                p1 = self._tournament_select(X, ranks, k=2)
                p2 = self._tournament_select(X, ranks, k=2)
                
                # Crossover (SBX)
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Mutation (Polynomial)
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                
                X_off[i] = c1
                # We evaluate c1 now. What about c2?
                # Standard NSGA-II creates 2 offspring per pair, but here we loop pop_size times.
                # To keep it simple and within budget:
                # If we generate 2 offspring per iteration, we need pop_size/2 iterations.
                # Let's adjust: generate 1 offspring per loop iteration to strictly control count.
                # Or generate 2 and evaluate both if budget allows.
                # Let's stick to 1 offspring per iteration for simplicity in budget tracking.
                # But SBX produces 2. Let's just take c1.
                
                F_off[i] = problem(c1)
                used_evals += 1
                
                if used_evals >= self.budget:
                    break
                    
            if used_evals >= self.budget:
                break
                
            # Combine parents and offspring
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sorting
            ranks = self._fast_non_dominated_sort(F_comb)
            
            # Crowding distance for selection
            # We need to select pop_size best individuals
            # Sort by rank, then by crowding distance within same rank
            
            # Calculate crowding distance for each front
            # Fast non-dominated sort returns ranks (array of same length as F_comb)
            # We need to group by rank
            
            # For simplicity and speed, we can use a simpler selection:
            # Sort by rank, then by distance to reference point or similar?
            # No, crowding is better. Let's implement a simplified crowding.
            
            # Actually, calculating full crowding is O(N^2) or O(N log N).
            # With N=40, it's fast.
            
            # Let's implement selection properly.
            selected_idx = self._select_best(X_comb, F_comb, ranks, pop_size)
            
            X = X_comb[selected_idx]
            F = F_comb[selected_idx]

        return X

    def _find_non_dominated_indices(self, F):
        n = len(F)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # If i dominates j, j is not non-dominated
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    is_nd[j] = False
        return np.where(is_nd)[0]

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)] # S[i] contains indices dominated by i
        n_i = np.zeros(n, dtype=int) # number of solutions dominating i
        rank = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_i[j] += 1
                # j dominates i?
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_i[i] += 1
                    
        # First front
        front = [i for i in range(n) if n_i[i] == 0]
        k = 0
        for i in front:
            rank[i] = k
            
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_i[j] -= 1
                    if n_i[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
        return rank

    def _tournament_select(self, X, ranks, k=2):
        n = len(X)
        if n < k:
            return X[np.random.randint(n)]
        indices = np.random.choice(n, k, replace=False)
        best_idx = indices[0]
        for i in indices[1:]:
            if ranks[i] < ranks[best_idx]:
                best_idx = i
        return X[best_idx]

    def _sbx_crossover(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        eta = 20.0
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, lo, hi, eta=20.0):
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                range_i = hi[i] - lo[i]
                # Simple polynomial mutation formula
                # y = x + delta * (hi - lo)
                # This is a simplified version. Standard poly mutation uses complex formula.
                # Let's use a simpler Gaussian mutation scaled by range for robustness
                # Or stick to the standard formula:
                # x_new = x + (hi - lo) * delta
                x_mut[i] = x[i] + range_i * delta
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def _select_best(self, X, F, ranks, n_select):
        n = len(X)
        if n <= n_select:
            return np.arange(n)
        
        # Sort by rank
        sorted_indices = np.argsort(ranks)
        
        selected = []
        current_rank = 0
        while len(selected) < n_select:
            # Get all indices with current rank
            idx_in_rank = sorted_indices[ranks[sorted_indices] == current_rank]
            
            if len(idx_in_rank) == 0:
                current_rank += 1
                continue
                
            # If we can fit all of them
            if len(selected) + len(idx_in_rank) <= n_select:
                selected.extend(idx_in_rank)
            else:
                # Need to select some from this rank based on crowding distance
                # Calculate crowding distance for these points
                # Simplified: distance to nearest neighbor in this front
                # For speed, let's just pick random ones if crowding is too complex to implement quickly
                # But crowding is important. Let's implement basic crowding.
                
                # Calculate distances
                dists = np.zeros(len(idx_in_rank))
                for i, idx_i in enumerate(idx_in_rank):
                    min_dist = float('inf')
                    for j, idx_j in enumerate(idx_in_rank):
                        if i == j:
                            continue
                        d = np.sum((F[idx_i] - F[idx_j])**2)
                        if d < min_dist:
                            min_dist = d
                    dists[i] = min_dist
                    
                # Select top (n_select - len(selected)) by distance
                n_need = n_select - len(selected)
                # Higher distance is better
                top_idx = np.argsort(dists)[::-1][:n_need]
                for t in top_idx:
                    selected.append(idx_in_rank[t])
            
            current_rank += 1
            
        return np.array(selected)

# EVOLVE-BLOCK-END
