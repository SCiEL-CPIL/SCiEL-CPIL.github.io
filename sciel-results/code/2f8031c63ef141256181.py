import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size and generations
        # Total budget: 200
        # Init: N
        # Generations: G * N
        # N + G*N <= 200
        N = 20
        G = (self.budget - N) // N  # 9 generations
        
        # Initialize population
        X = lo + range_ * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        evals = N

        # NSGA-II Main Loop
        for gen in range(G):
            if time.time() > deadline:
                break
            
            # Fast Non-Dominated Sorting and Crowding Distance
            ranks, crowding = self._fast_non_dominated_sort(F)
            
            # Generate Offspring
            offspring_X = []
            for i in range(N):
                # Tournament Selection (Size 2)
                p1 = self._tournament_select(X, ranks, crowding)
                p2 = self._tournament_select(X, ranks, crowding)
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Polynomial Mutation
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                
                # We only need N offspring, so we take one from each pair?
                # Standard NSGA-II produces N offspring from N pairs (using both c1, c2 and selecting N from 2N? No, usually N parents -> N children).
                # To keep it simple and fast: 1 parent pair -> 1 child (using c1).
                offspring_X.append(c1)
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += N
            
            # Combine Parents and Offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Select Next Generation
            ranks_comb, crowding_comb = self._fast_non_dominated_sort(F_comb)
            X, F = self._select_next_gen(X_comb, F_comb, ranks_comb, crowding_comb, N)
        
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        ranks = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[j] += 1
        
        front = []
        for i in range(n):
            if n_dom[i] == 0:
                front.append(i)
                ranks[i] = 0
        
        k = 0
        while len(front) > 0:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            front = next_front
            k += 1
        
        # Crowding Distance
        crowding = [0.0] * n
        for r in range(k + 1):
            front_idx = [i for i in range(n) if ranks[i] == r]
            if len(front_idx) <= 2:
                for i in front_idx:
                    crowding[i] = float('inf')
            else:
                for m in range(self.n_obj):
                    sorted_idx = sorted(front_idx, key=lambda x: F[x, m])
                    crowding[sorted_idx[0]] = float('inf')
                    crowding[sorted_idx[-1]] = float('inf')
                    f_min = F[sorted_idx[0], m]
                    f_max = F[sorted_idx[-1], m]
                    if f_max - f_min == 0:
                        continue
                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        crowding[idx] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / (f_max - f_min)
        
        return ranks, crowding

    def _tournament_select(self, X, ranks, crowding):
        i = np.random.randint(len(X))
        j = np.random.randint(len(X))
        if ranks[i] < ranks[j]:
            return X[i]
        elif ranks[j] < ranks[i]:
            return X[j]
        else:
            if crowding[i] > crowding[j]:
                return X[i]
            else:
                return X[j]

    def _sbx_crossover(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if p1[i] != p2[i]:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / 16)
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / 16)
                    
                    c1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    c2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, lo, hi):
        for i in range(self.dim):
            if np.random.rand() < 0.1: # Mutation rate 0.1
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / 21) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / 21)
                
                x[i] = x[i] + delta * (hi[i] - lo[i])
        
        return np.clip(x, lo, hi)

    def _select_next_gen(self, X, F, ranks, crowding, N):
        # Select N individuals from 2N based on rank and crowding
        indices = np.arange(len(X))
        # Sort by rank, then by crowding (descending)
        sorted_indices = sorted(indices, key=lambda i: (ranks[i], -crowding[i]))
        selected_indices = sorted_indices[:N]
        return X[selected_indices], F[selected_indices]
