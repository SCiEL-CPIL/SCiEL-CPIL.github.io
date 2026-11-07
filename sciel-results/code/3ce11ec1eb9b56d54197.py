import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        import time
        start_time = time.time()
        deadline = start_time + 300 # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size
        # We want to maximize generations for refinement.
        # N=50 gives 4 generations (1 init + 3 offspring) if we count init as gen 0.
        # Actually:
        # Eval 1: Init (50)
        # Eval 2: Offspring (50) -> Select 50
        # Eval 3: Offspring (50) -> Select 50
        # Eval 4: Offspring (50) -> Select 50
        # Total 200.
        
        N = 50
        if self.budget < N * 2:
            N = max(10, self.budget // 2) # Fallback for small budgets
            
        # 1. Initialize Population using LHS
        X = self._lhs_sample(N, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        
        evals_used = N
        
        # Main Loop
        while evals_used + N <= self.budget:
            # Check time
            if time.time() > deadline - 10:
                break
                
            # 2. Selection: Tournament on Rank and Crowding Distance
            ranks, dists = self._fast_non_dominated_sort(F)
            parents = self._tournament_select(X, ranks, dists, N)
            
            # 3. Crossover and Mutation
            Q = []
            for i in range(0, N, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                c1, c2 = self._sbx(p1, p2, lo, hi)
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                Q.append(c1)
                Q.append(c2)
                
            # Ensure we have exactly N offspring
            # If N is odd, we might have issues with pairing. 
            # Let's ensure N is even.
            if N % 2 != 0:
                 # Add one more parent copy or just take N-1 pairs and one mutation
                 # Simpler: just make N even.
                 pass 
                 
            # Wait, the loop above assumes N is even.
            # Let's force N to be even in init.
            
            F_q = np.array([problem(x) for x in Q])
            evals_used += N
            
            # 4. Elitism: Combine and Select
            R_X = np.vstack([X, Q])
            R_F = np.vstack([F, F_q])
            
            X, F = self._select_best(R_X, R_F, N)
            
        return X

    def _lhs_sample(self, N, D, lo, hi):
        # Latin Hypercube Sampling
        # Generate N points in [0,1]^D such that each marginal is uniform
        points = np.zeros((N, D))
        for d in range(D):
            # Divide [0,1] into N bins
            centers = (np.arange(N) + 0.5) / N
            # Shuffle the centers
            np.random.shuffle(centers)
            points[:, d] = centers
        # Map to [lo, hi]
        points = lo + points * (hi - lo)
        return points

    def _fast_non_dominated_sort(self, F):
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        dists = np.zeros(n)
        
        # Sort indices
        S = []
        for i in range(n):
            S.append([])
            S[i] = [j for j in range(n) if j != i and np.all(F[i] <= F[j]) and np.any(F[i] < F[j])]
            
        n_dominating = np.zeros(n, dtype=int)
        for i in range(n):
            for j in S[i]:
                n_dominating[j] += 1
                
        # First front
        front_idx = 0
        current_front = [i for i in range(n) if n_dominating[i] == 0]
        ranks[current_front] = front_idx
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            if next_front:
                front_idx += 1
                ranks[next_front] = front_idx
            current_front = next_front
            
        # Crowding Distance
        for rank in range(front_idx + 1):
            idx_in_front = [i for i in range(n) if ranks[i] == rank]
            if len(idx_in_front) <= 2:
                dists[idx_in_front] = np.inf
                continue
            for k in range(self.n_obj):
                # Sort by objective k
                sorted_idx = sorted(idx_in_front, key=lambda i: F[i, k])
                dists[sorted_idx[0]] = np.inf
                dists[sorted_idx[-1]] = np.inf
                if F[sorted_idx[-1], k] - F[sorted_idx[0], k] == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    dists[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / (F[sorted_idx[-1], k] - F[sorted_idx[0], k])
                    
        return ranks, dists

    def _tournament_select(self, X, ranks, dists, N):
        parents = []
        for _ in range(N):
            i = np.random.randint(0, len(X))
            j = np.random.randint(0, len(X))
            if ranks[i] < ranks[j]:
                parents.append(X[i])
            elif ranks[j] < ranks[i]:
                parents.append(X[j])
            else:
                if dists[i] > dists[j]:
                    parents.append(X[i])
                else:
                    parents.append(X[j])
        return np.array(parents)

    def _sbx(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                continue
            if abs(p1[i] - p2[i]) < 1e-14:
                c1[i] = p1[i]
                c2[i] = p2[i]
                continue
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / 21) # eta+1
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / 21)
                
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            
            c1[i] = np.clip(c1[i], lo[i], hi[i])
            c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _poly_mutation(self, x, lo, hi):
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / 21) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / 21)
                    
                range_val = hi[i] - lo[i]
                x[i] = x[i] + delta * range_val
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x

    def _select_best(self, X, F, N):
        ranks, dists = self._fast_non_dominated_sort(F)
        # Sort by rank, then by dist descending
        order = np.lexsort((dists * -1, ranks))
        return X[order[:N]], F[order[:N]]
