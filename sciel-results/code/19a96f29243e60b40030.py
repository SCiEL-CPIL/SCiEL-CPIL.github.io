import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size for DE
        N = 20
        if self.budget < N:
            X = lo + np.random.random((self.budget, self.dim)) * span
            F = np.array([problem(x) for x in X])
            return X

        G = (self.budget - N) // N
        
        # Initialize population using Latin Hypercube Sampling for better spread
        X = np.zeros((N, self.dim))
        for d in range(self.dim):
            bins = np.linspace(lo[d], hi[d], N + 1)
            indices = np.random.permutation(N)
            for i in range(N):
                bin_low = bins[indices[i]]
                bin_high = bins[indices[i] + 1]
                X[i, d] = np.random.uniform(bin_low, bin_high)
                
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_d = [0] * n
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
            
            current_front = []
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = self.n_obj
            dist = {i: 0.0 for i in front_indices}
            
            for k in range(m):
                sorted_indices = sorted(front_indices, key=lambda i: F[i, k])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0], k]
                f_max = F[sorted_indices[-1], k]
                range_k = f_max - f_min
                if range_k == 0:
                    continue
                
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    dist[idx] += (F[idx_next, k] - F[idx_prev, k]) / range_k
            
            return dist

        # DE Parameters
        F_diff = 0.8  # Differential weight
        CR = 0.9      # Crossover rate

        for gen in range(G):
            # Generate offspring
            Q = np.zeros((N, self.dim))
            for i in range(N):
                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                
                # Mutation
                v = X[r1] + F_diff * (X[r2] - X[r3])
                
                # Binomial Crossover
                # Create a random vector of 0s and 1s
                j_rand = np.random.randint(self.dim)
                mask = np.random.random(self.dim) < CR
                mask[j_rand] = True # Ensure at least one gene is taken from v
                
                Q[i] = np.where(mask, v, X[i])
                
                # Clip to bounds
                Q[i] = np.clip(Q[i], lo, hi)
            
            F_Q = np.array([problem(x) for x in Q])
            evals += N
            
            # Combine P and Q
            R_X = np.vstack((X, Q))
            R_F = np.vstack((F, F_Q))
            
            # Non-dominated sort on R
            fronts_R, rank_R = non_dominated_sort(R_F)
            
            # Select next P
            X_new = np.zeros((N, self.dim))
            F_new = np.zeros((N, self.n_obj))
            count = 0
            
            for front in fronts_R:
                if count + len(front) <= N:
                    for i in front:
                        X_new[count] = R_X[i]
                        F_new[count] = R_F[i]
                        count += 1
                else:
                    needed = N - count
                    cd_front = crowding_distance(R_F, front)
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    for i in sorted_front[:needed]:
                        X_new[count] = R_X[i]
                        F_new[count] = R_F[i]
                        count += 1
                    break
                
                if count >= N:
                    break
            
            X = X_new
            F = F_new
            
        return X
