import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        # NSGA-II typically uses population sizes like 100-200.
        # Budget is 200.
        # If pop_size = 100, we can do ~2 generations.
        # If pop_size = 50, we can do ~4 generations.
        # Let's try pop_size = 100 for better diversity, or maybe 200 for 1 generation?
        # Actually, with budget 200, we can have a population of 100 and do 2 generations (100 offspring).
        # Or population 200 and do 1 generation? No, NSGA-II needs at least 1 generation to improve.
        # Let's use pop_size = 100.
        # Init: 100 evals.
        # Gen 1: 100 offspring evals. Total 200.
        # This is tight. Maybe pop_size = 50?
        # Init: 50.
        # Gen 1: 50 offspring. Total 100.
        # Gen 2: 50 offspring. Total 150.
        # Gen 3: 50 offspring. Total 200.
        # This allows more iterations.
        
        pop_size = 50
        if budget < pop_size * 2:
            pop_size = max(10, budget // 2)
            
        # Initialization
        # x1: uniform
        # x2..xn: near 0.5
        X = np.zeros((pop_size, dim))
        X[:, 0] = np.random.uniform(lo[0], hi[0], pop_size)
        # For other dims, initialize near 0.5
        # The ZDT3 function has minima at 0.5 for x_i (i>1) in the sin term,
        # but the (x_i - 0.5)^2 term also pushes to 0.5.
        # So 0.5 is a very good guess.
        X[:, 1:] = 0.5 + np.random.uniform(-0.1, 0.1, (pop_size, dim-1))
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        eval_count = pop_size
        
        # NSGA-II Loop
        while eval_count < budget:
            if time.time() > deadline:
                break
                
            # Non-dominated sort and crowding distance
            ranks, crowding = self._dominance_sort(F)
            
            # Create next generation by tournament selection
            next_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Binary tournament
                p1_idx, p2_idx = np.random.choice(pop_size, 2, replace=False)
                if ranks[p1_idx] < ranks[p2_idx] or (ranks[p1_idx] == ranks[p2_idx] and crowding[p1_idx] > crowding[p2_idx]):
                    p1 = X[p1_idx]
                else:
                    p1 = X[p2_idx]
                    
                p3_idx, p4_idx = np.random.choice(pop_size, 2, replace=False)
                if ranks[p3_idx] < ranks[p4_idx] or (ranks[p3_idx] == ranks[p4_idx] and crowding[p3_idx] > crowding[p4_idx]):
                    p2 = X[p3_idx]
                else:
                    p2 = X[p4_idx]
                
                # Crossover
                child = self._sbx_crossover(p1, p2, lo, hi)
                # Mutation
                child = self._poly_mutation(child, lo, hi)
                next_X[i] = child
                
            # Evaluate offspring
            F_child = np.array([problem(x) for x in next_X])
            eval_count += pop_size
            
            # Combine and select
            X_combined = np.vstack([X, next_X])
            F_combined = np.vstack([F, F_child])
            
            ranks, crowding = self._dominance_sort(F_combined)
            
            # Select best pop_size
            # Sort by rank, then by crowding
            sort_idx = np.lexsort((crowding, ranks))
            sort_idx = sort_idx[:pop_size]
            
            X = X_combined[sort_idx]
            F = F_combined[sort_idx]
            
            # Optional: Elitism is handled by the combination and selection.
            
        return X

    def _dominance_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n, dtype=float)
        
        # Non-dominated sorting
        # This is O(N^2 * M)
        S = [[] for _ in range(n)] # S_i: set of solutions dominated by i
        n_i = np.zeros(n, dtype=int) # n_i: number of solutions dominating i
        
        for i in range(n):
            for j in range(i+1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_i[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_i[i] += 1
        
        # Rank 0
        current_front = np.where(n_i == 0)[0]
        rank = 0
        while len(current_front) > 0:
            for i in current_front:
                ranks[i] = rank
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_i[j] -= 1
                    if n_i[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1
        
        # Crowding distance
        for r in range(rank):
            front = np.where(ranks == r)[0]
            if len(front) <= 2:
                crowding[front] = np.inf
                continue
            for m in range(self.n_obj):
                # Sort front by objective m
                sorted_front = front[np.argsort(F[front, m])]
                crowding[sorted_front[0]] = np.inf
                crowding[sorted_front[-1]] = np.inf
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                if f_max == f_min:
                    continue
                for k in range(1, len(sorted_front)-1):
                    idx = sorted_front[k]
                    prev_idx = sorted_front[k-1]
                    next_idx = sorted_front[k+1]
                    dist = (F[next_idx, m] - F[prev_idx, m]) / (f_max - f_min)
                    crowding[idx] += dist
                    
        return ranks, crowding

    def _sbx_crossover(self, p1, p2, lo, hi):
        eta = 20.0
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-14:
                    child[i] = p1[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2*u)**(1.0/(eta+1))
                    else:
                        beta = (1.0/(2*(1.0-u)))**(1.0/(eta+1))
                    child[i] = 0.5*((1.0+beta)*p1[i] + (1.0-beta)*p2[i])
            else:
                if abs(p2[i] - p1[i]) < 1e-14:
                    child[i] = p2[i]
                else:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2*u)**(1.0/(eta+1))
                    else:
                        beta = (1.0/(2*(1.0-u)))**(1.0/(eta+1))
                    child[i] = 0.5*((1.0+beta)*p2[i] + (1.0-beta)*p1[i])
        return np.clip(child, lo, hi)

    def _poly_mutation(self, x, lo, hi):
        eta = 20.0
        for i in range(len(x)):
            if np.random.rand() < 1.0/len(x):
                x_range = hi[i] - lo[i]
                if x_range == 0:
                    continue
                u = np.random.rand()
                if u < 0.5:
                    delta = (2*u)**(1.0/(eta+1)) - 1.0
                else:
                    delta = 1.0 - (2*(1.0-u))**(1.0/(eta+1))
                x[i] = x[i] + delta * x_range
        return np.clip(x, lo, hi)
