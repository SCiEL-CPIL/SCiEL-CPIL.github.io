import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # Population size
        pop_size = 50
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Initialize
        # For ZDT6, x_2..x_n small is good for f_1.
        # x_1 should be spread.
        X = np.random.rand(pop_size, self.dim)
        # Bias x_2..x_n towards 0
        X[:, 1:] = np.random.rand(pop_size, self.dim - 1) ** 4
        # x_1 uniform
        X[:, 0] = np.random.rand(pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        tournament_size = 2
        crossover_rate = 0.9
        mutation_rate = 1.0 / self.dim
        eta_c = 15.0
        eta_m = 20.0
        
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[j] += 1
                        
            current_rank = 0
            front = [i for i in range(n) if n_dom[i] == 0]
            if not front:
                return [list(range(n))]
                
            ranks = []
            while front:
                ranks.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                front = next_front
            return ranks

        def crowding_distance(F, rank):
            n = len(F)
            dist = np.zeros(n)
            if n == 0:
                return dist
            if n == 1:
                return np.ones(n)
            
            for m in range(self.n_obj):
                obj = F[:, m]
                min_val = np.min(obj)
                max_val = np.max(obj)
                if max_val == min_val:
                    continue
                # Sort by this objective
                sorted_idx = np.argsort(obj)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (obj[sorted_idx[i+1]] - obj[sorted_idx[i-1]]) / (max_val - min_val)
            return dist

        while used < self.budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            offspring_F = []
            
            # Create children until we have pop_size
            while len(offspring) < pop_size:
                # Select parents
                p1_idx, p2_idx = -1, -1
                for _ in range(2):
                    tournament = np.random.randint(0, len(X), tournament_size)
                    # Compare ranks
                    ranks = fast_non_dominated_sort(F)
                    rank_map = {}
                    for r_idx, front in enumerate(ranks):
                        for idx in front:
                            rank_map[idx] = r_idx
                    
                    best = min(tournament, key=lambda i: rank_map[i])
                    if p1_idx == -1:
                        p1_idx = best
                    else:
                        p2_idx = best
                
                # Crossover
                if np.random.rand() < crossover_rate:
                    # SBX Crossover
                    c1 = X[p1_idx].copy()
                    c2 = X[p2_idx].copy()
                    for k in range(self.dim):
                        if np.random.rand() < 0.5:
                            if X[p1_idx][k] != X[p2_idx][k]:
                                if np.random.rand() < 0.5:
                                    beta = 1.0 / (1.0 + 2.0 * (X[p1_idx][k] - X[p2_idx][k]))
                                    if beta <= 0: beta = 1e-6
                                    alpha = 2.0 - beta ** (eta_c + 1.0)
                                    c1[k] = 0.5 * ((1.0 + alpha) * X[p1_idx][k] + (1.0 - alpha) * X[p2_idx][k])
                                    c2[k] = 0.5 * ((1.0 - alpha) * X[p1_idx][k] + (1.0 + alpha) * X[p2_idx][k])
                else:
                    c1 = X[p1_idx].copy()
                    c2 = X[p2_idx].copy()
                
                # Mutation
                for k in range(self.dim):
                    if np.random.rand() < mutation_rate:
                        delta = np.random.rand()
                        if delta < 0.5:
                            beta = (2.0 * delta) ** (1.0 / (eta_m + 1.0))
                        else:
                            beta = (2.0 * (1.0 - delta)) ** (1.0 / (eta_m + 1.0))
                        c1[k] = c1[k] + (beta - 1.0) * (hi[k] - lo[k])
                        c2[k] = c2[k] + (beta - 1.0) * (hi[k] - lo[k])
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                offspring.append(c1)
                offspring.append(c2)
                
                # Evaluate
                if used < self.budget:
                    f1 = problem(c1)
                    offspring_F.append(f1)
                    used += 1
                if used < self.budget:
                    f2 = problem(c2)
                    offspring_F.append(f2)
                    used += 1
                else:
                    # If we hit budget, we might not have enough offspring
                    # We break and handle the rest
                    break
            
            if used >= self.budget:
                break
                
            # Combine parent and offspring
            X_all = np.vstack([X, np.array(offspring)])
            F_all = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort
            ranks = fast_non_dominated_sort(F_all)
            rank_map = {}
            for r_idx, front in enumerate(ranks):
                for idx in front:
                    rank_map[idx] = r_idx
            
            # Crowding distance for each front
            crowd = np.zeros(len(F_all))
            for front in ranks:
                if len(front) > 0:
                    F_front = F_all[front]
                    d = crowding_distance(F_front, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = d[i]
            
            # Select next population
            X_next = []
            F_next = []
            for front in ranks:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                else:
                    # Sort by crowding distance
                    front_crowd = [(crowd[idx], idx) for idx in front]
                    front_crowd.sort(key=lambda x: -x[0])
                    needed = pop_size - len(X_next)
                    for i in range(needed):
                        idx = front_crowd[i][1]
                        X_next.append(X_all[idx])
                        F_next.append(F_all[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)

        return X
