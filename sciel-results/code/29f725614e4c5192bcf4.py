import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        pop_size = 50
        # Budget is 200. Init takes 50. 3 generations of 50 offspring = 150. Total 200.
        max_generations = (self.budget - pop_size) // pop_size
        
        # 1. Initialization
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            front = [0]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # Check if p dominates q
                    # p dominates q if p_i <= q_i for all i and p_j < q_j for some j
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                        
                if n_dom[p] == 0:
                    rank[p] = 0
                    front[0].append(p)
                    
            i = 0
            while len(front[i]) > 0:
                next_front = []
                for p in front[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                front.append(next_front)
                
            return rank, front

        # Helper: Crowding distance
        def crowding_distance(F, rank, front_indices):
            n = len(F)
            dist = [0.0] * n
            m = len(front_indices)
            if m <= 2:
                for i in front_indices:
                    dist[i] = float('inf')
                return dist
            
            for k in range(n_obj):
                # Sort front by objective k
                sorted_indices = sorted(front_indices, key=lambda idx: F[idx][k])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                if f_max == f_min:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_indices[i]] += (F[sorted_indices[i+1]][k] - F[sorted_indices[i-1]][k]) / (f_max - f_min)
                    
            return dist

        # Helper: Tournament selection
        def tournament_select(ranks, dist, n_select=2):
            best = None
            best_rank = float('inf')
            best_dist = -float('inf')
            
            for _ in range(n_select):
                idx = np.random.randint(0, len(ranks))
                r = ranks[idx]
                d = dist[idx]
                
                if best is None:
                    best = idx
                    best_rank = r
                    best_dist = d
                else:
                    if r < best_rank:
                        best = idx
                        best_rank = r
                        best_dist = d
                    elif r == best_rank and d > best_dist:
                        best = idx
                        best_rank = r
                        best_dist = d
                        
            return best

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            x = np.empty_like(x1)
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if 2.0 * np.random.rand() <= (x1[i] - x2[i]) / (lo[i] - hi[i]) + 1.0:
                            beta = 1.0 + 2.0 * (x1[i] - lo[i]) / (x1[i] - x2[i])
                            alpha = 2.0 - beta ** (eta + 1.0)
                            rand = np.random.rand()
                            if rand <= 1.0 / alpha:
                                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                            else:
                                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                            x[i] = 0.5 * ((x1[i] + x2[i]) - betaq * (x1[i] - x2[i]))
                        else:
                            beta = 1.0 + 2.0 * (hi[i] - x2[i]) / (x2[i] - x1[i])
                            alpha = 2.0 - beta ** (eta + 1.0)
                            rand = np.random.rand()
                            if rand <= 1.0 / alpha:
                                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                            else:
                                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                            x[i] = 0.5 * ((x1[i] + x2[i]) + betaq * (x2[i] - x1[i]))
                    else:
                        x[i] = x1[i]
                else:
                    x[i] = x1[i]
            return x

        # Helper: Polynomial Mutation
        def poly_mutation(x, lo, hi, eta=20.0):
            for i in range(len(x)):
                if np.random.rand() < 0.5:
                    if abs(hi[i] - lo[i]) > 1e-14:
                        y = x[i]
                        delta1 = (y - lo[i]) / (hi[i] - lo[i])
                        delta2 = (hi[i] - y) / (hi[i] - lo[i])
                        rand = np.random.rand()
                        if rand < 0.5:
                            xy = 1.0 - delta1
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                            deltaq = (val ** (1.0 / (eta + 1.0))) - 1.0
                        else:
                            xy = 1.0 - delta2
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                            deltaq = 1.0 - (val ** (1.0 / (eta + 1.0)))
                        y = y + deltaq * (hi[i] - lo[i])
                        if y < lo[i]:
                            y = lo[i]
                        elif y > hi[i]:
                            y = hi[i]
                        x[i] = y
            return x

        # Main Loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
                
            # 1. Environmental selection on current population to get ranks and distances
            ranks, fronts = non_dominated_sort(F)
            dist = [0.0] * pop_size
            for front in fronts:
                if len(front) == 0:
                    continue
                d = crowding_distance(F, ranks, front)
                for idx in front:
                    dist[idx] = d[idx]
            
            # 2. Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size:
                # Selection
                idx1 = tournament_select(ranks, dist)
                idx2 = tournament_select(ranks, dist)
                
                # Crossover
                child1 = sbx_crossover(X[idx1], X[idx2], lo, hi)
                child2 = sbx_crossover(X[idx1], X[idx2], lo, hi)
                
                # Mutation
                child1 = poly_mutation(child1, lo, hi)
                child2 = poly_mutation(child2, lo, hi)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                evaluations += 2
                
                offspring_X.append(child1)
                offspring_F.append(f1)
                offspring_X.append(child2)
                offspring_F.append(f2)
                
                if evaluations >= self.budget:
                    break
            
            # Trim offspring if we exceeded budget (shouldn't happen with exact division, but safe)
            offspring_X = offspring_X[:pop_size]
            offspring_F = offspring_F[:pop_size]
            
            # 3. Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # 4. Non-dominated sort combined
            c_ranks, c_fronts = non_dominated_sort(combined_F)
            
            # 5. Select next population
            next_X = []
            next_F = []
            next_ranks = [0] * (2 * pop_size)
            next_dist = [0.0] * (2 * pop_size)
            
            for front in c_fronts:
                if len(next_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_ranks[idx] = c_ranks[idx]
                else:
                    # Need to fill remaining spots from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    # Calculate crowding distance for this front
                    d = crowding_distance(combined_F, c_ranks, front)
                    # Sort by distance descending
                    sorted_front = sorted(front, key=lambda idx: d[idx], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_ranks[idx] = c_ranks[idx]
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
