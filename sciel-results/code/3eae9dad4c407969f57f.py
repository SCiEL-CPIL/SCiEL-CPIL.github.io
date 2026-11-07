import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial Population
        # With budget 200, let's try a population of 40 and 5 generations.
        # Or maybe 50 pop and 4 gens.
        # Let's stick to 40 pop.
        pop_size = 40
        if self.budget < pop_size:
            pop_size = max(10, self.budget)
            
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        # Helper functions for NSGA-II
        
        def fast_non_dominated_sort(F):
            M = len(F)
            S = [set() for _ in range(M)]  # Solutions dominated by i
            n = [0] * M  # Number of solutions dominating i
            rank = [0] * M
            
            for p in range(M):
                for q in range(M):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            
            I = [i for i in range(M) if n[i] == 0]
            i = 0
            while I:
                for p in I:
                    rank[p] = i
                Q = []
                for p in I:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            rank[q] = i + 1
                            Q.append(q)
                I = Q
                i += 1
            return rank
        
        def crowding_distance(F, rank):
            M = len(F)
            dist = np.zeros(M)
            for r in range(max(rank) + 1):
                idx = [i for i in range(M) if rank[i] == r]
                if len(idx) <= 2:
                    for i in idx:
                        dist[i] = np.inf
                    continue
                for m in range(self.n_obj):
                    # Sort by objective m
                    sorted_idx = sorted(idx, key=lambda i: F[i, m])
                    dist[sorted_idx[0]] = np.inf
                    dist[sorted_idx[-1]] = np.inf
                    f_min = F[sorted_idx[0], m]
                    f_max = F[sorted_idx[-1], m]
                    if f_max - f_min == 0:
                        continue
                    for j in range(1, len(sorted_idx) - 1):
                        i_prev = sorted_idx[j-1]
                        i_next = sorted_idx[j+1]
                        i_cur = sorted_idx[j]
                        dist[i_cur] += (F[i_next, m] - F[i_prev, m]) / (f_max - f_min)
            return dist
        
        def crossover(x1, x2):
            # SBX
            if np.random.rand() < 0.9:
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        beta = 1.0
                        if np.random.rand() <= 0.5:
                            beta = (2 * np.random.rand()) ** (1.0 / (2 + 1))
                        else:
                            beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (2 + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    else:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutation(x):
            # Polynomial Mutation
            eta_m = 20
            for i in range(self.dim):
                if np.random.rand() < 0.9:
                    delta1 = (2 * np.random.rand()) ** (1.0 / (eta_m + 1)) - 1
                    delta2 = 1 - (2 * (1 - np.random.rand())) ** (1.0 / (eta_m + 1))
                    x[i] += 0.5 * (hi[i] - lo[i]) * (delta1 + delta2)
            return np.clip(x, lo, hi)

        # Main Loop
        while evals_used < self.budget:
            if time.time() > deadline:
                break
                
            # Non-dominated sort
            rank = fast_non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
            # Selection (Tournament)
            def tournament(sel_x, sel_f, sel_rank, sel_dist):
                a = np.random.randint(len(sel_x))
                b = np.random.randint(len(sel_x))
                if sel_rank[a] < sel_rank[b]:
                    return a
                elif sel_rank[b] < sel_rank[a]:
                    return b
                else:
                    if sel_dist[a] > sel_dist[b]:
                        return a
                    else:
                        return b
            
            children_X = []
            children_F = []
            
            while len(children_X) < pop_size:
                p1 = tournament(X, F, rank, dist)
                p2 = tournament(X, F, rank, dist)
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                if evals_used + 1 > self.budget:
                    break
                f1 = problem(c1)
                evals_used += 1
                children_X.append(c1)
                children_F.append(f1)
                
                if evals_used + 1 > self.budget:
                    break
                f2 = problem(c2)
                evals_used += 1
                children_X.append(c2)
                children_F.append(f2)
                
                if time.time() > deadline:
                    break
            
            if not children_X:
                break
                
            # Combine
            X_all = np.vstack([X, np.array(children_X)])
            F_all = np.vstack([F, np.array(children_F)])
            
            # Selection for next generation
            rank_all = fast_non_dominated_sort(F_all)
            dist_all = crowding_distance(F_all, rank_all)
            
            next_X = []
            next_F = []
            # Sort by rank, then crowding distance
            indices = np.arange(len(X_all))
            indices_sorted = sorted(indices, key=lambda i: (rank_all[i], -dist_all[i]))
            
            for i in indices_sorted[:pop_size]:
                next_X.append(X_all[i])
                next_F.append(F_all[i])
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        # Polish: Refine non-dominated points
        # Find non-dominated points in final population
        rank_final = fast_non_dominated_sort(F)
        nd_indices = [i for i in range(len(F)) if rank_final[i] == 0]
        
        if nd_indices:
            # Try a few small random perturbations on these points
            # We have used self.budget evals, so we can't do more evaluations.
            # The task says "More than 200 evaluations ... makes the whole result invalid".
            # So we must stop at exactly <= budget.
            pass

        return X
