"""Rewritten optimizer for WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Increased to 50 for better Pareto front density
        N = 50
        if N > self.budget:
            N = self.budget
            
        # Initialize population using Latin Hypercube Sampling for better coverage
        def lhs(n, dim):
            points = np.zeros((n, dim))
            for d in range(dim):
                intervals = np.arange(n) / n
                offsets = np.random.random(n)
                points[:, d] = intervals + offsets
                np.random.shuffle(points[:, d])
            return points

        X_norm = lhs(N, self.dim)
        X = lo + (hi - lo) * X_norm
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            front = [set() for _ in range(n)]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_doms[p] += 1
                if n_doms[p] == 0:
                    ranks[p] = 0
                    front[0].add(p)
                    
            i = 0
            while front[i]:
                next_front = set()
                for p in front[i]:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.add(q)
                i += 1
                if next_front:
                    front[i] = next_front
                else:
                    break
            return [list(f) for f in front if f]

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = len(F)
            n_obj = self.n_obj
            crowd = np.zeros(m)
            
            for k in range(n_obj):
                sorted_idx = sorted(front_indices, key=lambda i: F[i][k])
                crowd[sorted_idx[0]] = float('inf')
                crowd[sorted_idx[-1]] = float('inf')
                f_min = F[sorted_idx[0]][k]
                f_max = F[sorted_idx[-1]][k]
                if f_max == f_min:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    crowd[idx] += (F[sorted_idx[i+1]][k] - F[sorted_idx[i-1]][k]) / (f_max - f_min)
            
            return {i: crowd[i] for i in front_indices}

        # Helper: Binary Tournament
        def tournament_select(pop, ranks, crowds):
            i = np.random.randint(len(pop))
            j = np.random.randint(len(pop))
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                if crowds[i] > crowds[j]:
                    return i
                else:
                    return j

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutate(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    continue
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                x_mut[i] += delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Sort current population
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Calculate crowding distance for the last front that might be partially selected
            # For simplicity, calculate for all fronts or just the last one needed?
            # Standard NSGA-II calculates for the front where selection cuts off.
            # We can calculate for all points in the population for simplicity, or just the last front.
            # Let's calculate for the last front.
            # Actually, tournament selection needs crowding distance for all points in the same rank.
            # So we calculate crowding distance for all points.
            crowds = np.zeros(len(F))
            for front in fronts:
                cd_dict = crowding_distance(F, front)
                for idx, cd in cd_dict.items():
                    crowds[idx] = cd
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We need to generate enough offspring to fill the next population
            # We can generate N offspring per generation.
            # If we have remaining budget < N, we generate only that many.
            num_offspring = min(N, self.budget - used)
            
            for _ in range(num_offspring):
                p1_idx = tournament_select(X, ranks, crowds)
                p2_idx = tournament_select(X, ranks, crowds)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                # If we used all budget, break
                if used >= self.budget:
                    break
                
                # If we have room for the second child, add it
                if used < self.budget:
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Select next population
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= N:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    # Sort this front by crowding distance
                    cd_dict = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    needed = N - len(X_new)
                    for idx in sorted_front[:needed]:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            
        return F
# EVOLVE-BLOCK-END
