import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        N = 30
        if N > budget:
            N = budget
            
        # Initialize
        rng = np.random
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        while True:
            if evals >= budget:
                break
            if time.time() > deadline:
                break
                
            # 1. Non-dominated sort
            ranks, fronts = self._non_dominated_sort(F)
            
            # 2. Crowding distance
            cd = self._crowding_distance(F, fronts)
            
            # 3. Determine offspring size
            max_offspring = min(N, budget - evals)
            if max_offspring == 0:
                break
                
            # 4. Generate offspring
            new_X = []
            new_F = []
            
            # We can generate pairs (2 evals each) or singles
            # Let's try to fill up to max_offspring
            while len(new_X) < max_offspring and evals < budget:
                # Select two parents
                p1_idx = self._tournament_select(ranks, cd, N, 2, rng)
                p2_idx = self._tournament_select(ranks, cd, N, 2, rng)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX Crossover
                c1, c2 = self._sbx(p1, p2, lo, hi)
                
                # Polynomial Mutation
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                
                # Evaluate
                if evals + 1 <= budget:
                    f1 = problem(c1)
                    new_X.append(c1)
                    new_F.append(f1)
                    evals += 1
                else:
                    break
                    
                if evals + 1 <= budget and len(new_X) < max_offspring:
                    f2 = problem(c2)
                    new_X.append(c2)
                    new_F.append(f2)
                    evals += 1
                else:
                    break
                    
            if not new_X:
                break
                
            new_X = np.array(new_X)
            new_F = np.array(new_F)
            
            # 5. Merge and Select
            total_X = np.vstack([X, new_X])
            total_F = np.vstack([F, new_F])
            
            total_ranks, total_fronts = self._non_dominated_sort(total_F)
            total_cd = self._crowding_distance(total_F, total_fronts)
            
            # Select best N
            selected_idx = []
            for front in total_fronts:
                if len(selected_idx) + len(front) <= N:
                    selected_idx.extend(front)
                else:
                    sorted_front = sorted(front, key=lambda i: total_cd[i], reverse=True)
                    needed = N - len(selected_idx)
                    selected_idx.extend(sorted_front[:needed])
                    break
            
            X = total_X[selected_idx]
            F = total_F[selected_idx]
            
        return X

    def _tournament_select(self, ranks, cd, N, k, rng):
        idxs = rng.integers(N, size=k)
        best_idx = idxs[0]
        best_rank = ranks[best_idx]
        best_cd = cd[best_idx]
        for i in idxs[1:]:
            if ranks[i] < best_rank:
                best_idx = i
                best_rank = ranks[i]
                best_cd = cd[i]
            elif ranks[i] == best_rank and cd[i] > best_cd:
                best_idx = i
                best_cd = cd[i]
        return best_idx

    def _non_dominated_sort(self, F):
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        fronts = []
        remaining = list(range(N))
        
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            
            if not front:
                front = remaining
                
            for i in front:
                ranks[i] = len(fronts)
            fronts.append(front)
            remaining = [x for x in remaining if x not in front]
            
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        N = len(F)
        cd = np.zeros(N)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    cd[i] = float('inf')
                continue
            
            n_obj = self.n_obj
            for m in range(n_obj):
                sorted_front = sorted(front, key=lambda i: F[i, m])
                cd[sorted_front[0]] = float('inf')
                cd[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                range_m = f_max - f_min
                if range_m == 0:
                    continue
                    
                for k in range(1, len(sorted_front) - 1):
                    i_prev = sorted_front[k-1]
                    i_next = sorted_front[k+1]
                    cd[sorted_front[k]] += (F[i_next, m] - F[i_prev, m]) / range_m
                    
        return cd

    def _sbx(self, p1, p2, lo, hi):
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p1)
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (20 + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (20 + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
                
        return c1, c2

    def _poly_mutation(self, x, lo, hi):
        x_mut = x.copy()
        prob_mut = 1.0 / self.dim
        eta_m = 20
        
        for i in range(self.dim):
            if np.random.random() < prob_mut:
                if np.random.random() < 0.5:
                    y = x[i] - lo[i]
                    y = y / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                    delta1 = (1 + y) ** (eta_m + 1)
                    u = np.random.random()
                    delta = (2 * u + (1 - 2 * u) * delta1) ** (1.0 / (eta_m + 1)) - 1
                    y_new = y * (1 + delta)
                    x_mut[i] = lo[i] + y_new * (hi[i] - lo[i])
                else:
                    y = hi[i] - x[i]
                    y = y / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                    delta2 = (1 + y) ** (eta_m + 1)
                    u = np.random.random()
                    delta = (2 * u + (1 - 2 * u) * delta2) ** (1.0 / (eta_m + 1)) - 1
                    y_new = y * (1 + delta)
                    x_mut[i] = hi[i] - y_new * (hi[i] - lo[i])
                    
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
                
        return x_mut
