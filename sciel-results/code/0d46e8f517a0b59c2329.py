import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        n = len(F)
        m = F.shape[1]
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = list(range(n))
        
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
                front = [remaining[0]]
            fronts.append(front)
            for i in front:
                ranks[i] = len(fronts) - 1
            remaining = [i for i in remaining if i not in front]
        
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        n = len(F)
        m = F.shape[1]
        dist = np.zeros(n)
        
        for f in fronts:
            if len(f) == 0:
                continue
            if len(f) == 1:
                dist[f[0]] = np.inf
                continue
            
            for k in range(m):
                sorted_f = sorted(f, key=lambda i: F[i][k])
                dist[sorted_f[0]] = np.inf
                dist[sorted_f[-1]] = np.inf
                min_val = F[sorted_f[0]][k]
                max_val = F[sorted_f[-1]][k]
                range_val = max_val - min_val
                if range_val < 1e-14:
                    continue
                for i in range(1, len(sorted_f) - 1):
                    idx = sorted_f[i]
                    dist[idx] += (F[sorted_f[i + 1]][k] - F[sorted_f[i - 1]][k]) / range_val
        
        return dist

    def _sbx_crossover(self, x1, x2, lo, hi, eta):
        child1 = np.copy(x1)
        child2 = np.copy(x2)
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                continue
            
            x1_i = x1[i]
            x2_i = x2[i]
            
            if abs(x2_i - x1_i) < 1e-14:
                continue
                
            if np.random.random() < 0.5:
                beta = 1.0 + (2.0 * (x1_i - lo[i]) / (x2_i - x1_i))
                alpha = 2.0 - beta ** (eta + 1.0)
                betaq = ((beta + 1.0) / alpha) ** (1.0 / (eta + 1.0))
                child1[i] = 0.5 * (x1_i + x2_i - betaq * (x2_i - x1_i))
            else:
                beta = 1.0 + (2.0 * (hi[i] - x2_i) / (x2_i - x1_i))
                alpha = 2.0 - beta ** (eta + 1.0)
                betaq = ((beta + 1.0) / alpha) ** (1.0 / (eta + 1.0))
                child2[i] = 0.5 * (x1_i + x2_i + betaq * (x2_i - x1_i))
                
        return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, eta):
        mutated = np.copy(x)
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                continue
                
            x_i = x[i]
            range_i = hi[i] - lo[i]
            
            if range_i < 1e-14:
                continue
                
            if np.random.random() < 0.5:
                delta1 = (x_i - lo[i]) / range_i
                xq = 1.0 - (1.0 - delta1) ** (eta + 1.0)
            else:
                delta2 = (hi[i] - x_i) / range_i
                xq = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                
            yq = 1.0 + 2.0 * (np.random.random() - 0.5)
            
            if abs(yq - xq) < 1e-14:
                continue
                
            delta = (yq - xq) / (1.0 + 2.0 * (1.0 - yq)) ** (1.0 / (eta + 1.0))
            
            if delta > 0:
                delta_q = delta ** (1.0 / (eta + 1.0))
            else:
                delta_q = -((-delta) ** (1.0 / (eta + 1.0)))
                
            mutated[i] = x_i + delta_q * range_i
            
        return np.clip(mutated, lo, hi)

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        N = 40
        
        # Initialize population using LHS
        X = np.empty((N, n))
        for j in range(n):
            perm = np.random.permutation(N)
            X[:, j] = lo[j] + (hi[j] - lo[j]) * (perm + np.random.random(N)) / N
        
        F = np.array([problem(x) for x in X])
        used = N
        
        eta_c = 20.0
        eta_m = 20.0
        pm_base = 1.0 / n
        
        gen = 0
        total_gens = max(1, (self.budget - used) // N)
        
        while used < self.budget:
            gen += 1
            progress = gen / max(1, total_gens)
            pm = pm_base * (1.0 + 0.5 * np.sin(np.pi * progress))
            
            Y = np.empty_like(X)
            F_Y = np.empty_like(F)
            num_offspring = 0
            
            while num_offspring < N and used < self.budget:
                p1, p2 = np.random.choice(N, size=2, replace=False)
                
                child1, child2 = self._sbx_crossover(X[p1], X[p2], lo, hi, eta_c)
                
                if np.random.random() < pm:
                    child1 = self._polynomial_mutation(child1, lo, hi, eta_m)
                if np.random.random() < pm:
                    child2 = self._polynomial_mutation(child2, lo, hi, eta_m)
                
                Y[num_offspring] = child1
                F_Y[num_offspring] = problem(child1)
                used += 1
                num_offspring += 1
                
                if num_offspring < N and used < self.budget:
                    Y[num_offspring] = child2
                    F_Y[num_offspring] = problem(child2)
                    used += 1
                    num_offspring += 1
            
            R = np.vstack([X, Y[:num_offspring]])
            F_R = np.vstack([F, F_Y[:num_offspring]])
            
            ranks, fronts = self._non_dominated_sort(F_R)
            dist = self._crowding_distance(F_R, fronts)
            
            order = np.lexsort((-dist, ranks))
            X = R[order[:N]]
            F = F_R[order[:N]]
            
            # Local search on top 3 individuals
            if used < self.budget:
                top_indices = order[:3]
                for idx in top_indices:
                    if used >= self.budget:
                        break
                    x_best = X[idx]
                    f_best = F[idx]
                    
                    for _ in range(2):
                        if used >= self.budget:
                            break
                        x_trial = x_best + np.random.normal(0, 0.02, n)
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = problem(x_trial)
                        used += 1
                        
                        dominated = False
                        for f_other in F:
                            if np.all(f_other <= f_trial) and np.any(f_other < f_trial):
                                dominated = True
                                break
                                
                        if not dominated:
                            X[idx] = x_trial
                            F[idx] = f_trial
                            x_best = x_trial
                            f_best = f_trial

        return X
