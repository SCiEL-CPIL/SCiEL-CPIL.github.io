import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        range_ = hi - lo
        
        # Phase 1: NSGA-II exploration
        pop_size = 15
        # WFG7-aware initialization
        X = np.empty((pop_size, self.dim))
        for i in range(pop_size):
            X[i, 0] = lo[0] + range_[0] * np.random.random()
            for d in range(1, self.dim):
                u = np.random.random()
                center = (lo[d] + hi[d]) / 2.0
                spread = range_[d] / 2.0
                if u < 0.5:
                    X[i, d] = lo[d] + spread * (2 * u) ** 0.7
                else:
                    X[i, d] = hi[d] - spread * (2 * (1 - u)) ** 0.7
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        eta_c = 15.0
        pc = 0.9
        
        # Exploration phase: use ~60% of budget
        explore_budget = int(self.budget * 0.6)
        gen = 0
        total_explore_gens = (explore_budget - used) // pop_size
        
        while used < explore_budget:
            if time.time() > deadline:
                break
            
            progress = gen / max(total_explore_gens, 1)
            sigma = 0.12 * (1.0 - 0.5 * progress) * range_
            
            Q = np.empty_like(X)
            for i in range(pop_size):
                p1 = self._tournament_select(X, F)
                p2 = self._tournament_select(X, F)
                
                if np.random.random() < pc:
                    c1, _ = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                else:
                    c1 = p1.copy()
                
                c1 = c1 + sigma * np.random.standard_normal(self.dim)
                c1 = np.clip(c1, lo, hi)
                Q[i] = c1
            
            F_q = np.array([problem(x) for x in Q])
            used += pop_size
            
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            X, F = self._nsga2_select(P, F_p, pop_size)
            gen += 1
        
        # Phase 2: Local polish - use remaining budget for fine-grained search
        # around the best hypervolume contributors
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            X, F = self._local_polish(problem, X, F, lo, hi, remaining, deadline)
            used = self.budget
        
        return X

    def _local_polish(self, problem, X, F, lo, hi, budget, deadline):
        """Polish the best solutions using local search."""
        # Find non-dominated front
        fronts = self._non_dominated_sort(F)
        if not fronts:
            return X, F
        
        # Get the first front (non-dominated solutions)
        front_idx = np.array(fronts[0])
        X_front = X[front_idx]
        F_front = F[front_idx]
        
        # Compute hypervolume contribution for each point
        # Simple approach: try to improve each point via local search
        n_front = len(X_front)
        if n_front == 0:
            return X, F
        
        # Distribute budget among front members
        evals_per_point = max(1, budget // n_front)
        
        X_new = X_front.copy()
        F_new = F_front.copy()
        used = 0
        
        for i in range(n_front):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Local search around X_new[i]
            best_x = X_new[i].copy()
            best_f = F_new[i].copy()
            
            for step in range(evals_per_point):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Adaptive step size
                step_size = 0.02 * (hi - lo) * (1.0 - step / max(evals_per_point, 1))
                perturbation = step_size * np.random.standard_normal(self.dim)
                x_trial = np.clip(best_x + perturbation, lo, hi)
                
                f_trial = np.array(problem(x_trial))
                used += 1
                
                # Accept if non-dominated or dominates current best
                if self._dominates(f_trial, best_f):
                    best_x = x_trial
                    best_f = f_trial
                elif not self._dominates(best_f, f_trial):
                    # Non-dominated: accept with some probability based on hypervolume gain
                    if np.random.random() < 0.3:
                        best_x = x_trial
                        best_f = f_trial
            
            X_new[i] = best_x
            F_new[i] = best_f
        
        # Combine polished front with original population
        P = np.vstack([X, X_new])
        F_p = np.vstack([F, F_new])
        
        # Select best pop_size solutions
        X_sel, F_sel = self._nsga2_select(P, F_p, len(X))
        return X_sel, F_sel

    def _tournament_select(self, X, F, k=3):
        n = len(F)
        idxs = np.random.choice(n, size=k, replace=False)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _nsga2_select(self, P, F_p, target_size):
        fronts = self._non_dominated_sort(F_p)
        
        next_X = []
        next_F = []
        
        for front in fronts:
            if len(next_X) + len(front) <= target_size:
                next_X.extend(P[front])
                next_F.extend(F_p[front])
            else:
                front_indices = np.array(front)
                F_front = F_p[front_indices]
                cd = self._crowding_distance(F_front)
                sorted_indices = np.argsort(-cd)
                
                needed = target_size - len(next_X)
                for idx in sorted_indices[:needed]:
                    global_idx = front_indices[idx]
                    next_X.append(P[global_idx])
                    next_F.append(F_p[global_idx])
                break
        
        if len(next_X) == 0:
            return P[:target_size], F_p[:target_size]
        
        return np.array(next_X), np.array(next_F)

    def _non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        n_dominating = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dominating[i] += 1
        
        current_front = [i for i in range(n) if n_dominating[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            sorted_idx = np.argsort(F[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            
            f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
            if f_range == 0:
                continue
            
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range
        
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    r = np.random.random()
                    if 2 * r <= (1 / (eta + 1)):
                        beta = (2 * r) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - r))) ** (1 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2
