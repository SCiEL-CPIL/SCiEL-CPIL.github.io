import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # State tracking
        X_all = []
        F_all = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            
            X_all.append(x)
            F_all.append(f)
            used += 1
            return f

        # 1. Initial Sampling: Latin Hypercube
        n_init = 30
        X_init = np.empty((n_init, dim))
        for i in range(dim):
            X_init[:, i] = lo[i] + span[i] * (np.random.rand(n_init) + np.arange(n_init)) / n_init
        idx = np.random.permutation(n_init)
        X_init = X_init[idx]
        
        for x in X_init:
            evaluate(x)
            if used >= self.budget or time.time() > deadline:
                break

        if not X_all:
            return X_all

        X_all = np.array(X_all)
        F_all = np.array(F_all)

        # 2. Evolutionary Search (NSGA-II like)
        pop_size = 20
        # Initialize population from initial samples
        nd_idx = self._get_non_dominated(F_all)
        
        if len(nd_idx) >= pop_size:
            X_pop = X_all[nd_idx[:pop_size]].copy()
            F_pop = F_all[nd_idx[:pop_size]].copy()
        else:
            X_pop = np.empty((pop_size, dim))
            F_pop = np.empty((pop_size, n_obj))
            X_pop[:len(nd_idx)] = X_all[nd_idx]
            F_pop[:len(nd_idx)] = F_all[nd_idx]
            remaining = pop_size - len(nd_idx)
            all_idx = np.arange(len(X_all))
            nd_set = set(nd_idx.tolist())
            available = [i for i in all_idx if i not in nd_set]
            if len(available) >= remaining:
                fill_idx = np.random.choice(available, remaining, replace=False)
            else:
                fill_idx = np.random.choice(len(X_all), remaining, replace=True)
            X_pop[len(nd_idx):] = X_all[fill_idx]
            F_pop[len(nd_idx):] = F_all[fill_idx]
            
        # Evolution loop
        while used < self.budget - 50 and time.time() < deadline: # Reserve budget for polish
            n_offspring = pop_size
            X_off = np.empty((n_offspring, dim))
            F_off = np.empty((n_offspring, n_obj))
            valid_offspring = 0
            
            for k in range(n_offspring):
                if used >= self.budget - 50 or time.time() > deadline:
                    break
                
                p1_idx, p2_idx = self._tournament_select(F_pop)
                child = self._crossover(X_pop[p1_idx], X_pop[p2_idx], lo, hi)
                child = self._mutate(child, lo, hi)
                
                f = evaluate(child)
                if f is None:
                    break
                X_off[k] = child
                F_off[k] = f
                valid_offspring += 1
            
            if valid_offspring == 0:
                break
                
            X_m = np.vstack([X_pop, X_off[:valid_offspring]])
            F_m = np.vstack([F_pop, F_off[:valid_offspring]])
            
            X_pop, F_pop = self._select_population(X_m, F_m, pop_size)

        # Update global lists with any new points from evolution
        # Note: evaluate() already added them to X_all, F_all
        # We just need to ensure X_all, F_all are arrays
        if isinstance(X_all, list):
            X_all = np.array(X_all)
            F_all = np.array(F_all)

        # 3. Local Polish
        # Select top K non-dominated points
        nd_idx = self._get_non_dominated(F_all)
        if len(nd_idx) > 0:
            # Sort non-dominated points by sum of objectives to pick "best" ones
            nd_points = F_all[nd_idx]
            sums = np.sum(nd_points, axis=1)
            best_nd_idx = nd_idx[np.argsort(sums)[:10]] # Polish top 10
            
            for idx in best_nd_idx:
                if used >= self.budget or time.time() > deadline:
                    break
                
                x0 = X_all[idx]
                f0 = F_all[idx]
                
                # Define scalarized objective for local search
                # Weights based on inverse of current objective values
                # Add small epsilon to avoid division by zero
                eps = 1e-6
                w = 1.0 / (f0 + eps)
                w = w / np.sum(w)
                
                def objective(x):
                    if time.time() > deadline:
                        return 0
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped))
                    return np.dot(w, f)
                
                # Run Nelder-Mead
                # maxiter is limited to save budget
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 5, 'maxfev': 5})
                
                # Record the result if it's better or different
                # The 'problem' call inside objective counts towards 'used'
                # We don't need to re-evaluate if res.success or just take res.x
                # But we need to record the point in X_all, F_all
                # The 'objective' function calls 'problem', which increments 'used'
                # and adds to X_all, F_all.
                # So the polish points are already recorded.
        
        return X_all

    def _tournament_select(self, F):
        n = len(F)
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        s1 = np.sum(F[idx1])
        s2 = np.sum(F[idx2])
        return (idx1, idx2) if s1 < s2 else (idx2, idx1)
        
    def _crossover(self, p1, p2, lo, hi):
        alpha = np.random.rand(self.dim)
        child = alpha * p1 + (1 - alpha) * p2
        return np.clip(child, lo, hi)
        
    def _mutate(self, x, lo, hi):
        span = hi - lo
        rate = 0.2
        mask = np.random.rand(self.dim) < rate
        if np.any(mask):
            x[mask] = x[mask] + 0.1 * span[mask] * np.random.randn(np.sum(mask))
        return np.clip(x, lo, hi)
        
    def _get_non_dominated(self, F):
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]
        
    def _select_population(self, X, F, pop_size):
        n = len(F)
        nd_idx = self._get_non_dominated(F)
        
        if len(nd_idx) >= pop_size:
            sums = np.sum(F[nd_idx], axis=1)
            keep = np.argsort(sums)[:pop_size]
            nd_idx = nd_idx[keep]
            return X[nd_idx].copy(), F[nd_idx].copy()
        else:
            dominated_idx = np.where(np.isin(np.arange(n), nd_idx, invert=True))[0]
            if len(dominated_idx) > 0:
                sums_d = np.sum(F[dominated_idx], axis=1)
                keep_d = np.argsort(sums_d)[:pop_size - len(nd_idx)]
                dominated_idx = dominated_idx[keep_d]
                nd_idx = np.concatenate([nd_idx, dominated_idx])
            return X[nd_idx].copy(), F[nd_idx].copy()
