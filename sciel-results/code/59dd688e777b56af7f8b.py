import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _fast_nondominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = [0] * n
        rank = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                rank[i] = 0
                if not fronts:
                    fronts.append([])
                fronts[0].append(i)
        
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            for i in fronts[front_idx]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return {i: np.inf for i in indices}
        
        cd = {i: 0.0 for i in indices}
        for m in range(self.n_obj):
            sorted_idx = sorted(indices, key=lambda i: F[i][m])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0]][m]
            f_max = F[sorted_idx[-1]][m]
            f_range = f_max - f_min
            if f_range == 0:
                continue
            for k in range(1, n - 1):
                i_curr = sorted_idx[k]
                i_prev = sorted_idx[k-1]
                i_next = sorted_idx[k+1]
                cd[i_curr] += (F[i_next][m] - F[i_prev][m]) / f_range
        return cd

    def __call__(self, problem, bounds):
        deadline = time.time() + 285.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        rng = np.random
        
        # Use a population size that balances exploration and exploitation
        pop_size = 30
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        evals = 0
        
        # 1. Initialization: Latin Hypercube Sampling for better coverage
        lhs = np.zeros((pop_size, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, pop_size + 1)
            pts = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(pts)
            lhs[:, d] = pts
        X = lo + lhs * span
        for j in range(pop_size):
            F[j] = problem(X[j])
            evals += 1
        
        # 2. Main loop: Differential Evolution + Non-dominated sorting + Crowding distance
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Sort into fronts
            fronts = self._fast_nondominated_sort(F)
            
            # Select parents from the first front
            if len(fronts) > 0 and len(fronts[0]) > 0:
                cd = self._crowding_distance(F, fronts[0])
                # Tournament selection: pick 2 random from first front, keep the one with higher CD
                i1, i2 = rng.randint(len(fronts[0])), rng.randint(len(fronts[0]))
                a, b = fronts[0][i1], fronts[0][i2]
                p1_idx = a if cd[a] > cd[b] else b
                
                # Second parent: also from first front
                i1, i2 = rng.randint(len(fronts[0])), rng.randint(len(fronts[0]))
                a, b = fronts[0][i1], fronts[0][i2]
                p2_idx = a if cd[a] > cd[b] else b
            else:
                p1_idx = rng.randint(pop_size)
                p2_idx = rng.randint(pop_size)
            
            # DE mutation: x_child = x_p1 + F * (x_p2 - x_p3)
            p3_idx = rng.randint(pop_size)
            F_factor = 0.5 + 0.5 * rng.random()  # Random between 0.5 and 1.0
            CR = 0.3
            
            child = X[p1_idx].copy()
            for d in range(dim):
                if rng.random() < CR or d == rng.randint(dim):
                    child[d] = X[p1_idx][d] + F_factor * (X[p2_idx][d] - X[p3_idx][d])
                else:
                    child[d] = X[p1_idx][d]
            
            # Clip
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Replacement: if child is non-dominated, replace the worst point in the last front
            dominated = False
            for j in range(pop_size):
                if self._dominates(F[j], fc):
                    dominated = True
                    break
            
            if not dominated:
                # Find the worst point to replace
                fronts = self._fast_nondominated_sort(F)
                last_front = fronts[-1]
                if len(last_front) > 1:
                    cd = self._crowding_distance(F, last_front)
                    worst_idx = min(last_front, key=lambda i: cd[i])
                else:
                    worst_idx = last_front[0]
                
                X[worst_idx] = child
                F[worst_idx] = fc
        
        # 3. Final polish: refine the non-dominated points with smaller steps
        fronts = self._fast_nondominated_sort(F)
        nd_indices = fronts[0]
        
        polish_budget = self.budget - evals
        if polish_budget > 0 and len(nd_indices) > 0:
            n_nd = len(nd_indices)
            evals_per_point = max(1, polish_budget // n_nd)
            
            for idx in nd_indices:
                if evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                best_child = X[idx].copy()
                best_child_F = F[idx].copy()
                
                for _ in range(evals_per_point):
                    if evals >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    # Small Gaussian step
                    sigma = 0.02 * span
                    child = X[idx] + sigma * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    evals += 1
                    
                    if self._dominates(fc, best_child_F):
                        best_child = child
                        best_child_F = fc
                
                X[idx] = best_child
                F[idx] = best_child_F
        
        return X
