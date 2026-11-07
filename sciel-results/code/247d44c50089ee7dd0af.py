import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 40
        if pop_size > self.budget:
            pop_size = self.budget
            
        eval_count = 0
        def evaluate(x):
            nonlocal eval_count
            eval_count += 1
            return np.array(problem(x))

        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([evaluate(x) for x in X])
        
        n_crossover = 20.0
        n_mutation = self.dim + 1.0
        p_mutation = 1.0 / self.dim
        p_crossover = 0.9

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            front = []
            fronts = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    front.append(i)
            
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                front = next_front
            
            return fronts, rank

        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [np.inf] * n
            
            dist = np.zeros(n)
            for m in range(self.n_obj):
                vals = np.array([F[i][m] for i in front_indices])
                min_val = np.min(vals)
                max_val = np.max(vals)
                
                if max_val == min_val:
                    dist[:] = np.inf
                    continue
                
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / (max_val - min_val)
            
            return dist

        def crossover(x1, x2):
            if np.random.rand() > p_crossover:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for j in range(self.dim):
                if np.random.rand() < 0.5:
                    if x1[j] != x2[j]:
                        if x1[j] < x2[j]:
                            xl, xh = x1[j], x2[j]
                        else:
                            xl, xh = x2[j], x1[j]
                        
                        range_val = xh - xl
                        if range_val < 1e-10:
                            continue
                            
                        u = np.random.rand()
                        beta = 1.0
                        if u <= 1.0 / (n_crossover + 1.0):
                            beta = (2.0 * u) ** (1.0 / (n_crossover + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (n_crossover + 1.0))
                        
                        c1[j] = 0.5 * ((1.0 + beta) * x1[j] + (1.0 - beta) * x2[j])
                        c2[j] = 0.5 * ((1.0 - beta) * x1[j] + (1.0 + beta) * x2[j])
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            return c1, c2

        def mutation(x):
            x_mut = x.copy()
            for j in range(self.dim):
                if np.random.rand() < p_mutation:
                    delta1 = (x_mut[j] - lo[j]) / (hi[j] - lo[j])
                    delta2 = (hi[j] - x_mut[j]) / (hi[j] - lo[j])
                    
                    u = np.random.rand()
                    if u < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (n_mutation + 1.0))
                        deltaq = val ** (1.0 / (n_mutation + 1.0)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (n_mutation + 1.0))
                        deltaq = 1.0 - val ** (1.0 / (n_mutation + 1.0))
                    
                    x_mut[j] += deltaq * (hi[j] - lo[j])
            
            return np.clip(x_mut, lo, hi)

        while eval_count < self.budget:
            fronts, rank = fast_non_dominated_sort(F)
            
            crowd_dist = np.zeros(pop_size)
            for front in fronts:
                if not front:
                    continue
                dists = crowding_distance(F, front)
                for idx, d in zip(front, dists):
                    crowd_dist[idx] = d
            
            def tournament_select():
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                
                if rank[i1] < rank[i2]:
                    return i1
                elif rank[i2] < rank[i1]:
                    return i2
                else:
                    if crowd_dist[i1] > crowd_dist[i2]:
                        return i1
                    else:
                        return i2
            
            temp_X = []
            temp_F = []
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                p1_idx = tournament_select()
                p2_idx = tournament_select()
                
                if p1_idx == p2_idx and pop_size > 1:
                    p2_idx = (p2_idx + 1) % pop_size
                
                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                
                c1 = mutation(c1)
                c2 = mutation(c2)
                
                if eval_count < self.budget:
                    f_c1 = evaluate(c1)
                    temp_X.append(c1)
                    temp_F.append(f_c1)
                
                if eval_count < self.budget:
                    f_c2 = evaluate(c2)
                    temp_X.append(c2)
                    temp_F.append(f_c2)
            
            if not temp_X:
                break
            
            X_off = np.array(temp_X)
            F_off = np.array(temp_F)
            
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            n_comb = len(F_comb)
            fronts_comb, rank_comb = fast_non_dominated_sort(F_comb)
            
            crowd_comb = np.zeros(n_comb)
            for front in fronts_comb:
                if not front:
                    continue
                dists = crowding_distance(F_comb, front)
                for idx, d in zip(front, dists):
                    crowd_comb[idx] = d
            
            X_next = []
            F_next = []
            for front in fronts_comb:
                if len(X_next) + len(front) <= pop_size:
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    sorted_front = sorted(front, key=lambda i: crowd_comb[i], reverse=True)
                    needed = pop_size - len(X_next)
                    for idx in sorted_front[:needed]:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)
            
            if eval_count >= self.budget:
                break

        return X
