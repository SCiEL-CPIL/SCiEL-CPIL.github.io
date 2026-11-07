import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # Population size for NSGA-II
        # With budget 200, we can do about 10-20 generations if pop size is 10-20.
        # Let's try N=20. Init: 20. Each gen: 20 offspring + 20 parents = 40 evals? 
        # No, standard NSGA-II evaluates offspring. 
        # If we evaluate offspring only, and replace parents, we use N evals per gen.
        # Total evals = N (init) + G * N (offspring).
        # 200 = 20 + G*20 => 190 = G*20 => G=9.5. So 9 full generations + 10 extra.
        # Or N=10. 200 = 10 + G*10 => 190 = G*10 => G=19.
        # Smaller population might explore less, but more generations.
        # Let's use N=20 for better diversity in sorting.
        N = 20
        if N > budget:
            N = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, n))
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II parameters
        # Crossover probability
        pc = 0.9
        # Mutation probability
        pm = 1.0 / n
        # Gaussian mutation scale
        sigma = 0.5
        
        while used < budget:
            # If remaining budget is less than N, we cannot form a full offspring population
            # for standard selection, but we can still evaluate some and merge if we want, 
            # or just stop. Given 200 budget and N=20, we have exactly 9.5 generations worth.
            # Let's run full generations while used + N <= budget.
            if used + N > budget:
                break
            
            # Create offspring population
            X_off = np.zeros((N, n))
            F_off = np.zeros((N, m))
            
            for i in range(N):
                # Binary tournament selection
                p1_idx = self._tournament_select(F, N)
                p2_idx = self._tournament_select(F, N)
                
                # Simulated Binary Crossover (SBX)
                if np.random.random() < pc:
                    c1, c2 = self._sbx(X[p1_idx], X[p2_idx], lo, hi)
                else:
                    c1 = X[p1_idx].copy()
                    c2 = X[p2_idx].copy()
                
                # Polynomial Mutation
                c1 = self._mutate(c1, lo, hi, pm, sigma)
                c2 = self._mutate(c2, lo, hi, pm, sigma)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                
                used += 2
                X_off[2*i] = c1
                F_off[2*i] = f1
                X_off[2*i+1] = c2
                F_off[2*i+1] = f2
                
            # Combine parents and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sorting
            ranks, fronts = self._fast_non_dominated_sort(F_all)
            
            # Select next generation
            X_next = np.zeros((N, n))
            F_next = np.zeros((N, m))
            
            i = 0
            while i < len(fronts):
                front = fronts[i]
                if len(front) <= N - np.sum([len(f) for f in fronts[:i]]):
                    # Add entire front
                    for idx in front:
                        X_next[np.sum([len(f) for f in fronts[:i]]) + front.index(idx)] = X_all[idx]
                        F_next[np.sum([len(f) for f in fronts[:i]]) + front.index(idx)] = F_all[idx]
                    i += 1
                else:
                    # Partial front: use crowding distance
                    # Calculate crowding distance for this front
                    cd = self._crowding_distance(F_all[front])
                    # Sort by crowding distance descending
                    sorted_indices = np.argsort(-cd)
                    remaining_slots = N - np.sum([len(f) for f in fronts[:i]])
                    for j in range(remaining_slots):
                        orig_idx = front[sorted_indices[j]]
                        X_next[np.sum([len(f) for f in fronts[:i]]) + j] = X_all[orig_idx]
                        F_next[np.sum([len(f) for f in fronts[:i]]) + j] = F_all[orig_idx]
                    break
            
            X = X_next
            F = F_next
            
        return X

    def _tournament_select(self, F, N):
        idx1 = np.random.randint(N)
        idx2 = np.random.randint(N)
        if self._dominates(F[idx1], F[idx2]):
            return idx1
        elif self._dominates(F[idx2], F[idx1]):
            return idx2
        else:
            return np.random.choice([idx1, idx2])

    def _sbx(self, p1, p2, lo, hi):
        # Simple SBX implementation
        # eta = 20
        eta = 20
        u = np.random.random(n=len(p1))
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p1)
        for j in range(len(p1)):
            if np.random.random() < 0.5:
                if abs(p1[j] - p2[j]) < 1e-14:
                    c1[j] = p1[j]
                    c2[j] = p2[j]
                else:
                    if u[j] <= (1 + 1/eta) / 2:
                        beta = (2 * u[j]) ** (1/(eta+1))
                    else:
                        beta = (1 / (2 * (1 - u[j]))) ** (1/(eta+1))
                    c1[j] = 0.5 * (1 + beta) * p1[j] + 0.5 * (1 - beta) * p2[j]
                    c2[j] = 0.5 * (1 - beta) * p1[j] + 0.5 * (1 + beta) * p2[j]
            else:
                c1[j] = p2[j]
                c2[j] = p1[j]
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _mutate(self, x, lo, hi, pm, sigma):
        # Polynomial mutation
        # eta_m = 20
        eta_m = 20
        u = np.random.random(n=len(x))
        x_new = x.copy()
        for j in range(len(x)):
            if np.random.random() < pm:
                if u[j] < 0.5:
                    delta1 = (2 * u[j]) ** (1/(eta_m+1)) - 1
                else:
                    delta1 = 1 - (2 * (1 - u[j])) ** (1/(eta_m+1))
                x_new[j] = x[j] + delta1 * (hi[j] - lo[j])
        return np.clip(x_new, lo, hi)

    def _fast_non_dominated_sort(self, F):
        N = len(F)
        domination_count = np.zeros(N, dtype=int)
        dominated_list = [[] for _ in range(N)]
        fronts = [[]]
        
        for i in range(N):
            for j in range(i+1, N):
                if self._dominates(F[i], F[j]):
                    dominated_list[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    dominated_list[j].append(i)
                    domination_count[i] += 1
        
        for i in range(N):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for k in fronts[i]:
                for m in dominated_list[k]:
                    domination_count[m] -= 1
                    if domination_count[m] == 0:
                        next_front.append(m)
            if len(next_front) > 0:
                fronts.append(next_front)
            i += 1
            
        # Remove empty fronts at end if any
        while fronts and len(fronts[-1]) == 0:
            fronts.pop()
            
        return fronts

    def _crowding_distance(self, F):
        N = len(F)
        if N <= 2:
            return np.full(N, np.inf)
        
        cd = np.zeros(N)
        m = F.shape[1]
        
        for j in range(m):
            order = np.argsort(F[:, j])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if F[order[-1], j] - F[order[0], j] == 0:
                continue
            for i in range(1, N-1):
                cd[order[i]] += (F[order[i+1], j] - F[order[i-1], j]) / (F[order[-1], j] - F[order[0], j])
                
        return cd

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)
