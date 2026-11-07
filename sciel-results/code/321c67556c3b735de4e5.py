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
        
        # Population size
        N = 20
        if N > budget:
            N = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, n))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Parameters
        eta_c = 15.0
        eta_m = 20.0
        pc = 0.9
        pm = 1.0 / n
        
        while used < budget:
            # Check if we can afford a full generation of offspring
            if used + N > budget:
                # If not, we might do a partial generation or just stop.
                # To maximize evals, we can generate one at a time if budget allows.
                # But NSGA-II needs a full population to sort properly.
                # If remaining < N, we can't form a valid 2N pool for selection.
                # We can just do random search for the remaining budget.
                for _ in range(budget - used):
                    x_rand = lo + (hi - lo) * np.random.random(n)
                    problem(x_rand)
                break
                
            # Create offspring
            X_off = np.empty((N, n))
            F_off = np.empty((N, m))
            
            idx = 0
            while idx < N:
                # Calculate ranks and crowding for current population
                ranks, crowding = self._non_dominated_sort_and_crowding(X, F)
                
                # Tournament selection for parent 1
                p1_idx = self._tournament_select(ranks, crowding, N)
                # Tournament selection for parent 2
                p2_idx = self._tournament_select(ranks, crowding, N)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                if np.random.random() < pc:
                    c1, c2 = self._sbx_crossover(x1, x2, lo, hi, eta_c)
                else:
                    c1 = x1.copy()
                    c2 = x2.copy()
                
                # Mutation
                c1 = self._poly_mutation(c1, lo, hi, eta_m, pm)
                c2 = self._poly_mutation(c2, lo, hi, eta_m, pm)
                
                # Evaluate first child
                f1 = problem(c1)
                used += 1
                X_off[idx] = c1
                F_off[idx] = f1
                idx += 1
                
                if idx < N:
                    # Evaluate second child
                    f2 = problem(c2)
                    used += 1
                    X_off[idx] = c2
                    F_off[idx] = f2
                    idx += 1

            # Combine parents and offspring
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            # Select next generation
            ranks, crowding = self._non_dominated_sort_and_crowding(X_comb, F_comb)
            # Sort indices by rank (ascending), then by crowding distance (descending)
            # lexsort sorts by last key first. So (-crowding, ranks) means 
            # primary: ranks (asc), secondary: -crowding (asc) -> crowding (desc)
            sort_idx = np.lexsort(((-crowding), ranks))
            X = X_comb[sort_idx[:N]]
            F = F_comb[sort_idx[:N]]
            
        return X

    def _dominates(self, a, b):
        # a dominates b if a <= b in all objectives and a < b in at least one
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort_and_crowding(self, X, F):
        N = len(F)
        M = F.shape[1]
        ranks = np.zeros(N, dtype=int)
        crowding = np.zeros(N)
        
        # Fast non-dominated sorting
        S = [set() for _ in range(N)]
        n_dom = np.zeros(N, dtype=int)
        front = []
        
        for p in range(N):
            for q in range(N):
                if p == q: continue
                if self._dominates(F[p], F[q]):
                    S[p].add(q)
                elif self._dominates(F[q], F[p]):
                    n_dom[p] += 1
        
        for p in range(N):
            if n_dom[p] == 0:
                front.append(p)
                ranks[p] = 0
                
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            front = next_front
            
        # Calculate crowding distance for each front
        # Group by rank
        for rank_val in np.unique(ranks):
            members = np.where(ranks == rank_val)[0]
            if len(members) <= 2:
                crowding[members] = float('inf')
                continue
                
            for m_obj in range(M):
                # Sort members by objective m_obj
                obj_vals = F[members, m_obj]
                sorted_idx = np.argsort(obj_vals)
                members_sorted = members[sorted_idx]
                
                # Assign inf to extremes
                crowding[members_sorted[0]] = float('inf')
                crowding[members_sorted[-1]] = float('inf')
                
                if obj_vals.max() == obj_vals.min():
                    continue
                    
                for k in range(1, len(members_sorted) - 1):
                    prev_val = F[members_sorted[k-1], m_obj]
                    next_val = F[members_sorted[k+1], m_obj]
                    diff = next_val - prev_val
                    if diff > 0:
                        crowding[members_sorted[k]] += (F[members_sorted[k], m_obj] - prev_val) / diff
                        
        return ranks, crowding

    def _tournament_select(self, ranks, crowding, N, k=2):
        candidates = np.random.choice(N, size=k, replace=False)
        best_idx = candidates[0]
        best_rank = ranks[best_idx]
        best_crowd = crowding[best_idx]
        
        for c in candidates[1:]:
            r = ranks[c]
            cd = crowding[c]
            if r < best_rank or (r == best_rank and cd > best_crowd):
                best_idx = c
                best_rank = r
                best_crowd = cd
                
        return best_idx

    def _sbx_crossover(self, x1, x2, lo, hi, eta_c):
        c1 = x1.copy()
        c2 = x2.copy()
        n = len(x1)
        
        for i in range(n):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if np.random.random() < 0.5:
                        beta = 1.0 + 2.0 * (x1[i] - x2[i]) / (hi[i] - lo[i])
                    else:
                        beta = 1.0 + 2.0 * (x2[i] - x1[i]) / (hi[i] - lo[i])
                    
                    if beta > 0:
                        alpha = 2.0 - beta ** (eta_c + 1.0)
                        beta_q = (0.5 * (1.0 - np.sign(alpha) * (alpha - 1.0) ** (1.0 / (eta_c + 1.0))))
                    else:
                        alpha = 2.0 - (2.0 * beta) ** (eta_c + 1.0)
                        beta_q = (0.5 * (1.0 + np.sign(alpha) * (alpha - 1.0) ** (1.0 / (eta_c + 1.0))))
                    
                    c1[i] = 0.5 * ((1.0 + beta_q) * x1[i] + (1.0 - beta_q) * x2[i])
                    c2[i] = 0.5 * ((1.0 - beta_q) * x1[i] + (1.0 + beta_q) * x2[i])
                    
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _poly_mutation(self, x, lo, hi, eta_m, pm):
        x = x.copy()
        n = len(x)
        for i in range(n):
            if np.random.random() < pm:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                
                r = np.random.random()
                if r < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * r + (1.0 - 2.0 * r) * (xy) ** (eta_m + 1.0)
                    delta_q = val ** (1.0 / (eta_m + 1.0))
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy) ** (eta_m + 1.0)
                    delta_q = val ** (1.0 / (eta_m + 1.0))
                    
                x[i] = x[i] + (hi[i] - lo[i]) * delta_q
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
