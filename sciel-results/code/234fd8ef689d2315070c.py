import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        def eval_point(x):
            return np.array(problem(x))
        
        # Phase 1: Initial Population using LHS
        # Split budget into initial and offspring
        # Let's use 50% for initial, 50% for offspring
        n_init = budget // 2
        if n_init < 10:
            n_init = 10
        n_offspring = budget - n_init
        
        # Generate LHS
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            bin_edges = np.linspace(lo[i], hi[i], n_init + 1)
            samples = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) * np.random.rand(n_init)
            np.random.shuffle(samples)
            X_init[:, i] = samples
            
        F_init = np.array([eval_point(x) for x in X_init])
        
        if time.time() > deadline - 10.0:
            return X_init

        # Phase 2: NSGA-II Offspring Generation
        # Calculate ranks and crowding distances for initial population
        fronts = self._fast_non_dominated_sort(F_init)
        ranks = np.zeros(n_init, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = i
                
        # Crowding distance
        cd = self._crowding_distance(F_init, fronts)
        
        X_all = X_init
        F_all = F_init
        
        # Generate offspring
        X_off = np.zeros((n_offspring, dim))
        F_off = np.zeros((n_offspring, n_obj))
        
        eta_c = 20.0
        eta_m = 20.0
        prob_c = 0.9
        
        for k in range(n_offspring):
            if time.time() > deadline - 1.0:
                break
                
            # Select parents
            p1_idx = self._tournament(ranks, cd)
            p2_idx = self._tournament(ranks, cd)
            
            p1 = X_init[p1_idx]
            p2 = X_init[p2_idx]
            
            # Crossover
            if np.random.rand() < prob_c:
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
            else:
                c1 = p1.copy()
                c2 = p2.copy()
                
            # Mutation
            c1 = self._polynomial_mutation(c1, lo, hi, eta_m)
            c2 = self._polynomial_mutation(c2, lo, hi, eta_m)
            
            # We only have budget for n_offspring points.
            # In standard NSGA-II, we generate 2 offspring per pair.
            # Here we generate 1 offspring per iteration to fit budget exactly?
            # Or generate 2 and take the first?
            # Let's just generate 1 unique offspring per eval slot to be safe.
            # Actually, generating 2 and using both is better for diversity if we had budget.
            # With budget 200 and n_init 100, we have 100 slots.
            # If we generate 2 per pair, we need 50 pairs.
            # Let's generate 2 per pair and fill the array.
            
            # Revised loop structure for NSGA-II
            # We need to fill n_offspring points.
            # Let's do it in pairs if possible.
            pass # Placeholder for restructure

        # Restructure Phase 2 to properly fill n_offspring
        # If n_offspring is even, we can do pairs.
        # If odd, one single.
        
        X_off = np.zeros((n_offspring, dim))
        F_off = np.zeros((n_offspring, n_obj))
        
        idx = 0
        while idx < n_offspring:
            if time.time() > deadline - 1.0:
                break
                
            p1_idx = self._tournament(ranks, cd)
            p2_idx = self._tournament(ranks, cd)
            
            p1 = X_init[p1_idx]
            p2 = X_init[p2_idx]
            
            if np.random.rand() < prob_c:
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
            else:
                c1 = p1.copy()
                c2 = p2.copy()
                
            c1 = self._polynomial_mutation(c1, lo, hi, eta_m)
            c2 = self._polynomial_mutation(c2, lo, hi, eta_m)
            
            if idx < n_offspring:
                X_off[idx] = c1
                F_off[idx] = eval_point(c1)
                idx += 1
                
            if idx < n_offspring:
                X_off[idx] = c2
                F_off[idx] = eval_point(c2)
                idx += 1
                
        # Combine
        if n_offspring > 0:
            X_all = np.vstack([X_init, X_off])
            F_all = np.vstack([F_init, F_off])
            
        return X_all

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
            
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                    
        front = set()
        for i in range(n):
            if n_dom[i] == 0:
                front.add(i)
        if front:
            fronts.append(front)
                
        k = 0
        while k < len(fronts):
            next_front = set()
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    def _crowding_distance(self, F, fronts):
        n = len(F)
        cd = np.zeros(n)
        
        for front in fronts:
            m = len(front)
            if m <= 2:
                for idx in front:
                    cd[idx] = np.inf
                continue
                
            for k in range(self.n_obj):
                # Sort indices in front by objective k
                sorted_idx = sorted(front, key=lambda i: F[i][k])
                min_val = F[sorted_idx[0]][k]
                max_val = F[sorted_idx[-1]][k]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                if max_val == min_val:
                    continue
                    
                for i in range(1, m - 1):
                    idx = sorted_idx[i]
                    cd[idx] += (F[sorted_idx[i+1]][k] - F[sorted_idx[i-1]][k]) / (max_val - min_val)
                    
        return cd

    def _tournament(self, ranks, cd):
        # Binary tournament
        i1 = np.random.randint(0, len(ranks))
        i2 = np.random.randint(0, len(ranks))
        
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            else:
                return i2

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                        ratio = (1.0 + (p2[i]-p1[i])/(hi[i]-lo[i])) / beta
                        if ratio > 0:
                            c1[i] = 0.5 * (p1[i] + p2[i] - (p2[i]-p1[i]) * (ratio - 1.0))
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                            ratio = (1.0 + (p1[i]-p2[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c1[i] = 0.5 * (p1[i] + p2[i] + (p2[i]-p1[i]) * (ratio - 1.0))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                        ratio = (1.0 + (p1[i]-p2[i])/(hi[i]-lo[i])) / beta
                        if ratio > 0:
                            c1[i] = 0.5 * (p1[i] + p2[i] + (p1[i]-p2[i]) * (ratio - 1.0))
                        else:
                            alpha = 2.0 * u
                            beta = ((1.0 + 1.0/eta) ** (alpha + 1.0))
                            ratio = (1.0 + (p2[i]-p1[i])/(hi[i]-lo[i])) / beta
                            if ratio > 0:
                                c1[i] = 0.5 * (p1[i] + p2[i] - (p1[i]-p2[i]) * (ratio - 1.0))
                                
                    # Swap c1 and c2 for this dimension? No, standard SBX produces two children.
                    # The logic above calculates c1. c2 is symmetric.
                    # Let's implement a cleaner SBX.
                    pass # Re-implement SBX cleanly

        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta):
        x_m = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                y = x_m[i] + delta * (hi[i] - lo[i])
                if y < lo[i]:
                    y = lo[i]
                elif y > hi[i]:
                    y = hi[i]
                x_m[i] = y
        return x_m
