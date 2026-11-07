import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        # Population size
        pop_size = 20
        
        if budget < pop_size:
            # Just random sampling
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X], dtype=np.float64)
            return X

        # 1. Initialization with Latin Hypercube Sampling
        X = np.empty((pop_size, dim))
        for d in range(dim):
            # LHS: divide range into pop_size bins, pick one random point in each
            bins = np.linspace(lo[d], hi[d], pop_size + 1)
            X[:, d] = np.random.uniform(bins[:-1], bins[1:])
        
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Main Evolutionary Loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Generate Offspring
            offspring_X = np.empty((pop_size, dim))
            for i in range(pop_size):
                # Tournament selection for parents
                p1 = self._tournament_select(F, 2)
                p2 = self._tournament_select(F, 2)
                
                x1, x2 = X[p1], X[p2]
                # SBX Crossover
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi, 0.9, 20)
                # Polynomial Mutation
                c1 = self._polynomial_mutation(c1, lo, hi, 1.0/dim, 20)
                c2 = self._polynomial_mutation(c2, lo, hi, 1.0/dim, 20)
                
                offspring_X[i*2] = c1
                if i*2 + 1 < pop_size:
                    offspring_X[i*2 + 1] = c2
                else:
                    # Handle odd population size if necessary, but 20 is even
                    pass

            # Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine and Select
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sort
            ranks, fronts = self._fast_non_dominated_sort(F_all)
            
            # Crowding Distance
            cd = self._crowding_distance(F_all, fronts)
            
            # Selection: Sort by rank, then by crowding distance (descending)
            sort_keys = np.lexsort((-cd, ranks))
            next_idx = sort_keys[:pop_size]
            
            X = X_all[next_idx]
            F = F_all[next_idx]

        # 2. Local Refinement Phase
        # Use remaining budget to refine the best non-dominated points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 10:
            # Get non-dominated points
            ranks, fronts = self._fast_non_dominated_sort(F)
            non_dom_idx = np.where(ranks == 0)[0]
            
            # Sort non-dominated points by crowding distance to pick diverse ones
            cd = self._crowding_distance(F, fronts)
            non_dom_idx = non_dom_idx[np.argsort(-cd[non_dom_idx])]
            
            # Allocate local search budget to top K non-dominated points
            K = min(len(non_dom_idx), 5) # Refine top 5
            if K > 0:
                budget_per_point = remaining // K
                for i in range(K):
                    if budget_per_point <= 0:
                        break
                    idx = non_dom_idx[i]
                    x_best = X[idx].copy()
                    f_best = F[idx].copy()
                    
                    # Simple Random Hill Climbing
                    for step in range(budget_per_point):
                        if time.time() > deadline:
                            break
                        # Perturb
                        scale = 0.1 * (hi - lo)
                        direction = np.random.randn(dim)
                        x_trial = x_best + 0.05 * scale * direction
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        f_trial = np.array(problem(x_trial), dtype=np.float64)
                        used += 1
                        
                        # Check if trial is better (dominates or is non-dominated and closer to origin?)
                        # For local search, we want to minimize objectives.
                        # If trial is non-dominated with current best, or dominates it, accept.
                        # Simple heuristic: Accept if trial is non-dominated with the current population's front?
                        # To keep it simple, accept if it improves the "sum" of objectives or is non-dominated.
                        if not self._dominates(f_best, f_trial):
                            # If f_best does not dominate f_trial, f_trial is a candidate.
                            # Check if f_trial dominates f_best?
                            if self._dominates(f_trial, f_best):
                                x_best = x_trial
                                f_best = f_trial
                            else:
                                # Both are non-dominated. We can keep the one with lower sum or just keep current.
                                # To encourage exploration, we might accept if sum is lower.
                                if np.sum(f_trial) < np.sum(f_best):
                                    x_best = x_trial
                                    f_best = f_trial
                    
                    X[idx] = x_best
                    F[idx] = f_best

        return X

    def _fast_non_dominated_sort(self, F):
        n = F.shape[0]
        if n == 0:
            return np.array([], dtype=int), []
        
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = F[i], F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1

        front = []
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                front.append(i)

        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = len(fronts)
                        next_front.append(j)
            front = next_front

        return ranks, fronts

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, fronts):
        n = F.shape[0]
        n_obj = F.shape[1]
        cd = np.zeros(n)

        for front in fronts:
            m = len(front)
            if m <= 2:
                for i in front:
                    cd[i] = np.inf
                continue

            for k in range(n_obj):
                sorted_idx = sorted(front, key=lambda idx: F[idx, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                if m > 2:
                    f_min = F[sorted_idx[0], k]
                    f_max = F[sorted_idx[-1], k]
                    f_range = f_max - f_min

                    if f_range < 1e-14:
                        continue

                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / f_range

        return cd

    def _tournament_select(self, F, k=2):
        n = F.shape[0]
        candidates = np.random.choice(n, k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if self._dominates(F[c], F[best]):
                best = c
            elif not self._dominates(F[best], F[c]):
                # If non-dominated, we can pick randomly or based on a metric.
                # For simplicity, keep best or pick randomly.
                if np.random.random() < 0.5:
                    best = c
        return best

    def _sbx_crossover(self, x1, x2, lo, hi, p_c=0.9, eta_c=20):
        if np.random.random() > p_c:
            return x1.copy(), x2.copy()

        c1, c2 = x1.copy(), x2.copy()
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    x_min = min(x1[i], x2[i])
                    x_max = max(x1[i], x2[i])
                    diff = x_max - x_min

                    u = np.random.random()
                    beta = 1.0 + 2.0 * (x1[i] - x_min) / diff
                    alpha = 2.0 - beta ** (-eta_c - 1.0)
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    c1[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))

                    u = np.random.random()
                    beta = 1.0 + 2.0 * (x2[i] - x_min) / diff
                    alpha = 2.0 - beta ** (-eta_c - 1.0)
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    c2[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))

        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, p_m, eta_m):
        x_mut = x.copy()
        for i in range(len(x)):
            if np.random.random() < p_m:
                x_min = lo[i]
                x_max = hi[i]
                if x_max - x_min <= 1e-14:
                    continue

                u = np.random.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta_q = (x[i] - x_min) * delta
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    delta_q = (x_max - x[i]) * delta

                x_mut[i] = x[i] + delta_q
                x_mut[i] = np.clip(x_mut[i], x_min, x_max)
        return x_mut
