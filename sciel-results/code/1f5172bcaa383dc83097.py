"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with SBX crossover and Gaussian mutation for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Check if a dominates b (all objectives <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting. Returns list of fronts (indices)."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [set() for _ in range(n)]
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    dominated_set[i].add(j)
                    domination_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    dominated_set[j].add(i)
                    domination_count[i] += 1
        
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        # If any points remain unassigned, add them to the last front
        if len(fronts) < n:
            remaining = [i for i in range(n) if i not in [idx for front in fronts for idx in front]]
            if remaining:
                if fronts:
                    fronts[-1].extend(remaining)
                else:
                    fronts.append(remaining)
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Calculate crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_front[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
            if f_range > 1e-10:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (F_front[sorted_idx[i+1], m] - F_front[sorted_idx[i-1], m]) / f_range
        
        return distances

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=15.0, prob=0.9):
        """Simulated Binary Crossover."""
        if rng.random() > prob:
            return p1.copy(), p2.copy()
        
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if p1[i] == p2[i]:
                continue
            if rng.random() < 0.5:
                if p1[i] < p2[i]:
                    x1, x2 = p1[i], p2[i]
                    swap = False
                else:
                    x1, x2 = p2[i], p1[i]
                    swap = True
            else:
                continue
            
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            
            c1_val = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2_val = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            
            if swap:
                c1[i] = c2_val
                c2[i] = c1_val
            else:
                c1[i] = c1_val
                c2[i] = c2_val
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _gaussian_mutation(self, x, lo, hi, rng, sigma=0.1, prob=0.1):
        """Gaussian mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if rng.random() < prob:
                x_mut[i] += sigma * (hi[i] - lo[i]) * rng.standard_normal()
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def _wfg7_aware_init(self, N, lo, hi, rng):
        """WFG7-aware initialization."""
        X = np.zeros((N, self.dim))
        for i in range(N):
            X[i, :3] = lo[:3] + (hi[:3] - lo[:3]) * rng.random(3)
            for j in range(3, self.dim):
                if rng.random() < 0.5:
                    X[i, j] = (lo[j] + hi[j]) / 2 + (hi[j] - lo[j]) * 0.3 * rng.standard_normal()
                else:
                    X[i, j] = lo[j] + (hi[j] - lo[j]) * rng.random()
            X[i] = np.clip(X[i], lo, hi)
        return X

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 40  # Population size
        rng = np.random.default_rng(0)
        
        # Initialize population
        X = self._wfg7_aware_init(N, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Tournament selection for mating pool
            offspring_X = []
            offspring_F = []
            
            for _ in range(N):
                if evals >= self.budget or time.time() > deadline:
                    break
                
                # Tournament selection (size 3)
                t1 = rng.integers(N)
                t2 = rng.integers(N)
                t3 = rng.integers(N)
                
                # Find non-dominated rank for each
                fronts = self._non_dominated_sort(F)
                ranks = np.zeros(N, dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = rank
                
                # Select best by rank, then crowding
                best_t = min([t1, t2, t3], key=lambda i: (ranks[i],))
                
                # Second parent
                t4 = rng.integers(N)
                t5 = rng.integers(N)
                t6 = rng.integers(N)
                best_t2 = min([t4, t5, t6], key=lambda i: (ranks[i],))
                
                # SBX crossover
                c1, c2 = self._sbx_crossover(X[best_t], X[best_t2], lo, hi, rng)
                
                # Gaussian mutation
                c1 = self._gaussian_mutation(c1, lo, hi, rng)
                c2 = self._gaussian_mutation(c2, lo, hi, rng)
                
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                evals += 1
                
                if evals < self.budget and time.time() <= deadline:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    evals += 1
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X[:N])
            offspring_F = np.array(offspring_F[:N])
            
            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(F_combined)
            
            # Select next generation
            X_next = []
            F_next = []
            
            for front in fronts:
                if len(X_next) + len(front) <= N:
                    for idx in front:
                        X_next.append(X_combined[idx])
                        F_next.append(F_combined[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = N - len(X_next)
                    cd = self._crowding_distance(F_combined, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        X_next.append(X_combined[idx])
                        F_next.append(F_combined[idx])
                    break
            
            if len(X_next) < N:
                # Fill remaining with best points
                for i in range(len(X_next), N):
                    X_next.append(X_combined[i % len(X_combined)])
                    F_next.append(F_combined[i % len(F_combined)])
            
            X = np.array(X_next[:N])
            F = np.array(F_next[:N])
        
        return X
# EVOLVE-BLOCK-END
