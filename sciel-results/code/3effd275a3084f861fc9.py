"""Multi-objective optimizer for WFG4 using fast NSGA-II with vectorized ops."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast NSGA-II with vectorized dominance for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Vectorized fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points dominating i
        fronts = []
        
        # Vectorized dominance check
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for points with given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            f_vals = F[indices, k]
            f_min = np.min(f_vals)
            f_max = np.max(f_vals)
            f_range = f_max - f_min
            if f_range < 1e-12:
                continue
            
            order = np.argsort(f_vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            for i in range(1, n - 1):
                dist[order[i]] += (f_vals[order[i + 1]] - f_vals[order[i - 1]]) / f_range
        
        return dist

    def _sbx_crossover(self, x1, x2, lo, hi, rng):
        """Simulated binary crossover."""
        if rng.random() < 0.5:
            return x1.copy(), x2.copy()
        
        x3 = x1.copy()
        x4 = x2.copy()
        beta = 0.5
        
        for d in range(len(x1)):
            if rng.random() < 0.5:
                continue
            
            if abs(x1[d] - x2[d]) < 1e-12:
                x3[d] = x1[d]
                x4[d] = x2[d]
                continue
            
            if x1[d] < x2[d]:
                u = rng.random()
                if u <= 1.0 / (2 * beta + 1):
                    betaq = (2 * u * beta) ** (1.0 / (beta + 1))
                elif u >= 1.0 - 1.0 / (2 * beta + 1):
                    betaq = (2 * (1 - u) * beta) ** (1.0 / (beta + 1))
                else:
                    betaq = 1.0
                
                x3[d] = 0.5 * ((1 + betaq) * x1[d] + (1 - betaq) * x2[d])
                x4[d] = 0.5 * ((1 - betaq) * x1[d] + (1 + betaq) * x2[d])
            else:
                u = rng.random()
                if u <= 1.0 / (2 * beta + 1):
                    betaq = (2 * (1 - u) * beta) ** (1.0 / (beta + 1))
                elif u >= 1.0 - 1.0 / (2 * beta + 1):
                    betaq = (2 * u * beta) ** (1.0 / (beta + 1))
                else:
                    betaq = 1.0
                
                x3[d] = 0.5 * ((1 + betaq) * x1[d] + (1 - betaq) * x2[d])
                x4[d] = 0.5 * ((1 - betaq) * x1[d] + (1 + betaq) * x2[d])
        
        x3 = np.clip(x3, lo, hi)
        x4 = np.clip(x4, lo, hi)
        return x3, x4

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0):
        """Polynomial mutation."""
        x_mut = x.copy()
        for d in range(len(x)):
            if rng.random() < 0.5:
                continue
            
            u = rng.random()
            if u < 0.5:
                val = (2 * u) ** (1.0 / (eta + 1))
                delta = (1 - val) * (hi[d] - lo[d])
            else:
                val = (2 * (1 - u)) ** (1.0 / (eta + 1))
                delta = (1 - val) * (hi[d] - lo[d])
            
            x_mut[d] = x[d] + delta
            x_mut[d] = np.clip(x_mut[d], lo[d], hi[d])
        
        return x_mut

    def _tournament_select(self, ranks, cd, n_sel, rng):
        """Binary tournament selection based on rank then crowding distance."""
        n = len(ranks)
        selected = np.empty(n_sel, dtype=int)
        
        for i in range(n_sel):
            i1 = rng.randint(0, n)
            i2 = rng.randint(0, n)
            
            if ranks[i1] < ranks[i2]:
                selected[i] = i1
            elif ranks[i2] < ranks[i1]:
                selected[i] = i2
            else:
                if cd[i1] > cd[i2]:
                    selected[i] = i1
                else:
                    selected[i] = i2
        
        return selected

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget
        used = 0

        # Population size
        pop_size = 15

        # Initialize population
        pop_X = np.empty((pop_size, dim))
        pop_F = np.empty((pop_size, self.n_obj))
        
        for i in range(pop_size):
            if used >= budget or time.time() > deadline:
                break
            pop_X[i] = lo + (hi - lo) * rng.random_sample(dim)
            pop_F[i] = np.array(problem(pop_X[i]), dtype=np.float64)
            used += 1
        
        # NSGA-II main loop
        while used < budget and time.time() < deadline:
            # Evaluate if needed (pop_F should already be evaluated)
            
            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(pop_F)
            
            # Compute ranks
            ranks = np.zeros(pop_size, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Compute crowding distance
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    front_cd = self._crowding_distance(pop_F, front)
                    for j, idx in enumerate(front):
                        cd[idx] = front_cd[j]
            
            # Selection
            n_sel = pop_size
            selected = self._tournament_select(ranks, cd, n_sel, rng)
            
            # Generate offspring
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, self.n_obj))
            
            for i in range(0, pop_size, 2):
                if used >= budget or time.time() > deadline:
                    break
                
                p1_idx = selected[i % n_sel]
                p2_idx = selected[(i + 1) % n_sel]
                
                x1 = pop_X[p1_idx].copy()
                x2 = pop_X[p2_idx].copy()
                
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi, rng)
                c1 = self._polynomial_mutation(c1, lo, hi, rng)
                c2 = self._polynomial_mutation(c2, lo, hi, rng)
                
                offspring_X[i] = c1
                offspring_F[i] = np.array(problem(c1), dtype=np.float64)
                used += 1
                
                if i + 1 < pop_size and used < budget and time.time() < deadline:
                    offspring_X[i + 1] = c2
                    offspring_F[i + 1] = np.array(problem(c2), dtype=np.float64)
                    used += 1
                else:
                    break
            
            # Combine parent and offspring
            combined_X = np.vstack([pop_X, offspring_X[:used - pop_size] if used > pop_size else np.empty((0, dim))])
            combined_F = np.vstack([pop_F, offspring_F[:used - pop_size] if used > pop_size else np.empty((0, self.n_obj))])
            
            n_combined = len(combined_X)
            
            # Fast non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Compute ranks
            combined_ranks = np.zeros(n_combined, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    combined_ranks[idx] = rank
            
            # Compute crowding distance
            combined_cd = np.zeros(n_combined)
            for front in fronts:
                if len(front) > 0:
                    front_cd = self._crowding_distance(combined_F, front)
                    for j, idx in enumerate(front):
                        combined_cd[idx] = front_cd[j]
            
            # Select next population
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    # Sort by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda x: combined_cd[x], reverse=True)
                    need = pop_size - len(next_indices)
                    next_indices.extend(front_sorted[:need])
                    break
            
            # Update population
            pop_X = combined_X[next_indices]
            pop_F = combined_F[next_indices]
        
        # Return all non-dominated solutions found
        # Collect all evaluated points
        # We only have current pop, so let's also track archive
        # For simplicity, return non-dominated from current pop
        fronts = self._fast_non_dominated_sort(pop_F)
        if fronts and len(fronts[0]) > 0:
            return pop_X[fronts[0]]
        return pop_X[:1]
# EVOLVE-BLOCK-END
