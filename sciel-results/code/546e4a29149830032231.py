import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Population size: 20 allows 10 generations with 200 budget
        pop_size = 20
        
        # WFG7-aware initialization:
        # WFG7 has a specific structure where:
        # - x[0] controls position on the Pareto front
        # - x[1:] are shape variables that should be near specific values
        # For WFG7, the Pareto front is achieved when shape variables are near 
        # the midpoint of their ranges, with some specific patterns.
        X = np.empty((pop_size, self.dim))
        
        for i in range(pop_size):
            # Position variable: uniform over range
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * rng.random()
            
            for d in range(1, self.dim):
                center = (lo[d] + hi[d]) / 2.0
                spread = (hi[d] - lo[d]) / 2.0
                
                # WFG7 shape variables: concentrate near center with exponential decay
                # Use a mixture: 70% near center, 30% more spread
                if rng.random() < 0.7:
                    # Concentrated near center
                    offset = rng.standard_normal() * spread * 0.15
                else:
                    # More spread for exploration
                    offset = rng.standard_normal() * spread * 0.5
                
                X[i, d] = np.clip(center + offset, lo[d], hi[d])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Decomposition weights for MOEA/D-style guidance
        # Generate weight vectors for 3 objectives
        n_weights = pop_size
        weights = self._generate_weights(n_weights, self.n_obj, rng)
        
        # Parameters
        eta_c = 15.0
        pc = 0.9
        base_sigma = 0.2  # Initial mutation sigma
        
        gen = 0
        total_gens = (self.budget - used) // pop_size
        
        # Track best solutions for local search
        best_archive = []
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive mutation strength: start high, decrease over generations
            progress = gen / max(total_gens, 1)
            # Use a smoother decay: start at base_sigma, end at 0.05*base_sigma
            sigma_scale = base_sigma * (0.05 + 0.95 * (1.0 - progress) ** 1.5)
            sigma = sigma_scale * (hi - lo)
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                # Weight-based selection: pick parent closest to weight vector i
                w = weights[i]
                # Normalize F for weight computation
                F_min = F.min(axis=0)
                F_max = F.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                F_norm = (F - F_min) / F_range
                
                # Compute Tchebycheff distance to weight
                dists = np.max(w * F_norm, axis=1)
                parent_idx = np.argmin(dists)
                p1 = X[parent_idx].copy()
                
                # Second parent: tournament selection
                p2 = self._tournament_select(X, F, rng)
                
                # Crossover (SBX)
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Adaptive Gaussian mutation with per-dimension scaling
                # Shape variables (d > 0) get smaller mutations than position variable
                for d in range(self.dim):
                    if d == 0:
                        # Position variable: larger mutation
                        c1[d] += sigma[d] * rng.standard_normal()
                        c2[d] += sigma[d] * rng.standard_normal()
                    else:
                        # Shape variables: smaller mutation, centered around current value
                        mut_scale = sigma[d] * 0.3
                        c1[d] += mut_scale * rng.standard_normal()
                        c2[d] += mut_scale * rng.standard_normal()
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                # Use both offspring if budget allows, otherwise just c1
                if used + len(offspring) + 2 <= self.budget:
                    offspring.append(c1)
                    offspring.append(c2)
                else:
                    offspring.append(c1)
                
                if used + len(offspring) >= self.budget:
                    break
            
            if not offspring:
                break
            
            Q = np.array(offspring)
            F_q = np.array([problem(x) for x in Q])
            used += len(Q)
            
            # Combine parent and offspring
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            # Select next generation using NSGA-II
            X, F = self._nsga2_select(P, F_p, pop_size, rng)
            
            # Update archive with non-dominated solutions
            self._update_archive(best_archive, X, F)
            
            gen += 1
        
        # Local search phase: refine the best solutions
        if used < self.budget and time.time() < deadline - 30:
            X, F, used = self._local_search(problem, bounds, X, F, used, deadline, rng, lo, hi)
        
        return X

    def _generate_weights(self, n, n_obj, rng):
        """Generate weight vectors for decomposition."""
        if n_obj == 1:
            return np.ones((n, 1))
        
        weights = np.empty((n, n_obj))
        
        # Use structured weight vectors for better coverage
        if n_obj == 3:
            # Generate points on the 2D simplex (triangle)
            # Use a grid-like pattern for better distribution
            n_per_axis = int(np.ceil(n ** 0.5))
            idx = 0
            for i in range(n_per_axis):
                for j in range(n_per_axis):
                    if idx >= n:
                        break
                    # Map to simplex: w1 = i/(n-1), w2 = j/(n-1), w3 = 1 - w1 - w2
                    w1 = i / max(n_per_axis - 1, 1)
                    w2 = j / max(n_per_axis - 1, 1)
                    w3 = 1.0 - w1 - w2
                    if w3 >= -1e-10:
                        weights[idx] = [max(w1, 0), max(w2, 0), max(w3, 0)]
                        # Normalize
                        s = weights[idx].sum()
                        if s > 0:
                            weights[idx] /= s
                        else:
                            weights[idx] = [1/3, 1/3, 1/3]
                        idx += 1
                    if idx >= n:
                        break
                if idx >= n:
                    break
            
            # Fill remaining with random Dirichlet
            while idx < n:
                w = rng.dirichlet([1.0, 1.0, 1.0])
                weights[idx] = w
                idx += 1
        else:
            for i in range(n):
                w = rng.dirichlet(np.ones(n_obj))
                weights[i] = w
        
        return weights

    def _update_archive(self, archive, X, F):
        """Update the archive with non-dominated solutions."""
        for i in range(len(X)):
            is_dominated = False
            for j in range(len(archive)):
                if self._dominates(archive[1][j], F[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                # Check if this solution dominates any in archive
                new_archive = []
                for j in range(len(archive)):
                    if not self._dominates(F[i], archive[1][j]):
                        new_archive.append((archive[0][j], archive[1][j]))
                new_archive.append((X[i], F[i]))
                archive[:] = new_archive

    def _local_search(self, problem, bounds, X, F, used, deadline, rng, lo, hi):
        """Perform local search on the best solutions to refine the Pareto front."""
        # Select top solutions based on hypervolume contribution
        # Use a simple approach: take the non-dominated front
        _, fronts = self._non_dominated_sort(F)
        
        # Take solutions from the first front
        if fronts:
            front1 = fronts[0]
            ls_candidates = [X[i] for i in front1]
        else:
            ls_candidates = [X[0]]
        
        # Limit local search to a few candidates
        max_ls = min(len(ls_candidates), 5)
        ls_candidates = ls_candidates[:max_ls]
        
        # Local search parameters
        ls_steps = 30  # Total steps for local search
        steps_per_candidate = max(1, ls_steps // max_ls)
        
        for candidate in ls_candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            best_x = candidate.copy()
            best_f = problem(best_x)
            
            for step in range(steps_per_candidate):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try small perturbations
                step_size = 0.05 * (hi - lo)
                perturbation = step_size * rng.standard_normal(self.dim)
                
                # Reduce step size over iterations
                perturbation *= (1.0 - step / steps_per_candidate * 0.7)
                
                new_x = np.clip(best_x + perturbation, lo, hi)
                new_f = problem(new_x)
                used += 1
                
                # Accept if it dominates or is non-dominated with best
                if self._dominates(new_f, best_f):
                    best_x = new_x
                    best_f = new_f
                elif not self._dominates(best_f, new_f):
                    # Non-dominated: keep with some probability
                    if rng.random() < 0.3:
                        best_x = new_x
                        best_f = new_f
            
            # Update the population with the improved solution
            for i in range(len(X)):
                if np.array_equal(X[i], candidate):
                    X[i] = best_x
                    F[i] = best_f
                    break
        
        return X, F, used

    def _tournament_select(self, X, F, rng, k=3):
        idxs = rng.integers(0, len(F), size=k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _nsga2_select(self, P, F_p, target_size, rng):
        ranks, fronts = self._non_dominated_sort(F_p)
        
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
        
        return np.array(next_X), np.array(next_F)

    def _non_dominated_sort(self, F):
        n = len(F)
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
        
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
        
        return k, fronts

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

    def _sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    r = rng.random()
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
