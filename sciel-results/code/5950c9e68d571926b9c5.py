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
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        
        # Population size: 24 gives ~8 generations with 200 budget
        pop_size = 24
        
        # Initialize with WFG7-aware sampling:
        # For WFG7, x[0] controls position on Pareto front (uniform),
        # x[1:] are shape variables. For WFG7, optimal shape vars are near
        # specific values - typically the Pareto front is achieved when
        # shape variables are at their optimal positions.
        X = np.empty((pop_size, self.dim))
        for i in range(pop_size):
            # x[0]: uniform across the range to cover the Pareto front
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * rng.random()
            for d in range(1, self.dim):
                # Shape variables: bias toward the midpoint with narrow spread
                # WFG7 optimal shape variables are often near the center
                spread = half_range[d] * 0.25
                X[i, d] = mid[d] + spread * rng.standard_normal()
                X[i, d] = np.clip(X[i, d], lo[d], hi[d])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Generate well-distributed weight vectors
        n_weights = pop_size
        weights = self._generate_weights(n_weights, self.n_obj)
        
        # Parameters
        eta_c = 15.0
        eta_m = 15.0
        pc = 0.95
        pm = 1.0 / self.dim
        
        gen = 0
        
        while used < self.budget and time.time() < deadline:
            # Compute Tchebycheff distances for parent assignment
            dists = self._compute_tcheb_dists(F, weights)
            parent_assignments = np.argmin(dists, axis=0)
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                if used + len(offspring) + 2 > self.budget:
                    if used + len(offspring) + 1 > self.budget:
                        break
                    p1 = X[parent_assignments[i]].copy()
                    p2 = self._tournament_select(X, F, rng)
                    if rng.random() < pc:
                        c1, _ = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                    else:
                        c1 = p1.copy()
                    c1 = self._poly_mutate(c1, lo, hi, eta_m, pm, rng)
                    offspring.append(c1)
                    break
                
                p1 = X[parent_assignments[i]].copy()
                p2 = self._tournament_select(X, F, rng)
                
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                c1 = self._poly_mutate(c1, lo, hi, eta_m, pm, rng)
                c2 = self._poly_mutate(c2, lo, hi, eta_m, pm, rng)
                
                offspring.append(c1)
                offspring.append(c2)
            
            if not offspring:
                break
            
            Q = np.array(offspring)
            F_q = np.array([problem(x) for x in Q])
            used += len(Q)
            
            # Combine and select
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            X, F = self._nsga2_select(P, F_p, pop_size, rng)
            gen += 1
        
        # Final local refinement: try small perturbations on best solutions
        if used < self.budget and time.time() < deadline:
            X, F = self._local_refine(problem, X, F, lo, hi, rng, 
                                      remaining=self.budget - used, deadline=deadline)
        
        return X
    
    def _compute_tcheb_dists(self, F, weights):
        """Compute Tchebycheff distances."""
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range
        dists = np.max(weights[None, :, :] * F_norm[:, None, :], axis=2)
        return dists
    
    def _generate_weights(self, n, n_obj):
        """Generate well-distributed weight vectors on the simplex."""
        if n_obj == 1:
            return np.ones((n, 1))
        
        weights = []
        if n_obj == 3:
            # Use a stratified grid on the 2D simplex
            n_per_edge = int(np.ceil(np.sqrt(2 * n / 3)))
            count = 0
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    if count >= n:
                        break
                    k = n_per_edge - i - j
                    total = n_per_edge
                    if total == 0:
                        continue
                    w = np.array([i, j, k], dtype=np.float64) / total
                    weights.append(w)
                    count += 1
                if count >= n:
                    break
            
            # If we have fewer than n, add more points
            while len(weights) < n:
                # Add points along edges and center
                t = rng = np.random.default_rng(len(weights))
                w = t.dirichlet([1.0, 1.0, 1.0])
                weights.append(w)
        else:
            for i in range(n):
                r = np.random.default_rng(i)
                weights.append(r.dirichlet(np.ones(n_obj)))
        
        return np.array(weights[:n])
    
    def _local_refine(self, problem, X, F, lo, hi, rng, remaining, deadline):
        """Refine the best solutions with small local perturbations."""
        # Find non-dominated solutions and refine them
        fronts = self._non_dominated_sort(F)
        nd_indices = np.array(fronts[0]) if fronts else np.arange(len(F))
        
        # Refine up to 'remaining' solutions from the first front
        n_refine = min(len(nd_indices), remaining)
        if n_refine == 0:
            return X, F
        
        # Sort by crowding distance to refine the most diverse
        F_nd = F[nd_indices]
        cd = self._crowding_distance(F_nd)
        sorted_nd = nd_indices[np.argsort(-cd)]
        refine_idx = sorted_nd[:n_refine]
        
        for idx in refine_idx:
            if time.time() > deadline:
                break
            x = X[idx].copy()
            f = F[idx].copy()
            
            # Try small perturbations in each dimension
            for d in range(self.dim):
                if time.time() > deadline:
                    break
                step = (hi[d] - lo[d]) * 0.01
                for direction in [-1, 1]:
                    x_new = x.copy()
                    x_new[d] = np.clip(x_new[d] + direction * step, lo[d], hi[d])
                    f_new = np.array(problem(x_new))
                    
                    # Accept if it dominates or is non-dominated
                    if np.all(f_new <= f) and np.any(f_new < f):
                        x = x_new
                        f = f_new
                    elif not np.any(np.all(f <= f_new) and np.any(f < f_new)):
                        # Check if f_new is not dominated by f
                        if not (np.all(f <= f_new) and np.any(f < f_new)):
                            # Keep the better one based on hypervolume contribution
                            # Simple heuristic: keep if sum of objectives is smaller
                            if np.sum(f_new) < np.sum(f):
                                x = x_new
                                f = f_new
        
        X[idx] = x
        F[idx] = f
        
        return X, F
    
    def _tournament_select(self, X, F, rng, k=3):
        idxs = rng.integers(0, len(F), size=k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _poly_mutate(self, x, lo, hi, eta, pm, rng):
        n = len(x)
        x_mut = x.copy()
        for i in range(n):
            if rng.random() < pm:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                range_i = hi[i] - lo[i]
                if range_i <= 0:
                    continue
                
                if delta >= 0:
                    x_mut[i] = x[i] + delta * (hi[i] - x[i])
                else:
                    x_mut[i] = x[i] - (-delta) * (x[i] - lo[i])
                
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def _nsga2_select(self, P, F_p, target_size, rng):
        fronts = self._non_dominated_sort(F_p)
        
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
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

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
                    u = rng.random()
                    if 2 * u <= (1 / (eta + 1)):
                        beta = (2 * u) ** (1 / (eta + 1))
                    elif 2 * (1 - u) <= (1 / (eta + 1)):
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    else:
                        beta = 1.0
                    
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

    def _generate_weights(self, n, n_obj, rng):
        """Generate weight vectors for decomposition using simplex lattice."""
        if n_obj == 1:
            return np.ones((n, 1))
        
        weights = np.empty((n, n_obj))
        if n_obj == 3:
            # Generate 20 points on the 2D simplex for 3 objectives
            # Use a grid-like approach for better coverage
            n_per_edge = int(np.ceil(np.sqrt(2 * n)))
            count = 0
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    if count >= n:
                        break
                    k = n_per_edge - i - j
                    if k < 0:
                        continue
                    total = i + j + k
                    if total == 0:
                        continue
                    weights[count] = [i / total, j / total, k / total]
                    count += 1
                if count >= n:
                    break
            # If we didn't fill all, add random Dirichlet points
            while count < n:
                weights[count] = rng.dirichlet([1.0, 1.0, 1.0])
                count += 1
        else:
            for i in range(n):
                weights[i] = rng.dirichlet(np.ones(n_obj))
        return weights

    def _tournament_select(self, X, F, rng, k=3):
        idxs = rng.integers(0, len(F), size=k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _poly_mutate(self, x, lo, hi, eta, pm, rng):
        """Polynomial mutation for bounded real-valued problems."""
        n = len(x)
        x_mut = x.copy()
        for i in range(n):
            if rng.random() < pm:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                range_i = hi[i] - lo[i]
                if range_i <= 0:
                    continue
                
                # Polynomial mutation formula
                if delta >= 0:
                    x_mut[i] = x[i] + delta * (hi[i] - x[i])
                else:
                    x_mut[i] = x[i] - (-delta) * (x[i] - lo[i])
                
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

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
            # Sort by k-th objective
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
                    u = rng.random()
                    if 2 * u <= (1 / (eta + 1)):
                        beta = (2 * u) ** (1 / (eta + 1))
                    elif 2 * (1 - u) <= (1 / (eta + 1)):
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    else:
                        beta = 1.0
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2
