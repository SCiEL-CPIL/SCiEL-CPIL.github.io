"""Multi-objective optimizer for WFG4 using NSGA-II + Local Search Polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        # 1. Initialization with LHS
        init_size = 20
        X = np.zeros((init_size, self.dim))
        for i in range(init_size):
            for j in range(self.dim):
                X[i, j] = lo[j] + (hi[j] - lo[j]) * (i + rng.random()) / init_size
        
        F = np.array([problem(x) for x in X])
        used = init_size

        # Pad population to size 40 if needed
        pop_size = 40
        if init_size < pop_size:
            X_pad = lo + (hi - lo) * rng.random((pop_size - init_size, self.dim))
            F_pad = np.array([problem(x) for x in X_pad])
            used += len(X_pad)
            X = np.vstack([X, X_pad])
            F = np.vstack([F, F_pad])

        # 2. NSGA-II Loop
        # Reserve 20 evaluations for polishing
        polish_budget = 20
        main_budget = self.budget - polish_budget

        while used < main_budget:
            if time.time() - start_time > deadline:
                break
                
            # Generate offspring
            n_offspring = pop_size
            X_off = np.zeros((n_offspring, self.dim))
            for i in range(0, n_offspring, 2):
                # Tournament Selection
                p1 = self._tournament(F, rng)
                p2 = self._tournament(F, rng)
                
                # SBX Crossover
                c1, c2 = self._sbx(p1, p2, lo, hi, rng)
                X_off[i] = c1
                if i + 1 < n_offspring:
                    X_off[i+1] = c2
            
            # Mutation
            for i in range(n_offspring):
                X_off[i] = self._polymut(X_off[i], lo, hi, rng)
            
            # Evaluate
            F_off = np.array([problem(x) for x in X_off])
            used += n_offspring
            
            # Combine and Select
            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            X, F = self._nsga2_select(X_comb, F_comb, pop_size, rng)

        # 3. Polish Phase
        # Get non-dominated points
        non_dom_idx = self._nondominated_indices(F)
        X_nd = X[non_dom_idx]
        F_nd = F[non_dom_idx]
        
        # Select top K points to polish (e.g., 5 points with lowest sum of objectives)
        if len(F_nd) > 0:
            sums = np.sum(F_nd, axis=1)
            top_k_indices = np.argsort(sums)[:5]
            polish_candidates = X_nd[top_k_indices].copy()
            
            # Use remaining budget to polish
            remaining = self.budget - used
            for i in range(len(polish_candidates)):
                if used >= self.budget:
                    break
                if time.time() - start_time > deadline:
                    break
                
                # Local search: try small perturbations
                best_x = polish_candidates[i]
                # We don't have F for best_x in this context easily without re-eval, 
                # but we know it's from the archive. Let's just do a simple hill climb.
                # Actually, we should re-evaluate or keep track.
                # Let's just do random perturbations and keep if better than current best sum.
                
                # Get current objective sum
                f_best = problem(best_x)
                sum_best = np.sum(f_best)
                
                steps = max(1, remaining // len(polish_candidates))
                for _ in range(steps):
                    if used >= self.budget:
                        break
                    if time.time() - start_time > deadline:
                        break
                    
                    # Perturb
                    x_new = best_x + rng.normal(0, 0.05, size=self.dim)
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    sum_new = np.sum(f_new)
                    
                    if sum_new < sum_best:
                        best_x = x_new
                        f_best = f_new
                        sum_best = sum_new
                        
                # Add polished point to archive if it's non-dominated
                # Check if best_x dominates any existing non-dominated points
                # Or just add it to a final list and do a final non-dominated sort
                # For simplicity, let's just collect all polished points and do a final cleanup
                # But we need to return the set of points. The evaluator records EVERY returned objective vector.
                # So we should add the polished points to the final population before returning.
                
                # We'll just add it to X and F
                X = np.vstack([X, best_x.reshape(1, -1)])
                F = np.vstack([F, f_best.reshape(1, -1)])

        return X

    def _tournament(self, F, rng):
        """Binary tournament selection."""
        i1 = rng.integers(0, len(F))
        i2 = rng.integers(0, len(F))
        f1 = F[i1]
        f2 = F[i2]
        
        # Check domination
        if (f1 <= f2).all() and (f1 < f2).any():
            return i1
        elif (f2 <= f1).all() and (f2 < f1).any():
            return i2
        else:
            # Tie-break by sum (simple)
            if np.sum(f1) < np.sum(f2):
                return i1
            else:
                return i2

    def _sbx(self, p1, p2, lo, hi, rng):
        """Simulated Binary Crossover."""
        eta = 20.0
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        
        if rng.random() < 0.5:
            for i in range(len(p1)):
                if rng.random() < 0.5:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    if abs(p1[i] - p2[i]) < 1e-14:
                        c1[i] = p1[i]
                        c2[i] = p2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        
                        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                        
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
        else:
            c1 = p1.copy()
            c2 = p2.copy()
            
        return c1, c2

    def _polymut(self, x, lo, hi, rng):
        """Polynomial Mutation."""
        eta = 20.0
        pm = 1.0 / len(x)
        x_new = x.copy()
        for i in range(len(x)):
            if rng.random() < pm:
                y = x_new[i]
                yl = lo[i]
                yu = hi[i]
                
                if yu - yl < 1e-14:
                    continue
                    
                u = rng.random()
                if u < 0.5:
                    delta1 = (y - yl) / (yu - yl)
                    xq = 1 + (1 - delta1) * (2 * u + 1 - (1 - delta1) * (2 * u + 1)) ** (1 / (eta + 1))
                    # Simplified polynomial mutation formula
                    # Standard: 
                    # if u < 0.5:
                    #   delta = (2*u)^(1/(eta+1))
                    #   xq = y - delta * (yu - y)
                    # else:
                    #   delta = (2*(1-u))^(1/(eta+1))
                    #   xq = y + delta * (y - yl)
                    
                    delta = (2 * u) ** (1 / (eta + 1))
                    xq = y - delta * (yu - y)
                else:
                    delta = (2 * (1 - u)) ** (1 / (eta + 1))
                    xq = y + delta * (y - yl)
                    
                x_new[i] = np.clip(xq, yl, yu)
                
        return x_new

    def _nsga2_select(self, X, F, pop_size, rng):
        """NSGA-II selection: Non-dominated sort + Crowding distance."""
        # 1. Non-dominated sort
        ranks = np.zeros(len(F), dtype=int)
        front = []
        remaining = set(range(len(F)))
        
        rank = 0
        while remaining:
            current_front = self._nondominated_indices_subset(F[list(remaining)], list(remaining))
            if not current_front:
                break
            for idx in current_front:
                ranks[idx] = rank
                remaining.remove(idx)
            front.append(current_front)
            rank += 1
            
        # 2. Crowding distance for each front
        distances = np.zeros(len(F))
        for f in front:
            if len(f) <= 2:
                for idx in f:
                    distances[idx] = np.inf
            else:
                for m in range(self.n_obj):
                    order = np.argsort(F[f, m])
                    distances[order[0]] = np.inf
                    distances[order[-1]] = np.inf
                    for i in range(1, len(order) - 1):
                        denom = F[f[order[i+1]], m] - F[f[order[i-1]], m]
                        if denom > 1e-10:
                            distances[f[order[i]]] += 1.0 / denom

        # 3. Selection
        selected_idx = []
        for f in front:
            if len(selected_idx) + len(f) <= pop_size:
                selected_idx.extend(f)
            else:
                # Sort front by crowding distance
                f_sorted = sorted(f, key=lambda x: distances[x], reverse=True)
                needed = pop_size - len(selected_idx)
                selected_idx.extend(f_sorted[:needed])
                break
                
        return X[selected_idx], F[selected_idx]

    def _nondominated_indices(self, F):
        """Return indices of non-dominated points in F."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        leq = (F[:, np.newaxis, :] <= F[np.newaxis, :, :]).all(axis=2)
        lt = (F[:, np.newaxis, :] < F[np.newaxis, :, :]).any(axis=2)
        dom_matrix = leq & lt
        np.fill_diagonal(dom_matrix, False)
        dominated = dom_matrix.any(axis=1)
        return np.where(~dominated)[0]

    def _nondominated_indices_subset(self, F_sub, indices):
        """Return original indices of non-dominated points in subset."""
        if len(F_sub) == 0:
            return []
        local_nd = self._nondominated_indices(F_sub)
        return [indices[i] for i in local_nd]


# EVOLVE-BLOCK-END
