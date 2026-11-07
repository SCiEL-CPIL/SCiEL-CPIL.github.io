"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary search with adaptive mutation and archive-based parent selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        # domination count S[i] = set of indices dominated by i
        S = [[] for _ in range(n)]
        # domination count n_i = number of points that dominate i
        n_dom = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_idx = np.argsort(F_front[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            f_min = F_front[sorted_idx[0], m]
            f_max = F_front[sorted_idx[-1], m]
            f_range = f_max - f_min
            
            if f_range > 1e-15:
                for k in range(1, n - 1):
                    idx = sorted_idx[k]
                    distances[idx] += (F_front[sorted_idx[k + 1], m] - F_front[sorted_idx[k - 1], m]) / f_range
        
        return distances

    def _sbx_crossover(self, x1, x2, lo, hi, prob=0.9, eta=20.0):
        """Simulated Binary Crossover."""
        if np.random.rand() > prob:
            return x1.copy(), x2.copy()
        
        x1c = x1.copy()
        x2c = x2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                continue
            x1c[i], x2c[i] = x2[i], x1[i]
        
        # Apply SBX
        for i in range(self.dim):
            if x1c[i] == x2c[i]:
                continue
            if x1c[i] < x2c[i]:
                lo_i, hi_i = x1c[i], x2c[i]
            else:
                lo_i, hi_i = x2c[i], x1c[i]
            
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            
            c1 = 0.5 * ((1 + beta) * x1c[i] + (1 - beta) * x2c[i])
            c2 = 0.5 * ((1 - beta) * x1c[i] + (1 + beta) * x2c[i])
            
            if c1 < lo_i or c1 > hi_i:
                c1 = x1c[i]
            if c2 < lo_i or c2 > hi_i:
                c2 = x2c[i]
            
            x1c[i] = c1
            x2c[i] = c2
        
        return x1c, x2c

    def _polynomial_mutation(self, x, lo, hi, prob=1.0 / 12, eta=20.0):
        """Polynomial mutation."""
        x_new = x.copy()
        for i in range(self.dim):
            if np.random.rand() < prob:
                x_min = lo[i]
                x_max = hi[i]
                y = x_new[i]
                delta1 = (y - x_min) / (x_max - x_min) if (x_max - x_min) > 1e-15 else 0.5
                delta2 = (x_max - y) / (x_max - x_min) if (x_max - x_min) > 1e-15 else 0.5
                
                u = np.random.rand()
                if u < 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * (xy) ** (eta + 1)
                    deltaq = val ** (1.0 / (eta + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy) ** (eta + 1)
                    deltaq = 1 - val ** (1.0 / (eta + 1))
                
                x_new[i] += deltaq * (x_max - x_min)
                x_new[i] = np.clip(x_new[i], x_min, x_max)
        
        return x_new

    def _hypervolume_contribution(self, F, ref_point):
        """Compute hypervolume contribution of each point (approximate via simple method)."""
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Only consider points better than reference
        valid = np.all(F < ref_point, axis=1)
        if not np.any(valid):
            return np.zeros(n)
        
        F_valid = F[valid]
        n_valid = len(F_valid)
        
        # Simple HV contribution: for each point, compute volume of region it uniquely dominates
        # This is an approximation; for 3D we can do it more carefully
        contributions = np.zeros(n)
        
        for i in range(n):
            if not valid[i]:
                continue
            # Volume of box from F[i] to ref_point
            vol = 1.0
            for m in range(self.n_obj):
                if F[i, m] < ref_point[m]:
                    vol *= (ref_point[m] - F[i, m])
                else:
                    vol = 0
                    break
            contributions[i] = vol
        
        # Subtract overlaps approximately: for each pair, if j dominates i, reduce i's contribution
        # This is a rough approximation
        for i in range(n):
            if not valid[i]:
                continue
            for j in range(n):
                if i == j or not valid[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    # j dominates i, so i's unique contribution is reduced
                    # Approximate: reduce by overlap volume
                    overlap = 1.0
                    for m in range(self.n_obj):
                        low_m = max(F[i, m], F[j, m])
                        if low_m < ref_point[m]:
                            overlap *= (ref_point[m] - low_m)
                        else:
                            overlap = 0
                            break
                    contributions[i] = max(0, contributions[i] - 0.5 * overlap)
        
        return contributions

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        pop_size = min(40, max(10, budget // 5))
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        evaluations = pop_size
        
        # Archive of best non-dominated points
        archive = list(zip(X.tolist(), F.tolist()))
        
        while evaluations < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break
            
            # Non-dominated sort
            fronts = self._nondominated_sort(F)
            if not fronts:
                break
            
            # Compute crowding distance for each front
            front_dists = []
            for front in fronts:
                dists = self._crowding_distance(F, front)
                front_dists.append(dists)
            
            # Assign ranks and distances to each individual
            ranks = np.zeros(pop_size, dtype=int)
            cdist = np.zeros(pop_size)
            for rank, front in enumerate(fronts):
                dists = front_dists[rank]
                for pos, idx in enumerate(front):
                    ranks[idx] = rank
                    cdist[idx] = dists[pos]
            
            # Create mating pool using binary tournament on (rank, crowding)
            offspring = []
            n_offspring = pop_size
            
            for _ in range(n_offspring):
                # Tournament selection
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and cdist[i1] > cdist[i2]):
                    p1 = X[i1]
                else:
                    p1 = X[i2]
                
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and cdist[i1] > cdist[i2]):
                    p2 = X[i1]
                else:
                    p2 = X[i2]
                
                # Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, prob=0.9, eta=20.0)
                
                # Mutation
                # Adaptive mutation probability and strength
                progress = evaluations / budget
                mut_prob = max(0.05, 1.0 / dim * (1.0 - 0.5 * progress))
                c1 = self._polynomial_mutation(c1, lo, hi, prob=mut_prob, eta=20.0)
                c2 = self._polynomial_mutation(c2, lo, hi, prob=mut_prob, eta=20.0)
                
                offspring.append(c1)
                offspring.append(c2)
            
            offspring = np.array(offspring[:n_offspring])
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring], dtype=np.float64)
            evaluations += len(offspring)
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sort on combined population
            fronts_comb = self._nondominated_sort(F_comb)
            
            # Select next generation
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    # Need to select some from this front based on crowding
                    remaining = pop_size - len(X_new)
                    if remaining > 0:
                        dists = self._crowding_distance(F_comb, front)
                        # Sort by crowding distance (descending)
                        sorted_pos = np.argsort(dists)[::-1]
                        for pos in sorted_pos[:remaining]:
                            idx = front[pos]
                            X_new.append(X_comb[idx])
                            F_new.append(F_comb[idx])
                    break
            
            if len(X_new) > 0:
                X = np.array(X_new)
                F = np.array(F_new)
            
            # Update archive
            # Remove dominated points from archive
            new_archive = []
            for x, f in archive:
                dominated = False
                for _, f2 in archive:
                    if np.all(f2 <= f) and np.any(f2 < f):
                        dominated = True
                        break
                if not dominated:
                    new_archive.append((x, f))
            
            # Add new non-dominated points
            for i in range(len(F)):
                f = F[i]
                x = X[i]
                # Check if f is dominated by any archive point
                is_dominated = False
                for _, f2 in new_archive:
                    if np.all(f2 <= f) and np.any(f2 < f):
                        is_dominated = True
                        break
                if not is_dominated:
                    # Remove archive points dominated by f
                    new_archive = [(x2, f2) for x2, f2 in new_archive if not (np.all(f <= f2) and np.any(f < f2))]
                    new_archive.append((x, f))
            
            archive = new_archive
            
            # Occasionally add some random points to maintain diversity
            if evaluations < budget * 0.8 and evaluations % pop_size == 0:
                n_rand = max(1, pop_size // 10)
                for _ in range(n_rand):
                    if evaluations >= budget:
                        break
                    x_rand = lo + (hi - lo) * np.random.random(dim)
                    f_rand = problem(x_rand)
                    evaluations += 1
                    # Add to population if non-dominated
                    is_dominated = False
                    for i in range(len(F)):
                        if np.all(F[i] <= f_rand) and np.any(F[i] < f_rand):
                            is_dominated = True
                            break
                    if not is_dominated:
                        X = np.vstack([X, x_rand])
                        F = np.vstack([F, f_rand])
                        # Trim if too large
                        if len(X) > pop_size + 5:
                            # Keep only non-dominated
                            fronts = self._nondominated_sort(F)
                            if fronts:
                                keep = []
                                for front in fronts:
                                    if len(keep) + len(front) <= pop_size:
                                        keep.extend(front)
                                    else:
                                        remaining = pop_size - len(keep)
                                        if remaining > 0:
                                            dists = self._crowding_distance(F, front)
                                            sorted_pos = np.argsort(dists)[::-1]
                                            keep.extend(front[pos] for pos in sorted_pos[:remaining])
                                        break
                                X = X[keep]
                                F = F[keep]
        
        return X
# EVOLVE-BLOCK-END
