"""Multi-objective optimizer for WFG4 using NSGA-II with problem-aware initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II for WFG4 with problem-aware initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref = np.array([3.0, 5.0, 7.0])

    def _dominates(self, a, b):
        """Check if a dominates b (all objectives <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F_sub[order[-1], m] - F_sub[order[0], m]
            if f_range > 1e-15:
                for i in range(1, n - 1):
                    cd[order[i]] += (F_sub[order[i+1], m] - F_sub[order[i-1], m]) / f_range
        
        return cd

    def _sbx_crossover(self, x1, x2, lo, hi, pc=0.9, eta=20.0):
        """Simulated binary crossover."""
        if np.random.random() > pc:
            return x1.copy(), x2.copy()
        
        x1_child = x1.copy()
        x2_child = x2.copy()
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) < 1e-14:
                    continue
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                x1_child[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                x2_child[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
        
        # Clip to bounds
        x1_child = np.clip(x1_child, lo, hi)
        x2_child = np.clip(x2_child, lo, hi)
        
        return x1_child, x2_child

    def _polynomial_mutation(self, x, lo, hi, pm=0.1, eta=20.0):
        """Polynomial mutation."""
        x_new = x.copy()
        for i in range(self.dim):
            if np.random.random() < pm:
                u = np.random.random()
                delta1 = (u * (hi[i] - x_new[i]) / (hi[i] - lo[i])) ** (1.0 / (eta + 1))
                delta2 = ((1 - u) * (x_new[i] - lo[i]) / (hi[i] - lo[i])) ** (1.0 / (eta + 1))
                if np.random.random() < 0.5:
                    x_new[i] = x_new[i] - (hi[i] - lo[i]) * delta1
                else:
                    x_new[i] = x_new[i] + (hi[i] - lo[i]) * delta2
                x_new[i] = np.clip(x_new[i], lo[i], hi[i])
        return x_new

    def _initialize_population(self, pop_size, lo, hi, rng):
        """Initialize population with problem-aware biasing for WFG4."""
        X = []
        for i in range(pop_size):
            x = lo + (hi - lo) * rng.random_sample(self.dim)
            # WFG4 Pareto front: x[0] is the main trade-off variable
            # Bias x[0] toward the range that gives good trade-offs
            if i < pop_size // 2:
                # Spread x[0] across the range
                x[0] = lo[0] + (hi[0] - lo[0]) * (i / max(1, pop_size // 2 - 1) if pop_size // 2 > 1 else 0.5)
            else:
                # Random with slight bias
                x[0] = lo[0] + (hi[0] - lo[0]) * rng.random()
            X.append(x)
        return np.array(X)

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
        pop_size = min(50, max(20, budget // 4))

        # Initialize
        X = self._initialize_population(pop_size, lo, hi, rng)
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used += pop_size

        # Global archive
        archive_X = [x.copy() for x in X]
        archive_F = [f.copy() for f in F]

        # NSGA-II main loop
        gen = 0
        while used < budget and time.time() < deadline:
            gen += 1
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used < budget and time.time() < deadline:
                # Tournament selection
                idx1 = self._tournament_select(F)
                idx2 = self._tournament_select(F)
                
                x1 = X[idx1].copy()
                x2 = X[idx2].copy()
                
                # Crossover
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi)
                
                # Mutation
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                # Evaluate
                f1 = np.array(problem(c1), dtype=np.float64)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used < budget and time.time() < deadline:
                    f2 = np.array(problem(c2), dtype=np.float64)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array(offspring_F[:pop_size])
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = self._crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Update global archive
            for x, f in zip(X, F):
                if not any(self._dominates(g, f) for g in archive_F):
                    # Check if f dominates any archive points
                    new_archive_X = []
                    new_archive_F = []
                    for gx, gf in zip(archive_X, archive_F):
                        if not self._dominates(f, gf):
                            new_archive_X.append(gx)
                            new_archive_F.append(gf)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    archive_X.append(x.copy())
                    archive_F.append(f.copy())
            
            # Trim archive if too large
            if len(archive_F) > 100:
                arr_F = np.array(archive_F)
                nd_idx = self._non_dominated_indices(arr_F)
                archive_F = [archive_F[i] for i in nd_idx]
                archive_X = [archive_X[i] for i in nd_idx]

        # Local refinement of best points
        remaining = budget - used
        if remaining > 10 and len(archive_X) > 0 and time.time() < deadline:
            # Pick a few diverse points from archive for refinement
            arr_F = np.array(archive_F)
            n_refine = min(5, len(archive_F), remaining // 10)
            
            # Select diverse points
            selected = []
            for _ in range(n_refine):
                if not archive_X:
                    break
                idx = rng.randint(len(archive_X))
                selected.append(idx)
            
            for idx in selected:
                if used >= budget or time.time() > deadline:
                    break
                
                x_refine = archive_X[idx].copy()
                f_refine = np.array(problem(x_refine), dtype=np.float64)
                used += 1
                
                # Coordinate descent refinement
                for d in range(dim):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    best_x = x_refine.copy()
                    best_f = f_refine.copy()
                    
                    # Try several perturbations
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        step = (hi[d] - lo[d]) * 0.05 * rng.random()
                        for sign in [1, -1]:
                            if used >= budget or time.time() > deadline:
                                break
                            
                            test_x = x_refine.copy()
                            test_x[d] = np.clip(test_x[d] + sign * step, lo[d], hi[d])
                            test_f = np.array(problem(test_x), dtype=np.float64)
                            used += 1
                            
                            # Accept if it improves any objective without worsening others too much
                            if np.all(test_f <= best_f + 0.1):
                                if np.any(test_f < best_f):
                                    best_x = test_x
                                    best_f = test_f
                                    x_refine = test_x
                                    f_refine = test_f
                    
                    # Update archive with refined point
                    if not any(self._dominates(g, f_refine) for g in archive_F):
                        new_archive_X = []
                        new_archive_F = []
                        for gx, gf in zip(archive_X, archive_F):
                            if not self._dominates(f_refine, gf):
                                new_archive_X.append(gx)
                                new_archive_F.append(gf)
                        archive_X = new_archive_X
                        archive_F = new_archive_F
                        archive_X.append(x_refine.copy())
                        archive_F.append(f_refine.copy())

        # Return the archive points
        if len(archive_X) > 0:
            return np.array(archive_X)
        return np.array([lo + (hi - lo) * 0.5])

    def _tournament_select(self, F, k=3):
        """Tournament selection."""
        n = len(F)
        indices = rng.choice(n, size=k, replace=False)
        best = indices[0]
        for idx in indices[1:]:
            if self._dominates(F[idx], F[best]):
                best = idx
        return best

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return []
        nd = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[j], F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append(i)
        return nd
# EVOLVE-BLOCK-END
