# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer for WFG4 with NSGA-II selection and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (each is array of indices)."""
        n = F.shape[0]
        if n == 0:
            return []
        
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        front = np.array([i for i in range(n) if domination_count[i] == 0])
        if len(front) == 0:
            return [np.arange(n)]
        
        fronts = [front]
        current_front = front
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(np.array(next_front))
                current_front = np.array(next_front)
            else:
                break
        
        return fronts

    @staticmethod
    def _crowding_distance(F, idx):
        """Compute crowding distance for points at indices idx in F."""
        k = len(idx)
        if k <= 2:
            return np.full(k, np.inf)
        M = F.shape[1]
        dist = np.zeros(k)
        Ff = F[idx]
        for m in range(M):
            vals = Ff[:, m]
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1e-18:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, k - 1):
                dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (vmax - vmin)
        return dist

    @staticmethod
    def _hypervolume_contribution(F, ref_point):
        """Compute approximate hypervolume contribution of a set of points.
        Returns a score that can be used for comparison."""
        n = F.shape[0]
        if n == 0:
            return 0.0
        
        # Filter points that are better than reference in all objectives
        mask = np.all(F < ref_point, axis=1)
        F_valid = F[mask]
        n_valid = F_valid.shape[0]
        
        if n_valid == 0:
            return 0.0
        
        # For 3 objectives, compute hypervolume using inclusion-exclusion
        # Sort by first objective
        order = np.argsort(F_valid[:, 0])
        F_sorted = F_valid[order]
        
        # Build skyline in 2D (f2, f3) for each slice of f1
        # Use a grid-based approximation for speed
        total_hv = 0.0
        ref = np.array(ref_point)
        
        for i in range(n_valid):
            f1 = F_sorted[i, 0]
            # Points with f1 <= F_sorted[i, 0] and f2, f3 < ref
            # Contribution is volume of box from (f1, f2_i, f3_i) to ref
            # But we need to account for overlaps - use a simple approach
            
            # For each point, compute the box it uniquely contributes
            # This is approximate but good enough for selection
            f2_min = np.min(F_sorted[:i+1, 1])
            f3_min = np.min(F_sorted[:i+1, 2])
            
            # Better approach: compute HV incrementally
            # The contribution of point i is the volume of the box
            # [F_sorted[i,0], ref[0]] x [F_sorted[i,1], ref[1]] x [F_sorted[i,2], ref[2]]
            # minus overlaps with other points
        
        # Simpler: just use the sum of individual box volumes as a proxy
        # This overcounts but preserves relative ordering
        boxes = (ref - F_valid)
        boxes = np.maximum(boxes, 0)
        hv_approx = np.prod(boxes, axis=1).sum()
        
        return hv_approx

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: larger for better spread
        N = min(36, max(12, budget // 6))

        # Initial random population
        X = lo + span * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N

        # DE parameters
        CR = 0.9

        def time_ok():
            return (time.time() - start_time) < deadline and used < budget

        # Phase 1: DE-based exploration with NSGA-II selection
        while time_ok():
            n_offspring = min(N, budget - used)
            
            children = np.zeros((n_offspring, dim))
            Fc = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                if not time_ok():
                    break
                
                idx = np.random.choice(N, 3, replace=False)
                a, b, c = X[idx[0]], X[idx[1]], X[idx[2]]
                base = X[i % N]
                
                # Adaptive F: starts high for exploration, decreases for exploitation
                progress = used / budget
                F_factor = 0.6 + 0.4 * np.random.rand() * (1.0 - 0.5 * progress)
                
                v = a + F_factor * (b - c)
                
                u = np.random.rand(dim) < CR
                if not np.any(u):
                    jrand = np.random.randint(dim)
                    u[jrand] = True
                
                child = np.where(u, v, base)
                child = np.clip(child, lo, hi)
                
                children[i] = child
                Fc[i] = problem(child)
                used += 1
            
            if n_offspring == 0:
                break

            Xall = np.vstack([X, children[:n_offspring]])
            Fall = np.vstack([F, Fc[:n_offspring]])
            
            fronts = self._fast_non_dominated_sort(Fall)
            
            selected = np.zeros(N, dtype=int)
            count = 0
            for fr in fronts:
                if count + len(fr) <= N:
                    selected[count:count + len(fr)] = fr
                    count += len(fr)
                else:
                    need = N - count
                    cd = self._crowding_distance(Fall, fr)
                    order = np.argsort(-cd)
                    selected[count:count + need] = fr[order[:need]]
                    count += need
                    break
            
            X = Xall[selected]
            F = Fall[selected]

        # Phase 2: Local polish - refine best solutions with adaptive perturbations
        if time_ok() and used < budget:
            # Get the first front (best non-dominated solutions)
            fronts = self._fast_non_dominated_sort(F)
            if fronts and len(fronts[0]) > 0:
                best_idx = fronts[0]
                n_best = len(best_idx)
                
                # Try to refine each best solution with small Gaussian perturbations
                remaining_budget = budget - used
                attempts_per_solution = max(2, remaining_budget // max(1, n_best))
                
                # Track best hypervolume for comparison
                ref_point = np.array([3.0, 5.0, 7.0])
                
                for bi in range(n_best):
                    if used >= budget or (time.time() - start_time) > deadline:
                        break
                    
                    base_x = X[best_idx[bi]]
                    base_f = F[best_idx[bi]]
                    
                    # Adaptive sigma: starts larger, shrinks over attempts
                    sigma_start = 0.03 * span
                    sigma_end = 0.001 * span
                    
                    for attempt in range(attempts_per_solution):
                        if used >= budget or (time.time() - start_time) > deadline:
                            break
                        
                        # Linearly decrease sigma
                        frac = attempt / max(1, attempts_per_solution - 1)
                        sigma = sigma_start + (sigma_end - sigma_start) * frac
                        sigma = np.maximum(sigma, 0.0005 * span)
                        
                        child = base_x + sigma * np.random.randn(dim)
                        child = np.clip(child, lo, hi)
                        
                        fc = problem(child)
                        used += 1
                        
                        # Check if child dominates base
                        if np.all(fc <= base_f) and np.any(fc < base_f):
                            X[best_idx[bi]] = child
                            F[best_idx[bi]] = fc
                            base_f = fc
                            base_x = child
                        elif not (np.all(base_f <= fc) and np.any(base_f < fc)):
                            # Non-dominated: use hypervolume-aware comparison
                            # Create temporary set with base and child
                            F_temp = np.vstack([base_f.reshape(1, -1), fc.reshape(1, -1)])
                            hv_with_base = self._hypervolume_contribution(F_temp, ref_point)
                            
                            # Compare with just base
                            hv_base_only = self._hypervolume_contribution(
                                base_f.reshape(1, -1), ref_point)
                            
                            # If adding child increases HV, replace
                            if hv_with_base > hv_base_only * 1.001:
                                X[best_idx[bi]] = child
                                F[best_idx[bi]] = fc
                                base_f = fc
                                base_x = child

        return X
# EVOLVE-BLOCK-END
