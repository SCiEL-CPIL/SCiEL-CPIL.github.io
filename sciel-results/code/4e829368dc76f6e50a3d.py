# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer for RE37 using structured sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])  # reference point from scoring

        self._eval_count = 0
        self._max_budget = self.budget

        def evaluate(x):
            self._eval_count += 1
            if self._eval_count > self._max_budget:
                raise ValueError("Budget exceeded")
            return np.array(problem(x))

        def lhs_sample(n, d, lo, hi):
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = np.random.permutation(n)
                samples[:, dd] = (perm + np.random.random(n)) / n
            return lo + (hi - lo) * samples

        def hypervolume_3d(points, ref):
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            n = len(pts)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                        non_dom[i] = False
                        break
            pts = pts[non_dom]
            n = len(pts)
            if n == 0:
                return 0.0
            
            # Grid-based HV calculation (exact for small n)
            x_coords = sorted(set([ref[0]] + list(pts[:, 0])))
            y_coords = sorted(set([ref[1]] + list(pts[:, 1])))
            z_coords = sorted(set([ref[2]] + list(pts[:, 2])))
            
            hv = 0.0
            for xi in range(len(x_coords) - 1):
                for yi in range(len(y_coords) - 1):
                    for zi in range(len(z_coords) - 1):
                        covered = False
                        for p in pts:
                            if p[0] <= x_coords[xi] and p[1] <= y_coords[yi] and p[2] <= z_coords[zi]:
                                covered = True
                                break
                        if covered:
                            vol = (x_coords[xi+1] - x_coords[xi]) * (y_coords[yi+1] - y_coords[yi]) * (z_coords[zi+1] - z_coords[zi])
                            hv += vol
            return hv

        # Phase 1: Structured Global Sampling
        n_init = 60
        X_init = lhs_sample(n_init, dim, lo, hi)
        F_init = np.array([evaluate(x) for x in X_init])
        
        all_X = [X_init]
        all_F = [F_init]
        
        # Phase 2: Targeted Local Refinement
        # Identify non-dominated points from initial set
        def get_non_dominated(F):
            n = len(F)
            if n == 0: return []
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        nd_indices = get_non_dominated(F_init)
        
        # Generate weight vectors for scalarization
        # Use a simplex lattice for 3 objectives
        weights = []
        n_levels = 4
        for i in range(n_levels + 1):
            for j in range(n_levels + 1 - i):
                k = n_levels - i - j
                w = np.array([i, j, k], dtype=float)
                if np.sum(w) > 0:
                    weights.append(w / np.sum(w))
        
        # Shuffle weights to randomize order
        np.random.shuffle(weights)
        
        remaining_budget = self.budget - self._eval_count
        n_weights = len(weights)
        
        # Allocate budget to local searches
        # Reserve some budget for final random checks? No, let's use it all for local search.
        evals_per_search = max(10, remaining_budget // n_weights)
        
        for w in weights:
            if self._eval_count >= self.budget or time.time() > deadline:
                break
            
            # Find best starting point: minimize w^T F among non-dominated points
            if len(nd_indices) > 0:
                scores = F_init[nd_indices] @ w
                best_idx = nd_indices[np.argmin(scores)]
                x_start = X_init[best_idx].copy()
            else:
                x_start = X_init[0].copy()
            
            # Define scalarized objective
            def obj_scalar(x):
                if time.time() > deadline:
                    return 1e10
                fx = evaluate(x)
                return fx @ w

            # Run Nelder-Mead
            try:
                res = minimize(obj_scalar, x_start, method='Nelder-Mead', 
                               options={'maxiter': 50, 'maxfev': evals_per_search, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.success or res.fun < 1e9:
                    x_opt = res.x
                    f_opt = evaluate(x_opt)
                    # Check if it's non-dominated against current best
                    # For simplicity, just add it. We'll filter at the end.
                    all_X.append(x_opt.reshape(1, -1))
                    all_F.append(f_opt.reshape(1, -1))
            except Exception:
                pass

        # Phase 3: Final Random Polish (if budget remains)
        while self._eval_count < self.budget and time.time() < deadline:
            x_rand = lo + np.random.random(dim) * (hi - lo)
            f_rand = evaluate(x_rand)
            all_X.append(x_rand.reshape(1, -1))
            all_F.append(f_rand.reshape(1, -1))

        # Merge and Filter
        X_final = np.vstack(all_X)
        F_final = np.vstack(all_F)
        
        # Remove dominated points
        n = len(F_final)
        non_dom = np.ones(n, dtype=bool)
        for i in range(n):
            if not non_dom[i]:
                continue
            for j in range(n):
                if i == j or not non_dom[j]:
                    continue
                if np.all(F_final[j] <= F_final[i]) and np.any(F_final[j] < F_final[i]):
                    non_dom[i] = False
                    break
        
        X_final = X_final[non_dom]
        F_final = F_final[non_dom]
        
        return X_final
# EVOLVE-BLOCK-END
