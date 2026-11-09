"""Multi-objective optimizer for RE21 using hypervolume-driven search with local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer combining:
    1. Diverse initial sampling (including boundary points)
    2. Hypervolume-based selection for local search
    3. Adaptive step sizes with stagnation detection
    4. Weighted-sum local refinement via L-BFGS-B
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume for a set of points with given reference point."""
        if len(points) == 0:
            return 0.0
        # Only consider points that are better than reference in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        hv = 0.0
        for i in range(len(pts)):
            # Width in objective 1
            w1 = ref[0] - pts[i, 0] if i == len(pts) - 1 else pts[i + 1, 0] - pts[i, 0]
            # Height in objective 2: the best (minimum) value among points i..end
            h2 = ref[1] - np.min(pts[i:, 1])
            hv += w1 * h2
        return hv

    def _dominance_mask(self, F):
        """Return boolean mask where True means dominated."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # Check if any other point dominates i
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 30s safety margin before 310s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Track all evaluated points
        all_X = []
        all_F = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return x, f
        
        # Phase 1: Diverse initial sampling
        n_init = min(40, self.budget // 5)
        
        # Random interior points
        n_random = n_init // 2
        for _ in range(n_random):
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
        
        # Boundary points: sample at corners and edges
        n_boundary = n_init - n_random
        for _ in range(n_boundary):
            # Randomly choose which dimensions to set to bounds
            x = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    x[d] = lo[d]
                else:
                    x[d] = hi[d]
            # Add small random perturbation to avoid exact duplicates
            x = x + 0.01 * (hi - lo) * rng.standard_normal(self.dim)
            evaluate(x)
        
        # Phase 2: Hypervolume-driven local search
        # Maintain a population of non-dominated points
        pop_X = np.array(all_X[:n_init])
        pop_F = np.array(all_F[:n_init])
        
        # Adaptive step size
        sigma = 0.3  # Initial step size as fraction of range
        stagnation_counter = 0
        best_hv = self._hypervolume_2d(pop_F, [3000.0, 0.06])
        
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated points
            dominated = self._dominance_mask(pop_F)
            if np.any(~dominated):
                parents = pop_X[~dominated]
                parent_F = pop_F[~dominated]
            else:
                parents = pop_X
                parent_F = pop_F
            
            # Select parent with maximum hypervolume contribution
            # Approximate: select parent that, if improved, would give best HV gain
            # Simplified: select random non-dominated parent
            idx = rng.integers(len(parents))
            parent = parents[idx]
            
            # Generate child with adaptive mutation
            child = parent + sigma * (hi - lo) * rng.standard_normal(self.dim)
            child, fc = evaluate(child)
            
            # Check improvement
            current_hv = self._hypervolume_2d(pop_F, [3000.0, 0.06])
            # Add child to population and check if it improves HV
            new_F = np.vstack([pop_F, fc])
            new_hv = self._hypervolume_2d(new_F, [3000.0, 0.06])
            
            if new_hv > current_hv:
                # Child improves hypervolume, add to population
                pop_X = np.vstack([pop_X, child])
                pop_F = new_F
                stagnation_counter = 0
                # Slightly increase sigma for exploration
                sigma = min(0.5, sigma * 1.1)
            else:
                # No improvement, reduce step size
                sigma = max(0.05, sigma * 0.8)
                stagnation_counter += 1
            
            # Periodically run local refinement on best points
            if eval_count % 10 == 0 and eval_count < self.budget - 5:
                # Find point with best hypervolume contribution
                # Simplified: pick the point closest to ideal point
                ideal = np.min(pop_F, axis=0)
                distances = np.sum((pop_F - ideal) ** 2, axis=1)
                best_idx = np.argmin(distances)
                best_point = pop_X[best_idx]
                
                # Try weighted-sum local search
                weights = rng.dirichlet(np.ones(self.n_obj))
                def weighted_sum(x):
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x), dtype=float)
                    eval_count += 1
                    all_X.append(x.copy())
                    all_F.append(f.copy())
                    return np.dot(weights, f)
                
                try:
                    res = minimize(weighted_sum, best_point, method='L-BFGS-B',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 5, 'maxfun': 10})
                    if res.success or res.fun < np.dot(weights, np.array(problem(best_point))):
                        res_x = np.clip(res.x, lo, hi)
                        res_f = np.array(problem(res_x), dtype=float)
                        eval_count += 1
                        all_X.append(res_x.copy())
                        all_F.append(res_f.copy())
                        # Add to population if non-dominated
                        new_F = np.vstack([pop_F, res_f])
                        if not self._dominance_mask(new_F)[-1]:
                            pop_X = np.vstack([pop_X, res_x])
                            pop_F = new_F
                except Exception:
                    pass
        
        return all_X
# EVOLVE-BLOCK-END
