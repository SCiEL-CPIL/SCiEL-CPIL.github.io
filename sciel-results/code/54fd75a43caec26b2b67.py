"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = int(min(30, self.budget // 6))
        if pop_size < 5:
            pop_size = 5

        # Random initialization
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best known points
        def get_non_dominated_indices(F_mat):
            """Return indices of non-dominated points in F_mat."""
            n = len(F_mat)
            non_dominated = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        non_dominated[i] = False
                        break
            return np.where(non_dominated)[0]

        def hv_contribution(point, ref, others):
            """Compute approximate hypervolume contribution using inclusion-exclusion for 3 objectives."""
            if np.any(point >= ref):
                return 0.0
            # For 3 objectives, HV contribution can be computed more accurately
            # Sort other points by first objective
            valid = others[np.all(others < ref, axis=1)]
            if len(valid) == 0:
                return np.prod(ref - point)
            
            # Simple approximation: product of (ref - point) minus overlaps
            # This is a rough estimate but captures the key idea
            vol = np.prod(ref - point)
            # Subtract volume covered by other points that are "behind" this point
            for other in valid:
                if np.all(other <= point):
                    # This other point is dominated by point, so it doesn't affect HV of point
                    continue
                # Overlap volume
                overlap_low = np.maximum(point, other)
                overlap_high = ref
                if np.all(overlap_low < overlap_high):
                    vol -= np.prod(overlap_high - overlap_low)
            return max(0.0, vol)

        # Phase 1: Evolutionary search (use ~70% of budget)
        evo_budget = int(self.budget * 0.7)
        
        while used < evo_budget:
            if time.time() > deadline:
                break

            # Get non-dominated parents
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            parents = X[nd_idx]
            F_parents = F[nd_idx]

            # Select parent: use tournament selection based on hypervolume contribution
            if len(parents) > 2:
                # Tournament of 2
                t1, t2 = rng.choice(len(parents), size=2, replace=False)
                h1 = hv_contribution(F_parents[t1], np.array([3.0, 5.0, 7.0]), F_parents)
                h2 = hv_contribution(F_parents[t2], np.array([3.0, 5.0, 7.0]), F_parents)
                parent_idx = t1 if h1 >= h2 else t2
            else:
                parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]

            # Adaptive mutation step size
            progress = used / self.budget
            base_step = 0.3 * (1 - progress) + 0.05
            if rng.random() < 0.1:
                step_size = base_step * 3
            else:
                step_size = base_step

            # Generate child
            child = parent + step_size * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Determine which point to replace
            dominated_mask = np.array([
                np.all(F[i] >= fc) and np.any(F[i] > fc) for i in range(len(F))
            ])

            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                replace_idx = dominated_indices[rng.integers(len(dominated_indices))]
            else:
                sums = np.sum(F, axis=1)
                replace_idx = int(np.argmax(sums))
                if np.sum(fc) >= sums[replace_idx]:
                    continue

            X[replace_idx] = child
            F[replace_idx] = fc

        # Phase 2: Local refinement using Nelder-Mead on best points
        if time.time() < deadline:
            # Get non-dominated points for refinement
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))
            
            # Select top candidates for local optimization
            # Use hypervolume contribution to rank
            ref = np.array([3.0, 5.0, 7.0])
            hv_scores = []
            for i in nd_idx:
                hv_scores.append(hv_contribution(F[i], ref, F))
            hv_scores = np.array(hv_scores)
            
            # Sort by HV contribution (descending) and take top few
            n_refine = min(5, len(nd_idx))
            top_k = np.argsort(-hv_scores)[:n_refine]
            refine_indices = nd_idx[top_k]
            
            # For each top point, run Nelder-Mead on a weighted sum of objectives
            for idx in refine_indices:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                x0 = X[idx]
                f0 = F[idx]
                
                # Try multiple weight vectors to explore different parts of the front
                weight_vectors = [
                    [1.0, 1.0, 1.0],
                    [0.5, 0.3, 0.2],
                    [0.2, 0.3, 0.5],
                    [0.3, 0.5, 0.2],
                    [0.7, 0.1, 0.2],
                    [0.1, 0.7, 0.2],
                    [0.2, 0.1, 0.7],
                ]
                
                best_local = None
                best_local_f = None
                best_local_score = np.inf
                
                for w in weight_vectors:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    w_arr = np.array(w)
                    w_arr = w_arr / np.sum(w_arr)
                    
                    def objective(x):
                        nonlocal used
                        if used >= self.budget:
                            return np.inf
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        used += 1
                        return np.dot(w_arr, f)
                    
                    # Run Nelder-Mead with limited iterations
                    try:
                        result = minimize(
                            objective, x0, method='Nelder-Mead',
                            options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-6}
                        )
                        if result.fun < best_local_score:
                            best_local = result.x
                            best_local_f = problem(np.clip(result.x, lo, hi))
                            best_local_score = result.fun
                    except Exception:
                        pass
                
                # If we found a better local solution, replace the original point
                if best_local is not None and best_local_f is not None:
                    # Check if it's non-dominated or improves HV
                    is_dominated = False
                    for i in range(len(F)):
                        if i != idx:
                            if np.all(F[i] <= best_local_f) and np.any(F[i] < best_local_f):
                                is_dominated = True
                                break
                    
                    if not is_dominated:
                        X[idx] = best_local
                        F[idx] = best_local_f

        return X
# EVOLVE-BLOCK-END
