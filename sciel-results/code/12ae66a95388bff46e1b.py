# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 using Hypervolume-driven Evolutionary Strategy with local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo
        
        # Reference point for hypervolume calculation (from task description)
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Population size
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget

        # Initialize population
        X = lo + range_vec * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Adaptive step size
        step_size = 0.5 * np.max(range_vec) # Start with half the range
        min_step = 1e-3
        max_step = np.max(range_vec)
        
        # Track best non-dominated points for local refinement
        best_non_dominated = []
        
        def calc_hv_contributions(F_pop, ref):
            """
            Calculate the hypervolume contribution of each point in the population.
            """
            n = len(F_pop)
            contribs = np.zeros(n)
            for i in range(n):
                # Calculate HV of full set
                # Then HV of set without i
                # This is expensive if done naively every step.
                # Alternative: Just use non-dominated sorting.
                pass
            
            # Simpler: Return indices of non-dominated points
            non_dom_idx = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j: continue
                    if np.all(F_pop[j] <= F_pop[i]) and np.any(F_pop[j] < F_pop[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_idx.append(i)
            return non_dom_idx

        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Identify non-dominated points
            non_dom_idx = self._get_non_dominated(F)
            
            if len(non_dom_idx) == 0:
                # Fallback: pick best by sum
                parent_idx = np.argmin(np.sum(F, axis=1))
            else:
                # Pick a random non-dominated point to maintain diversity
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            
            parent = X[parent_idx]
            
            # Mutation
            mutation = rng.standard_normal(self.dim) * step_size
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Selection for replacement: Replace the point with the lowest "quality"
            # Quality: If dominated, it's bad. If non-dominated, use hypervolume contribution proxy.
            # Simple proxy: Replace the most dominated point, or if all non-dominated, replace the one with worst scalarized value.
            
            # Find the worst point to replace
            # 1. If child is non-dominated, it's good.
            # 2. We want to keep the best set.
            # Let's just replace the point that is dominated by the most others, or has the worst sum.
            
            # Calculate dominance count for each point in current pop
            dom_counts = np.zeros(pop_size)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom_counts[i] += 1
            
            # If child is non-dominated, it's better than any dominated point.
            # If child is dominated, it might still be better than some dominated points.
            # Simple strategy: Replace the point with the highest dom_count.
            # If tie, replace the one with worst sum of objectives.
            
            # Check if child is better than the worst current point
            worst_idx = np.argmax(dom_counts)
            
            # If the worst point is non-dominated (dom_count 0), we need a better criterion.
            # If dom_counts are all 0, replace the one with the largest sum (worst).
            if dom_counts[worst_idx] == 0:
                # All non-dominated (or similar). Replace the one with the largest sum (worst).
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
            
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Adaptive Step Size
            # If child is non-dominated, it's a good move.
            child_is_non_dom = False
            for j in range(pop_size):
                if j == worst_idx: continue
                if np.all(F[j] <= F[worst_idx]) and np.any(F[j] < F[worst_idx]):
                    child_is_non_dom = False
                    break
            else:
                child_is_non_dom = True
                
            if child_is_non_dom:
                step_size = min(max_step, step_size * 1.1)
            else:
                step_size = max(min_step, step_size * 0.9)
            
            # Track non-dominated points for potential local refinement
            if child_is_non_dom:
                best_non_dominated.append((child, fc))
        
        # Local refinement phase: Use L-BFGS-B to polish non-dominated points
        # This helps find the exact Pareto front by refining promising solutions
        if evals < self.budget and time.time() < deadline - 30:
            # Get unique non-dominated points from tracking
            unique_nd = []
            seen = set()
            for x, f in best_non_dominated:
                key = tuple(np.round(x, 6))
                if key not in seen:
                    seen.add(key)
                    unique_nd.append((x, f))
            
            # Sort by sum of objectives (better first)
            unique_nd.sort(key=lambda xf: np.sum(xf[1]))
            
            # Refine top candidates with L-BFGS-B
            max_refine = min(10, len(unique_nd))
            for i in range(max_refine):
                if evals >= self.budget or time.time() > deadline - 10:
                    break
                    
                x0, f0 = unique_nd[i]
                
                # Use weighted sum with different weights to explore different parts of Pareto front
                weights = [1.0 / self.n_obj] * self.n_obj
                # Vary weights slightly for diversity
                if i % 3 == 0:
                    weights[0] = 0.5
                    weights[1] = 0.3
                    weights[2] = 0.2
                elif i % 3 == 1:
                    weights[0] = 0.2
                    weights[1] = 0.5
                    weights[2] = 0.3
                else:
                    weights[0] = 0.3
                    weights[1] = 0.2
                    weights[2] = 0.5
                
                def objective(x):
                    nonlocal evals
                    if evals >= self.budget or time.time() > deadline - 5:
                        return np.inf
                    obj_vals = problem(x)
                    evals += 1
                    return np.dot(weights, obj_vals)
                
                # L-BFGS-B optimization
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 20, 'ftol': 1e-8}
                    )
                    if result.success and evals < self.budget:
                        x_opt = result.x
                        f_opt = problem(x_opt)
                        evals += 1
                        
                        # Check if this improved solution is non-dominated
                        is_nd = True
                        for j in range(pop_size):
                            if np.all(F[j] <= f_opt) and np.any(F[j] < f_opt):
                                is_nd = False
                                break
                        
                        if is_nd:
                            # Replace the worst non-dominated point
                            nd_idx = self._get_non_dominated(F)
                            if len(nd_idx) > 0:
                                # Find the one with worst sum
                                worst_nd = max(nd_idx, key=lambda idx: np.sum(F[idx]))
                                X[worst_nd] = x_opt
                                F[worst_nd] = f_opt
                except:
                    pass
        
        return X

    def _get_non_dominated(self, F):
        n = len(F)
        non_dom = []
        for i in range(n):
            is_dom = False
            for j in range(n):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom = True
                    break
            if not is_dom:
                non_dom.append(i)
        return non_dom
# EVOLVE-BLOCK-END
