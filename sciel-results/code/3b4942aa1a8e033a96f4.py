"""Multi-objective optimizer on WFG7: Tchebycheff decomposition with adaptive local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search using Tchebycheff decomposition with focused exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f_arr = np.array(f, dtype=np.float64)
            used += 1
            all_X.append(x.copy())
            all_F.append(f_arr)
            return f_arr

        # Phase 1: Generate Tchebycheff weight vectors (25 uniformly spaced on simplex)
        # For 3 objectives, we use 25 weight vectors
        n_weights = 25
        weights = []
        # Generate points on the 2-simplex (w1+w2+w3=1, wi>=0)
        for i in range(n_weights + 1):
            for j in range(n_weights - i + 1):
                k = n_weights - i - j
                if i + j + k == n_weights:
                    w = np.array([i, j, k], dtype=np.float64) / n_weights
                    # Only keep non-degenerate weights (all > 0) for better coverage
                    if np.all(w > 0.01):
                        weights.append(w)
        # If we don't have enough, add some
        if len(weights) < 10:
            weights = [np.array([1/3, 1/3, 1/3])]
            for _ in range(9):
                w = rng.dirichlet(np.ones(3))
                weights.append(w)
        weights = np.array(weights[:25])  # Cap at 25

        # Phase 2: Initial evaluation - evaluate diverse starting points
        # Use stratified sampling in the search space for better initial coverage
        n_init = min(30, budget - len(weights))  # Reserve some for local search
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            # Stratified sampling: divide each dimension into 2 parts
            x = np.zeros(dim)
            for d in range(dim):
                # Alternate between lower and upper half of each dimension
                frac = (i + 1) / (n_init + 1)
                x[d] = lo[d] + frac * span[d]
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # Also evaluate center
        if used < budget:
            evaluate(mid)

        # Phase 3: Tchebycheff-based local search for each weight
        # For each weight, do a focused local search minimizing the Tchebycheff scalarization
        remaining_budget = budget - used
        n_searches = len(weights)
        steps_per_search = max(2, remaining_budget // n_searches)
        steps_per_search = min(steps_per_search, 15)  # Cap to ensure we visit all weights

        # Sort weights by priority: prefer weights that give more balanced fronts
        # Use the current best points to estimate which weights are promising
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            # Compute Tchebycheff values for each weight
            best_f = np.min(F_arr, axis=0)
            worst_f = np.max(F_arr, axis=0)
            range_f = worst_f - best_f
            range_f[range_f < 1e-10] = 1.0
            
            weight_scores = []
            for w in weights:
                # Score based on how well this weight might cover the front
                # Weights closer to uniform are generally better for HV
                score = np.sum(np.abs(w - 1.0/n_obj))
                weight_scores.append(score)
            # Sort by score (lower is better - more balanced)
            sorted_indices = np.argsort(weight_scores)
            search_order = [sorted_indices[i] for i in range(len(sorted_indices))]
        else:
            search_order = list(range(len(weights)))

        for w_idx in search_order:
            if used >= budget or time.time() > deadline:
                break
            w = weights[w_idx]
            
            # Find the best point for this weight so far
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                # Tchebycheff scalarization: max_i w_i * (f_i - z_i^*)
                # Use the current best as reference
                best_f = np.min(F_arr, axis=0)
                # Compute Tchebycheff values
                tcheb_vals = np.max(w * (F_arr - best_f), axis=1)
                best_local_idx = np.argmin(tcheb_vals)
                center_x = all_X[best_local_idx].copy()
                center_f = all_F[best_local_idx].copy()
            else:
                center_x = mid.copy()
                center_f = None

            # Coordinate-descent style local search for Tchebycheff objective
            step_size = 0.15
            no_improve_rounds = 0
            max_no_improve = 4
            
            for step in range(steps_per_search):
                if used >= budget or time.time() > deadline:
                    break
                
                improved = False
                # Try coordinate directions (both positive and negative)
                for d in range(dim):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    for sign in [1.0, -1.0]:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        trial_x = center_x.copy()
                        trial_x[d] += sign * step_size * span[d]
                        trial_x = np.clip(trial_x, lo, hi)
                        
                        # Skip if no change
                        if np.array_equal(trial_x, center_x):
                            continue
                        
                        trial_f = evaluate(trial_x)
                        if trial_f is None:
                            break
                        
                        # Check if this improves the Tchebycheff value for weight w
                        if center_f is not None:
                            best_f = np.minimum(center_f, trial_f)
                            tcheb_center = np.max(w * (center_f - best_f))
                            tcheb_trial = np.max(w * (trial_f - best_f))
                            
                            if tcheb_trial < tcheb_center - 1e-10:
                                center_x = trial_x
                                center_f = trial_f
                                improved = True
                                break  # Move to next dimension
                        else:
                            center_x = trial_x
                            center_f = trial_f
                            improved = True
                            break
                
                if improved:
                    no_improve_rounds = 0
                    step_size *= 0.8
                else:
                    no_improve_rounds += 1
                    step_size *= 0.5
                    if no_improve_rounds >= max_no_improve:
                        break

        # Phase 4: Final exploitation - polish non-dominated points with fine coordinate search
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) > 0:
                # Select up to 6 diverse non-dominated points
                selected = []
                remaining = list(non_dom_idx)
                
                # Start with the point closest to the reference point [3, 5, 7]
                ref = np.array([3.0, 5.0, 7.0])
                dists = np.linalg.norm(F_arr - ref, axis=1)
                best_start = np.argmin(dists)
                if best_start in remaining:
                    selected.append(best_start)
                    remaining.remove(best_start)
                
                # Add diverse points in objective space
                max_selected = 6
                while len(selected) < max_selected and remaining:
                    if used >= budget or time.time() > deadline:
                        break
                    best_idx = -1
                    best_min_dist = -1
                    for idx in remaining:
                        f_point = F_arr[idx]
                        min_dist = np.min([np.linalg.norm(f_point - F_arr[s]) for s in selected])
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    if best_idx == -1:
                        break
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                
                # Polish each selected point with fine coordinate descent
                remaining_steps = budget - used
                steps_per_point = max(1, remaining_steps // len(selected))
                steps_per_point = min(steps_per_point, 10)
                
                for idx in selected:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.05
                    
                    for _ in range(steps_per_point):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        improved = False
                        for d in range(dim):
                            if used >= budget or time.time() > deadline:
                                break
                            for sign in [1.0, -1.0]:
                                if used >= budget or time.time() > deadline:
                                    break
                                trial_x = center_x.copy()
                                trial_x[d] += sign * step_size * span[d]
                                trial_x = np.clip(trial_x, lo, hi)
                                if np.array_equal(trial_x, center_x):
                                    continue
                                trial_f = evaluate(trial_x)
                                if trial_f is None:
                                    break
                                # Accept if it dominates current best
                                if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                                    center_x = trial_x
                                    center_f = trial_f
                                    improved = True
                                    break
                            if improved:
                                break
                        
                        if improved:
                            step_size *= 0.7
                        else:
                            step_size *= 0.5
                            if step_size < 0.005:
                                break

        # Return all evaluated points
        if len(all_F) > 0:
            return np.array(all_X)
        return np.array([])

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated points using efficient comparison."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        is_dominated = np.zeros(n, dtype=bool)

        # Efficient dominance check
        for i in range(n):
            if is_dominated[i]:
                continue
            # Check if any other point dominates i
            # F[j] <= F[i] for all objectives and F[j] != F[i]
            leq = np.all(F <= F[i], axis=1)
            strict = np.any(F < F[i], axis=1)
            leq[i] = False
            if np.any(leq & strict):
                is_dominated[i] = True
            else:
                # i is non-dominated; mark points dominated by i
                for j in range(n):
                    if j != i and not is_dominated[j]:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            is_dominated[j] = True

        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
