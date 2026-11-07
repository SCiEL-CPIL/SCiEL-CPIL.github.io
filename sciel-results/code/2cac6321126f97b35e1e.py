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

        # Phase 2: Initial evaluation - evaluate each weight vector's ideal point
        # For each weight, find a good starting point using a quick local search
        n_init_per_weight = 2
        for w in weights:
            if used >= budget or time.time() > deadline:
                break
            # Start from center with small perturbation
            for _ in range(n_init_per_weight):
                x = mid + 0.1 * span * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                evaluate(x)

        # Phase 3: Tchebycheff-based local search with coordinate descent for each weight
        remaining_budget = budget - used
        n_searches = len(weights)
        # Allocate more steps per search since coordinate descent is more efficient
        steps_per_search = max(2, remaining_budget // n_searches)
        steps_per_search = min(steps_per_search, 15)  # Cap to ensure we visit all weights

        # Sort weights by priority: prefer weights that give more balanced fronts
        if len(all_F) > 0:
            weight_scores = []
            for w in weights:
                # Weights closer to uniform are generally better for HV
                score = np.sum(np.abs(w - 1.0/n_obj))
                weight_scores.append(score)
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
                best_f = np.min(F_arr, axis=0)
                tcheb_vals = np.max(w * (F_arr - best_f), axis=1)
                best_local_idx = np.argmin(tcheb_vals)
                center_x = all_X[best_local_idx].copy()
                center_f = all_F[best_local_idx].copy()
            else:
                center_x = mid.copy()
                center_f = None

            # Coordinate descent with adaptive step size
            step_size = 0.05
            no_improve_count = 0
            max_no_improve = 5
            
            for step in range(steps_per_search):
                if used >= budget or time.time() > deadline:
                    break
                
                improved = False
                # Try coordinate directions (both positive and negative)
                for d in range(dim):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    for direction in [1, -1]:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        # Try a step in this coordinate
                        trial_x = center_x.copy()
                        trial_x[d] += direction * step_size * span[d]
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = evaluate(trial_x)
                        if trial_f is None:
                            break
                        
                        # Check if this improves the Tchebycheff value for weight w
                        if center_f is not None:
                            # Use soft-max for smoother gradient
                            beta = 10.0  # temperature for soft-max
                            tcheb_center = np.log(np.sum(np.exp(beta * w * (center_f - best_f)))) / beta
                            tcheb_trial = np.log(np.sum(np.exp(beta * w * (trial_f - best_f)))) / beta
                            
                            if tcheb_trial < tcheb_center - 1e-10:
                                center_x = trial_x
                                center_f = trial_f
                                improved = True
                        else:
                            center_x = trial_x
                            center_f = trial_f
                            improved = True
                
                if improved:
                    no_improve_count = 0
                    step_size *= 0.9
                else:
                    no_improve_count += 1
                    step_size *= 0.5
                    if no_improve_count >= max_no_improve:
                        break

        # Phase 4: Final high-precision polish of the best non-dominated points
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) > 0:
                # Select up to 7 diverse non-dominated points
                selected = []
                remaining = list(non_dom_idx)
                
                # Start with the point with best Tchebycheff value for uniform weight
                uniform_w = np.ones(n_obj) / n_obj
                best_f = np.min(F_arr, axis=0)
                # Use soft-max for smoother selection
                beta = 5.0
                tcheb_vals = np.log(np.sum(np.exp(beta * uniform_w * (F_arr - best_f)), axis=1)) / beta
                best_start = np.argmin(tcheb_vals)
                if best_start in remaining:
                    selected.append(best_start)
                    remaining.remove(best_start)
                
                # Add diverse points
                max_selected = 7
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
                
                # High-precision polish each selected point using coordinate descent
                remaining_steps = budget - used
                steps_per_point = max(2, remaining_steps // len(selected))
                steps_per_point = min(steps_per_point, 12)
                
                for idx in selected:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.02
                    no_improve = 0
                    max_no_improve = 4
                    
                    for _ in range(steps_per_point):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        improved = False
                        # Coordinate descent polish
                        for d in range(dim):
                            if used >= budget or time.time() > deadline:
                                break
                            for direction in [1, -1]:
                                if used >= budget or time.time() > deadline:
                                    break
                                
                                trial_x = center_x.copy()
                                trial_x[d] += direction * step_size * span[d]
                                trial_x = np.clip(trial_x, lo, hi)
                                trial_f = evaluate(trial_x)
                                if trial_f is None:
                                    break
                                
                                # Accept if it dominates center_f
                                if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                                    center_x = trial_x
                                    center_f = trial_f
                                    improved = True
                        
                        if improved:
                            no_improve = 0
                            step_size *= 0.85
                        else:
                            no_improve += 1
                            step_size *= 0.6
                            if no_improve >= max_no_improve:
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
