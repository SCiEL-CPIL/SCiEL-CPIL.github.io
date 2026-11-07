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

        # Phase 1: Generate Tchebycheff weight vectors (30 uniformly spaced on simplex)
        n_weights = 30
        weights = []
        # Generate points on the 2-simplex (w1+w2+w3=1, wi>=0)
        for i in range(n_weights + 1):
            for j in range(n_weights - i + 1):
                k = n_weights - i - j
                if i + j + k == n_weights:
                    w = np.array([i, j, k], dtype=np.float64) / n_weights
                    # Only keep non-degenerate weights (all > 0) for better coverage
                    if np.all(w > 0.02):
                        weights.append(w)
        # If we don't have enough, add some
        if len(weights) < 10:
            weights = [np.array([1/3, 1/3, 1/3])]
            for _ in range(9):
                w = rng.dirichlet(np.ones(3))
                weights.append(w)
        weights = np.array(weights[:30])  # Cap at 30

        # Phase 2: Initial evaluation - evaluate center point once
        if used < budget and time.time() < deadline:
            x_center = mid.copy()
            evaluate(x_center)
        
        # Also evaluate a few diverse starting points for better initialization
        n_diverse = min(5, budget - used)
        for _ in range(n_diverse):
            if used >= budget or time.time() > deadline:
                break
            x = lo + span * rng.random(dim)
            evaluate(x)

        # Phase 3: Tchebycheff-based local search for each weight using Nelder-Mead
        remaining_budget = budget - used
        n_searches = len(weights)
        steps_per_search = max(1, remaining_budget // n_searches)
        steps_per_search = min(steps_per_search, 8)  # Cap to ensure we visit all weights

        # Sort weights by priority: prefer weights that give more balanced fronts
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            weight_scores = []
            for w in weights:
                # Score based on how well this weight might cover the front
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

            # Nelder-Mead simplex method for Tchebycheff subproblem
            # Initialize simplex around center_x
            simplex = [center_x.copy()]
            for d in range(min(dim, 4)):  # Limit simplex size for efficiency
                x_new = center_x.copy()
                x_new[d] += 0.05 * span[d]
                x_new = np.clip(x_new, lo, hi)
                simplex.append(x_new)
            
            # Evaluate initial simplex
            for i, x_s in enumerate(simplex):
                if used >= budget or time.time() > deadline:
                    break
                if i == 0:
                    f_s = center_f
                else:
                    f_s = evaluate(x_s)
                    if f_s is None:
                        break
                simplex[i] = (x_s, f_s)
            
            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5  # Standard NM parameters
            for nm_iter in range(steps_per_search):
                if used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by Tchebycheff value
                tcheb_vals = []
                for x_s, f_s in simplex:
                    if f_s is not None:
                        best_f_local = np.min([f_s] + [f for _, f in simplex if f is not None], axis=0)
                        tcheb_vals.append(np.max(w * (f_s - best_f_local)))
                    else:
                        tcheb_vals.append(np.inf)
                
                order = np.argsort(tcheb_vals)
                simplex = [simplex[i] for i in order]
                
                # Reflection
                centroid = np.mean([x_s for x_s, _ in simplex[:-1]], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1][0])
                x_r = np.clip(x_r, lo, hi)
                f_r = evaluate(x_r)
                if f_r is None:
                    break
                
                tcheb_r = np.max(w * (f_r - np.min([f_r] + [f for _, f in simplex if f is not None], axis=0)))
                tcheb_best = tcheb_vals[0]
                tcheb_worst = tcheb_vals[-1]
                
                if tcheb_r < tcheb_best:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = evaluate(x_e)
                    if f_e is not None:
                        tcheb_e = np.max(w * (f_e - np.min([f_e] + [f for _, f in simplex if f is not None], axis=0)))
                        if tcheb_e < tcheb_r:
                            simplex[-1] = (x_e, f_e)
                        else:
                            simplex[-1] = (x_r, f_r)
                    else:
                        simplex[-1] = (x_r, f_r)
                elif tcheb_r < tcheb_worst:
                    simplex[-1] = (x_r, f_r)
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1][0] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = evaluate(x_c)
                    if f_c is not None:
                        tcheb_c = np.max(w * (f_c - np.min([f_c] + [f for _, f in simplex if f is not None], axis=0)))
                        if tcheb_c < tcheb_worst:
                            simplex[-1] = (x_c, f_c)
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                x_sh = simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0])
                                x_sh = np.clip(x_sh, lo, hi)
                                f_sh = evaluate(x_sh)
                                if f_sh is not None:
                                    simplex[i] = (x_sh, f_sh)
                    else:
                        # Shrink if evaluation fails
                        for i in range(1, len(simplex)):
                            x_sh = simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0])
                            x_sh = np.clip(x_sh, lo, hi)
                            f_sh = evaluate(x_sh)
                            if f_sh is not None:
                                simplex[i] = (x_sh, f_sh)

        # Phase 4: Final exploitation of the best non-dominated points with tighter polish
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
                tcheb_vals = np.max(uniform_w * (F_arr - best_f), axis=1)
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
                
                # Polish each selected point with finer steps
                remaining_steps = budget - used
                steps_per_point = max(1, remaining_steps // len(selected))
                steps_per_point = min(steps_per_point, 6)
                
                for idx in selected:
                    if used >= budget or time.time() > deadline:
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.02
                    
                    for _ in range(steps_per_point):
                        if used >= budget or time.time() > deadline:
                            break
                        # Try multiple small perturbations
                        for _trial in range(3):
                            if used >= budget or time.time() > deadline:
                                break
                            perturbation = step_size * span * rng.standard_normal(dim)
                            trial_x = np.clip(center_x + perturbation, lo, hi)
                            trial_f = evaluate(trial_x)
                            if trial_f is None:
                                break
                            # Accept if it dominates or is non-dominated
                            if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                                center_x = trial_x
                                center_f = trial_f
                                step_size *= 0.7
                                break
                        else:
                            step_size *= 0.6

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
