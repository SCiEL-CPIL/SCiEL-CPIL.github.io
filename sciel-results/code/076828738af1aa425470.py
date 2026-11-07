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
                    if np.all(w > 0.01):
                        weights.append(w)
        # If we don't have enough, add some
        if len(weights) < 10:
            weights = [np.array([1/3, 1/3, 1/3])]
            for _ in range(9):
                w = rng.dirichlet(np.ones(3))
                weights.append(w)
        weights = np.array(weights[:30])  # Cap at 30

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
        # Use smooth Tchebycheff scalarization (log-sum-exp approximation of max)
        # with Nelder-Mead local search per weight vector
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

        def tchebycheff_smooth(x, w, ref_f):
            """Smooth Tchebycheff scalarization using log-sum-exp."""
            f = problem(np.clip(x, lo, hi))
            f_arr = np.array(f, dtype=np.float64)
            # Soft-max approximation: max_i w_i * (f_i - ref_i)
            # Use log-sum-exp for numerical stability
            vals = w * (f_arr - ref_f)
            # Soft-max with temperature
            temp = 0.1
            max_val = np.max(vals)
            if max_val > 1e10:
                return max_val
            exp_vals = np.exp((vals - max_val) / temp)
            soft_max = max_val + temp * np.log(np.sum(exp_vals))
            return soft_max

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

            # Nelder-Mead style local search for smooth Tchebycheff objective
            # Initialize simplex
            simplex = [center_x.copy()]
            for d in range(dim - 1):
                x_new = center_x.copy()
                step = 0.1 * span[d]
                if x_new[d] + step > hi[d]:
                    x_new[d] = x_new[d] - step
                else:
                    x_new[d] = x_new[d] + step
                simplex.append(x_new)

            # Evaluate initial simplex
            simplex_f = []
            for sx in simplex:
                if used >= budget or time.time() > deadline:
                    break
                f = evaluate(sx)
                if f is not None:
                    simplex_f.append(f)
                else:
                    simplex_f.append(np.full(n_obj, np.inf))

            if len(simplex_f) < dim:
                continue

            # Reference for Tchebycheff
            if center_f is not None:
                ref_f = center_f.copy()
            else:
                ref_f = np.min(np.array(simplex_f), axis=0)

            # Nelder-Mead iterations
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            for it in range(steps_per_search):
                if used >= budget or time.time() > deadline:
                    break

                # Sort simplex by Tchebycheff value
                tcheb_vals = []
                for f in simplex_f:
                    vals = w * (f - ref_f)
                    max_val = np.max(vals)
                    if max_val > 1e10:
                        tcheb_vals.append(max_val)
                    else:
                        exp_vals = np.exp((vals - max_val) / 0.1)
                        soft_max = max_val + 0.1 * np.log(np.sum(exp_vals))
                        tcheb_vals.append(soft_max)

                order = np.argsort(tcheb_vals)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                tcheb_vals = [tcheb_vals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, lo, hi)

                if used >= budget or time.time() > deadline:
                    break
                f_reflect = evaluate(x_reflect)
                if f_reflect is None:
                    break

                vals_reflect = w * (f_reflect - ref_f)
                max_val_r = np.max(vals_reflect)
                if max_val_r > 1e10:
                    tcheb_reflect = max_val_r
                else:
                    exp_vals_r = np.exp((vals_reflect - max_val_r) / 0.1)
                    tcheb_reflect = max_val_r + 0.1 * np.log(np.sum(exp_vals_r))

                # Check if reflection improved
                if tcheb_reflect < tcheb_vals[-2]:  # Better than second worst
                    # Try expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, lo, hi)

                    if used >= budget or time.time() > deadline:
                        break
                    f_expand = evaluate(x_expand)
                    if f_expand is None:
                        break

                    vals_expand = w * (f_expand - ref_f)
                    max_val_e = np.max(vals_expand)
                    if max_val_e > 1e10:
                        tcheb_expand = max_val_e
                    else:
                        exp_vals_e = np.exp((vals_expand - max_val_e) / 0.1)
                        tcheb_expand = max_val_e + 0.1 * np.log(np.sum(exp_vals_e))

                    if tcheb_expand < tcheb_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif tcheb_reflect < tcheb_vals[-1]:  # Better than worst
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, lo, hi)

                    if used >= budget or time.time() > deadline:
                        break
                    f_contract = evaluate(x_contract)
                    if f_contract is None:
                        break

                    vals_contract = w * (f_contract - ref_f)
                    max_val_c = np.max(vals_contract)
                    if max_val_c > 1e10:
                        tcheb_contract = max_val_c
                    else:
                        exp_vals_c = np.exp((vals_contract - max_val_c) / 0.1)
                        tcheb_contract = max_val_c + 0.1 * np.log(np.sum(exp_vals_c))

                    if tcheb_contract < tcheb_vals[-1]:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, dim):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if used >= budget or time.time() > deadline:
                                break
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                simplex[i] = simplex[0]
                                break
                            simplex_f[i] = f_shrink

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
