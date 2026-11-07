"""Multi-objective optimizer on WFG7: Tchebycheff decomposition with Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search using Tchebycheff decomposition with Nelder-Mead local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 330s limit

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

        def is_deadline():
            return time.time() > deadline

        # Generate Tchebycheff weight vectors on the 2-simplex
        # Use a grid of 30 points for good coverage
        n_weights = 30
        weights = []
        # Systematic generation on 2-simplex with H=4 gives 15 points, H=5 gives 21
        # Use H=5 for 21 points, plus some extras
        H = 5
        for i in range(H + 1):
            for j in range(H - i + 1):
                k = H - i - j
                w = np.array([i, j, k], dtype=np.float64) / H
                if np.all(w > 0.05):  # Avoid degenerate weights
                    weights.append(w)
        
        # Add a few more diverse points if needed
        while len(weights) < n_weights:
            w = rng.dirichlet(np.ones(n_obj))
            if np.all(w > 0.05):
                weights.append(w)
        
        weights = np.array(weights[:n_weights])

        # Phase 1: Initial evaluations - evaluate a diverse set of starting points
        n_init = min(20, budget - n_weights)
        for _ in range(n_init):
            if used >= budget or is_deadline():
                break
            x = mid + 0.3 * span * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            evaluate(x)

        # Phase 2: For each weight, run Nelder-Mead on the Tchebycheff scalarization
        # First, establish a reference point from initial evaluations
        if len(all_F) > 0:
            F_init = np.array(all_F)
            ref_point = np.min(F_init, axis=0)
        else:
            ref_point = np.array([1.0, 1.0, 1.0])

        # Allocate budget for local searches
        remaining = budget - used
        n_searches = min(len(weights), remaining // 5)  # At least 5 evals per search
        n_searches = max(1, n_searches)

        # Sort weights: prioritize more balanced weights first
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            weight_scores = []
            for w in weights:
                # Prefer weights closer to uniform (better for hypervolume)
                score = np.sum(np.abs(w - 1.0 / n_obj))
                weight_scores.append(score)
            sorted_indices = np.argsort(weight_scores)
            search_order = [sorted_indices[i] for i in range(len(sorted_indices))]
        else:
            search_order = list(range(len(weights)))

        # Nelder-Mead parameters
        alpha = 0.5      # Reflection
        gamma = 2.0      # Expansion
        rho = 0.5        # Contraction
        sigma = 0.5      # Shrink
        initial_step = 0.1  # Fraction of span for initial simplex

        for w_idx in search_order:
            if used >= budget or is_deadline():
                break
            w = weights[w_idx]
            
            # Find the best point for this weight as starting point
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                # Tchebycheff: max_i w_i * (f_i - ref_i)
                tcheb_vals = np.max(w * (F_arr - ref_point), axis=1)
                best_idx = np.argmin(tcheb_vals)
                start_x = all_X[best_idx].copy()
            else:
                start_x = mid.copy()

            # Nelder-Mead local search
            # Create initial simplex
            simplex = [start_x.copy()]
            for i in range(dim):
                trial = start_x.copy()
                trial[i] += initial_step * span[i]
                trial = np.clip(trial, lo, hi)
                simplex.append(trial)
            
            # Evaluate simplex vertices
            simplex_F = []
            for sx in simplex:
                if used >= budget or is_deadline():
                    break
                f = evaluate(sx)
                if f is None:
                    break
                simplex_F.append(f)
            
            if len(simplex_F) < dim + 1:
                continue

            # Tchebycheff objective function
            def tcheb_obj(x):
                f = np.array([problem(np.clip(x, lo, hi))])
                return np.max(w * (f[0] - ref_point))

            # Nelder-Mead iterations
            max_iter = min(30, (budget - used) // (dim + 1))
            for _iter in range(max_iter):
                if used >= budget or is_deadline():
                    break
                
                # Sort simplex by Tchebycheff value
                tcheb_vals = np.array([np.max(w * (f - ref_point)) for f in simplex_F])
                order = np.argsort(tcheb_vals)
                simplex = [simplex[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                tcheb_vals = tcheb_vals[order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = evaluate(x_ref)
                if f_ref is None:
                    break
                tcheb_ref = np.max(w * (f_ref - ref_point))

                if tcheb_ref < tcheb_vals[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = evaluate(x_exp)
                    if f_exp is None:
                        break
                    tcheb_exp = np.max(w * (f_exp - ref_point))
                    if tcheb_exp < tcheb_ref:
                        simplex[-1] = x_exp
                        simplex_F[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_F[-1] = f_ref
                elif tcheb_ref < tcheb_vals[-1]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_F[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = evaluate(x_con)
                    if f_con is None:
                        break
                    tcheb_con = np.max(w * (f_con - ref_point))
                    if tcheb_con < tcheb_vals[-1]:
                        simplex[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_sh = evaluate(simplex[i])
                            if f_sh is None:
                                break
                            simplex_F[i] = f_sh
                        if used >= budget or is_deadline():
                            break

        # Phase 3: Final polish - refine best non-dominated points
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) > 0:
                # Select up to 5 diverse non-dominated points for final polish
                selected = []
                remaining_idx = list(non_dom_idx)
                
                # Start with the point closest to the ideal point
                ideal = np.min(F_arr, axis=0)
                dists = np.linalg.norm(F_arr - ideal, axis=1)
                best_start = np.argmin(dists[remaining_idx])
                best_start = remaining_idx[best_start]
                selected.append(best_start)
                remaining_idx.remove(best_start)
                
                # Add diverse points
                max_selected = min(5, len(remaining_idx) + 1)
                while len(selected) < max_selected and remaining_idx:
                    if used >= budget or is_deadline():
                        break
                    best_idx = -1
                    best_min_dist = -1
                    for idx in remaining_idx:
                        f_point = F_arr[idx]
                        min_dist = np.min([np.linalg.norm(f_point - F_arr[s]) for s in selected])
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    if best_idx == -1:
                        break
                    selected.append(best_idx)
                    remaining_idx.remove(best_idx)
                
                # Polish each selected point with fine-grained search
                remaining_steps = budget - used
                steps_per_point = max(2, remaining_steps // len(selected))
                steps_per_point = min(steps_per_point, 15)
                
                for idx in selected:
                    if used >= budget or is_deadline():
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.02
                    
                    for _ in range(steps_per_point):
                        if used >= budget or is_deadline():
                            break
                        improved = False
                        for _trial in range(3):
                            if used >= budget or is_deadline():
                                break
                            perturbation = step_size * span * rng.standard_normal(dim)
                            trial_x = np.clip(center_x + perturbation, lo, hi)
                            trial_f = evaluate(trial_x)
                            if trial_f is None:
                                break
                            # Accept if it dominates or is non-dominated w.r.t. center
                            if np.all(trial_f <= center_f) and np.any(trial_f < center_f):
                                center_x = trial_x
                                center_f = trial_f
                                step_size *= 0.7
                                improved = True
                                break
                            elif not (np.all(center_f <= trial_f) and np.any(center_f < trial_f)):
                                # Non-dominated: keep both, but move center slightly
                                center_x = 0.5 * center_x + 0.5 * trial_x
                                step_size *= 0.8
                                improved = True
                                break
                        if not improved:
                            step_size *= 0.5

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

        for i in range(n):
            if is_dominated[i]:
                continue
            leq = np.all(F <= F[i], axis=1)
            strict = np.any(F < F[i], axis=1)
            leq[i] = False
            if np.any(leq & strict):
                is_dominated[i] = True
            else:
                for j in range(n):
                    if j != i and not is_dominated[j]:
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            is_dominated[j] = True

        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
