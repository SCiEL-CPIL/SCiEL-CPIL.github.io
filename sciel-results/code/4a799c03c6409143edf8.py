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

        def is_deadline():
            return time.time() > deadline

        # Generate Tchebycheff weight vectors on the 2-simplex
        n_weights = 25
        weights = []
        H = 5
        for i in range(H + 1):
            for j in range(H - i + 1):
                k = H - i - j
                w = np.array([i, j, k], dtype=np.float64) / H
                if np.all(w > 0.05):
                    weights.append(w)
        
        while len(weights) < n_weights:
            w = rng.dirichlet(np.ones(n_obj))
            if np.all(w > 0.05):
                weights.append(w)
        
        weights = np.array(weights[:n_weights])

        # Phase 1: Initial evaluations
        n_init = min(25, budget - n_weights)
        for _ in range(n_init):
            if used >= budget or is_deadline():
                break
            x = mid + 0.3 * span * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            evaluate(x)

        # Phase 2: For each weight, run adaptive local search on Tchebycheff scalarization
        if len(all_F) > 0:
            F_init = np.array(all_F)
            ref_point = np.min(F_init, axis=0)
        else:
            ref_point = np.array([1.0, 1.0, 1.0])

        # Sort weights: prioritize more balanced weights first
        weight_scores = []
        for w in weights:
            score = np.sum(np.abs(w - 1.0 / n_obj))
            weight_scores.append(score)
        sorted_indices = np.argsort(weight_scores)
        search_order = [sorted_indices[i] for i in range(len(sorted_indices))]

        # Adaptive local search parameters
        initial_step = 0.15  # Fraction of span
        step_decay = 0.7
        min_step = 0.005
        max_trials = 8

        for w_idx in search_order:
            if used >= budget or is_deadline():
                break
            w = weights[w_idx]
            
            # Find the best point for this weight as starting point
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                tcheb_vals = np.max(w * (F_arr - ref_point), axis=1)
                best_idx = np.argmin(tcheb_vals)
                start_x = all_X[best_idx].copy()
                start_f = all_F[best_idx].copy()
            else:
                start_x = mid.copy()
                start_f = None

            # Adaptive local search with restarts
            step_size = initial_step
            best_x = start_x.copy()
            best_f = start_f.copy() if start_f is not None else np.full(n_obj, np.inf)
            best_tcheb = np.max(w * (best_f - ref_point))
            
            # Try multiple restarts with decreasing step sizes
            n_restarts = min(4, (budget - used) // 10)
            for restart in range(n_restarts):
                if used >= budget or is_deadline():
                    break
                
                # For restarts > 0, perturb the best solution
                if restart > 0:
                    current_x = best_x + 0.1 * step_size * span * rng.standard_normal(dim)
                    current_x = np.clip(current_x, lo, hi)
                    current_f = evaluate(current_x)
                    if current_f is None:
                        break
                    current_tcheb = np.max(w * (current_f - ref_point))
                    if current_tcheb < best_tcheb:
                        best_x = current_x.copy()
                        best_f = current_f.copy()
                        best_tcheb = current_tcheb
                else:
                    current_x = start_x.copy()
                    current_f = start_f.copy() if start_f is not None else best_f.copy()
                    current_tcheb = best_tcheb
                
                # Local search iterations
                for _ in range(max_trials):
                    if used >= budget or is_deadline():
                        break
                    
                    # Generate trial points in a grid around current best
                    improved = False
                    for _trial in range(6):
                        if used >= budget or is_deadline():
                            break
                        
                        # Random perturbation
                        perturbation = step_size * span * rng.standard_normal(dim)
                        trial_x = np.clip(current_x + perturbation, lo, hi)
                        trial_f = evaluate(trial_x)
                        if trial_f is None:
                            break
                        
                        trial_tcheb = np.max(w * (trial_f - ref_point))
                        
                        if trial_tcheb < current_tcheb:
                            # Accept and move
                            current_x = trial_x.copy()
                            current_f = trial_f.copy()
                            current_tcheb = trial_tcheb
                            improved = True
                            
                            if trial_tcheb < best_tcheb:
                                best_x = trial_x.copy()
                                best_f = trial_f.copy()
                                best_tcheb = trial_tcheb
                            
                            # Reduce step size on improvement
                            step_size = max(min_step, step_size * step_decay)
                            break
                        else:
                            # No improvement, try another direction
                            step_size = max(min_step, step_size * 0.9)
                    
                    if not improved:
                        # No improvement in this round, reduce step more
                        step_size = max(min_step, step_size * 0.5)
                        if step_size <= min_step:
                            break

        # Phase 3: Final polish - refine best non-dominated points
        if used < budget and len(all_F) > 0:
            F_arr = np.array(all_F)
            non_dom_idx = self._get_non_dominated_indices(F_arr)
            
            if len(non_dom_idx) > 0:
                # Select up to 6 diverse non-dominated points for final polish
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
                max_selected = min(6, len(remaining_idx) + 1)
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
                steps_per_point = max(3, remaining_steps // len(selected))
                steps_per_point = min(steps_per_point, 20)
                
                for idx in selected:
                    if used >= budget or is_deadline():
                        break
                    center_x = all_X[idx].copy()
                    center_f = all_F[idx].copy()
                    step_size = 0.03
                    
                    for _ in range(steps_per_point):
                        if used >= budget or is_deadline():
                            break
                        improved = False
                        for _trial in range(4):
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
