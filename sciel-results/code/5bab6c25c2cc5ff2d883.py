import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        ref_point = np.array([3000.0, 0.06])

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if not np.all(np.isfinite(f)):
                    return None
                all_X.append(np.array(x, dtype=float))
                all_F.append(f.copy())
                eval_count += 1
                return f
            except Exception:
                return None

        def get_pareto_front_2d(F):
            """Efficient 2D non-dominated sorting. O(N log N)."""
            if len(F) == 0:
                return np.array([], dtype=int)
            sorted_idx = np.argsort(F[:, 0])
            F_sorted = F[sorted_idx]
            min_f2 = np.inf
            nd_indices = []
            for i in range(len(F_sorted)):
                if F_sorted[i, 1] < min_f2:
                    nd_indices.append(sorted_idx[i])
                    min_f2 = F_sorted[i, 1]
            return np.array(nd_indices, dtype=int)

        def calc_hv_2d(F_list, ref):
            """Calculate exact 2D hypervolume. Only considers points better than ref."""
            if not F_list:
                return 0.0
            F = np.array(F_list)
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return 0.0
            F = F[mask]
            nd_idx = get_pareto_front_2d(F)
            F_nd = F[nd_idx]
            if len(F_nd) == 0:
                return 0.0
            hv = 0.0
            for i in range(len(F_nd)):
                f1 = F_nd[i, 0]
                f2 = F_nd[i, 1]
                next_f1 = F_nd[i+1, 0] if i + 1 < len(F_nd) else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                hv += width * height
            return hv

        def get_hv_contributions(F_list, ref):
            """Calculate HV contribution for each point in the Pareto front."""
            if not F_list:
                return {}
            F = np.array(F_list)
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return {}
            F_valid = F[mask]
            valid_indices = np.where(mask)[0]
            
            nd_idx = get_pareto_front_2d(F_valid)
            F_nd = F_valid[nd_idx]
            
            contribs = {}
            for k in range(len(F_nd)):
                orig_idx = valid_indices[nd_idx[k]]
                f1 = F_nd[k, 0]
                f2 = F_nd[k, 1]
                next_f1 = F_nd[k+1, 0] if k + 1 < len(F_nd) else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                contribs[orig_idx] = width * height
            return contribs

        def hv_incremental_update(F_new, F_list, ref):
            """Quick check if adding F_new improves HV. Returns (improves, new_hv_delta)."""
            f1, f2 = F_new
            if f1 >= ref[0] or f2 >= ref[1]:
                return False, 0.0

            F_arr = np.array(F_list)
            mask_valid = (F_arr[:, 0] < ref[0]) & (F_arr[:, 1] < ref[1])
            F_valid = F_arr[mask_valid]

            if len(F_valid) == 0:
                return True, (ref[0] - f1) * (ref[1] - f2)

            # Check if new point is dominated by any existing point
            dominated = (F_valid[:, 0] <= f1) & (F_valid[:, 1] <= f2) & \
                        ((F_valid[:, 0] < f1) | (F_valid[:, 1] < f2))
            if np.any(dominated):
                return False, 0.0

            # Get Pareto front
            nd_idx = get_pareto_front_2d(F_valid)
            F_nd = F_valid[nd_idx]

            # Find the "gap" the new point fills
            f1_right = ref[0]
            for i in range(len(F_nd)):
                if F_nd[i, 0] > f1 and F_nd[i, 1] < f2:
                    f1_right = F_nd[i, 0]
                    break

            delta_hv = (f1_right - f1) * (ref[1] - f2)

            # Subtract area that was already covered by points now dominated
            covered = 0.0
            for i in range(len(F_nd)):
                if F_nd[i, 0] >= f1 and F_nd[i, 0] < f1_right and F_nd[i, 1] >= f2:
                    old_right = F_nd[i+1, 0] if i + 1 < len(F_nd) else ref[0]
                    old_width = old_right - F_nd[i, 0]
                    old_height = ref[1] - F_nd[i, 1]
                    old_contrib = old_width * old_height

                    new_width = max(0, old_right - f1_right)
                    new_height = old_height
                    new_contrib = new_width * new_height

                    covered += old_contrib - new_contrib

            net_delta = delta_hv - covered
            return net_delta > 1e-12, net_delta

        # --- Phase 1: Strategic Initialization ---
        init_points = []
        
        # Corners
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + bounds_range[0]*i,
                            lo[1] + bounds_range[1]*j,
                            lo[2] + bounds_range[2]*k,
                            lo[3] + bounds_range[3]*l
                        ])
                        init_points.append(x)
        
        # Center point
        init_points.append(0.5 * (lo + hi))
        
        # Stratified Random Points
        n_target_init = 60
        n_rand_needed = n_target_init - len(init_points)
        if n_rand_needed > 0:
            r = np.random.random((n_rand_needed, self.dim))
            # Bias towards lower values (assuming minimization benefits from lower inputs)
            r = r ** 1.5
            rand_points = lo + bounds_range * r
            init_points.extend(rand_points)
            
        for x in init_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # --- Phase 2: DE with HV-guided selection ---
        pop_size = 50
        F_mutation = 0.8
        CR = 0.9
        no_improve_count = 0
        
        def get_top_population_indices(size):
            if len(all_F) < size:
                return list(range(len(all_F)))
            
            F_arr = np.array(all_F)
            contribs = get_hv_contributions(all_F, ref_point)
            
            if not contribs:
                return np.random.choice(len(all_F), size, replace=False).tolist()
            
            # Sort indices by HV contribution
            sorted_indices = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
            selected = sorted_indices[:size]
            
            if len(selected) < size:
                remaining = size - len(selected)
                available = [i for i in range(len(all_F)) if i not in selected]
                if available:
                    rand_fill = np.random.choice(available, min(remaining, len(available)), replace=False)
                    selected.extend(rand_fill.tolist())
            
            return selected

        pop_indices = get_top_population_indices(pop_size)
        
        # --- Phase 3: DE Loop ---
        budget_for_de = int((self.budget - eval_count) * 0.6)
        de_limit = eval_count + budget_for_de
        
        while eval_count < de_limit and time.time() < deadline:
            # Refresh population periodically
            if len(pop_indices) < pop_size or eval_count % 10 == 0:
                pop_indices = get_top_population_indices(pop_size)
                
            if len(pop_indices) < 3:
                break

            # Adaptive parameters
            if no_improve_count > 10:
                F_mutation = max(0.3, F_mutation * 0.9)
                CR = min(0.95, CR * 1.02)
                no_improve_count = 0
            
            # Pick parents
            parents = np.random.choice(pop_indices, 3, replace=False)
            x1 = all_X[parents[0]]
            x2 = all_X[parents[1]]
            x3 = all_X[parents[2]]

            # Mutation
            v = x1 + F_mutation * (x2 - x3)
            
            # Crossover
            u = np.random.random(self.dim) < CR
            y = np.where(u, v, x1)
            
            # Clip
            y = np.clip(y, lo, hi)

            fy = evaluate(y)
            if fy is None:
                break

            # Quick HV improvement check
            improves, hv_delta = hv_incremental_update(fy, all_F, ref_point)
            if improves:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Selection: replace a random population member if new point is competitive
            target_idx = np.random.choice(pop_indices)
            ft = all_F[target_idx]
            
            # Dominance check
            new_dominates = fy[0] <= ft[0] and fy[1] <= ft[1] and (fy[0] < ft[0] or fy[1] < ft[1])
            old_dominates = ft[0] <= fy[0] and ft[1] <= fy[1] and (ft[0] < fy[0] or ft[1] < fy[1])
            
            if new_dominates:
                pop_indices[pop_indices.index(target_idx)] = eval_count - 1
            elif not old_dominates:
                # Non-dominated: replace with some probability based on HV contribution
                if np.random.random() < 0.5:
                    pop_indices[pop_indices.index(target_idx)] = eval_count - 1

        # --- Phase 4: Gradient-based Polish using differentiable HV surrogate ---
        # We model the bi-objective minimization as maximizing the exact 2D hypervolume contribution.
        # Since the exact contribution is non-differentiable (depends on neighbors), we use a smooth surrogate.
        # Surrogate: -max(f1, f2) + epsilon * (f1 + f2) is a common smooth approximation for Pareto front points.
        # However, a better surrogate for HV improvement in 2D is minimizing the "gap" to the reference point
        # or using a weighted sum that balances the objectives based on the current front spread.
        
        # A more robust approach for 2D:
        # We want to find points that are non-dominated and close to the reference point in a transformed space.
        # Let's use the "augmented Tchebycheff" method or simply optimize a smooth approximation of the HV contribution.
        
        # For a point x with objectives f1, f2, its potential HV contribution is roughly:
        # (ref1 - f1) * (ref2 - f2) if it's the best in both, but it's more complex.
        
        # Let's try optimizing a smooth surrogate:
        # Objective: minimize f1 * f2 (product) or f1 + f2 (sum) or max(f1, f2).
        # Minimizing the product f1*f2 tends to find points with balanced low objectives.
        # Minimizing the sum f1+f2 tends to find the "knee" of the Pareto front.
        
        # We will use a combination: minimize (f1 + f2) + lambda * (f1 - f2)^2 to encourage balance?
        # Or simply use the "soft-min" of the objectives relative to the reference?
        
        # Let's stick to a robust multi-start local search using L-BFGS-B on a weighted sum.
        # We will run this for the top K candidates from the archive.
        
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            contribs = get_hv_contributions(all_F, ref_point)

            if contribs:
                top_candidates = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
                top_candidates = top_candidates[:10]

                remaining_budget = self.budget - eval_count
                # Allocate budget for polish
                # Each L-BFGS-B call might take several evaluations.
                # We limit the number of polish attempts.
                max_polish_attempts = min(len(top_candidates), remaining_budget // 5)
                
                for idx in top_candidates[:max_polish_attempts]:
                    if eval_count >= self.budget or time.time() > deadline:
                        break

                    x0 = all_X[idx].copy()
                    
                    # Define the surrogate objective
                    # We minimize a weighted sum of the objectives.
                    # To explore different parts of the front, we could vary weights, 
                    # but since we start from diverse top candidates, a fixed weight or adaptive weight is fine.
                    # Let's use equal weights for now, as it's a good default for 2D.
                    
                    def objective(x):
                        if eval_count >= self.budget or time.time() > deadline:
                            return 1e10
                        try:
                            f = problem(x)
                            f = np.array(f, dtype=float)
                            if not np.all(np.isfinite(f)):
                                return 1e10
                            # Smooth max approximation to encourage both objectives to be low
                            # beta controls the sharpness
                            beta = 20.0
                            f_max = (1.0/beta) * np.log(np.sum(np.exp(beta * f)))
                            # Also add a small term for the sum to avoid extreme imbalance
                            f_sum = np.sum(f)
                            return 0.8 * f_max + 0.2 * f_sum
                        except Exception:
                            return 1e10

                    # Bounds for scipy
                    scipy_bounds = list(zip(lo, hi))

                    try:
                        result = minimize(
                            objective, x0, method='L-BFGS-B',
                            bounds=scipy_bounds,
                            options={'maxiter': 20, 'ftol': 1e-12}
                        )
                        if result.success or result.fun < 1e9:
                            f_result = evaluate(result.x)
                            if f_result is not None:
                                # Check if this improves the front
                                improves, _ = hv_incremental_update(f_result, all_F, ref_point)
                    except Exception:
                        pass

        # --- Phase 5: Final fine-tuning with small perturbations ---
        # If we have budget left, do a few random small steps from the best points.
        if eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            contribs = get_hv_contributions(all_F, ref_point)
            if contribs:
                top_candidates = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
                top_candidates = top_candidates[:5]
                
                for idx in top_candidates:
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    x0 = all_X[idx].copy()
                    # Small random perturbations
                    for _ in range(3):
                        if eval_count >= self.budget or time.time() > deadline:
                            break
                        step = np.random.normal(0, 0.01 * bounds_range)
                        x_new = np.clip(x0 + step, lo, hi)
                        f_new = evaluate(x_new)
                        if f_new is not None:
                            improves, _ = hv_incremental_update(f_new, all_F, ref_point)
                            if improves:
                                x0 = x_new

        return all_X
