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
        range_ = hi - lo

        all_X = []
        all_F = []
        eval_count = 0
        ref_point = np.array([3000.0, 0.06])

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget or time.time() > deadline:
                return None
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if not np.all(np.isfinite(f)):
                    return None
                all_X.append(np.array(x, dtype=float))
                all_F.append(f)
                eval_count += 1
                return f
            except Exception:
                return None

        def get_pareto_front_indices_2d(F):
            """Returns indices of non-dominated points in F, sorted by f1."""
            if len(F) == 0:
                return np.array([], dtype=int)
            sorted_idx = np.argsort(F[:, 0])
            F_sorted = F[sorted_idx]
            min_f2 = np.inf
            nd_idx = []
            for i in range(len(F_sorted)):
                if F_sorted[i, 1] < min_f2:
                    nd_idx.append(sorted_idx[i])
                    min_f2 = F_sorted[i, 1]
            return np.array(nd_idx, dtype=int)

        def calc_hv_2d(F_list, ref):
            """Calculate exact 2D hypervolume."""
            if not F_list:
                return 0.0
            F = np.array(F_list)
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return 0.0
            F_valid = F[mask]
            nd_idx = get_pareto_front_indices_2d(F_valid)
            F_nd = F_valid[nd_idx]
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

        def hv_incremental_update(f_new, F_list, ref):
            """Quick check if adding f_new improves HV. Returns (improves, delta_hv)."""
            f1, f2 = f_new
            if f1 >= ref[0] or f2 >= ref[1]:
                return False, 0.0

            if not F_list:
                return True, (ref[0] - f1) * (ref[1] - f2)

            F_arr = np.array(F_list)
            # Mask for points valid w.r.t reference point
            mask_valid = (F_arr[:, 0] < ref[0]) & (F_arr[:, 1] < ref[1])
            F_valid = F_arr[mask_valid]
            
            if len(F_valid) == 0:
                return True, (ref[0] - f1) * (ref[1] - f2)

            # Check if new point is dominated by any existing point
            # Dominance: x <= y and x != y
            dominated = (F_valid[:, 0] <= f1) & (F_valid[:, 1] <= f2) & \
                        ((F_valid[:, 0] < f1) | (F_valid[:, 1] < f2))
            if np.any(dominated):
                return False, 0.0

            # Get Pareto front of existing points
            nd_idx = get_pareto_front_indices_2d(F_valid)
            F_nd = F_valid[nd_idx]

            # Find the "gap" the new point fills
            # We need the first point on the front with f1 > f1_new AND f2 < f2_new
            # Actually, since it's not dominated, for any point with f1' > f1_new, we must have f2' > f2_new?
            # No. If point A has f1_A > f1_new, and f2_A < f2_new, then A dominates New?
            # Wait. Dominance: A <= New and A != New.
            # If f1_A > f1_new, A cannot dominate New in the first objective.
            # So if f1_A > f1_new and f2_A < f2_new, they are non-dominated.
            # The new point improves HV if it expands the region.
            
            # The HV contribution of a point on the front is (next_f1 - f1) * (ref_f2 - f2).
            # When adding a new point, it cuts into the region previously covered by points to its right (higher f1) 
            # and potentially points to its left (lower f1) if it has a lower f2?
            # No, non-dominated front is monotonic.
            
            # Let's find the index in F_nd where the new point would be inserted.
            # Sort F_nd by f1 (it already is).
            # Find right_bound: min f1 among points in F_nd with f1 > f1_new.
            # If no such point, right_bound = ref[0].
            
            # However, we must check if the new point is "covered" by the front.
            # If the new point is non-dominated, it adds area.
            # The area added is the rectangle from f1_new to right_bound with height (ref[1] - f2_new),
            # MINUS the area of any existing front points that fall within this new rectangle and were previously contributing to HV.
            
            # Actually, simpler logic:
            # The new point dominates no existing points (we checked that).
            # So it adds area to the right of its f1 and below its f2.
            # The "right" limit is the first point on the front with f1 > f1_new.
            # The "left" limit is f1_new.
            # The "top" limit is f2_new.
            # The "bottom" limit is ref[1].
            
            # But we must subtract the area of any existing points that are in [f1_new, right_bound) x [f2_new, ref[1]).
            # Wait, existing points in that region would have f2 >= f2_new?
            # If an existing point p has f1_p > f1_new and f2_p < f2_new, then p is to the right and lower.
            # Does p dominate new? No (f1_p > f1_new).
            # Does new dominate p? No (f2_new > f2_p).
            # So they are non-dominated.
            # The new point adds a rectangle [f1_new, f1_p] x [f2_new, ref[1]]?
            # No, the front is a curve.
            
            # Let's just recalculate the HV contribution of the new front vs old front locally?
            # Too expensive.
            
            # Alternative:
            # delta_hv = Area covered by new point that was NOT covered by old front.
            # Old front covers region R_old. New front covers R_new.
            # R_new = R_old U Rectangle(new).
            # Area(R_new) - Area(R_old) = Area( Rectangle(new) \ R_old ).
            
            # Rectangle(new) is [f1_new, ref[0]] x [f2_new, ref[1]].
            # We need to subtract the part of this rectangle that is already in R_old.
            # R_old is defined by the existing front.
            
            # Let's find the intersection of the rectangle [f1_new, ref[0]] x [f2_new, ref[1]] with R_old.
            # R_old is the union of rectangles for each point in F_nd.
            # This is complex.
            
            # Simpler approach for 2D:
            # The new point improves HV if and only if it is not dominated (already checked) AND
            # the rectangle [f1_new, f1_next] x [f2_new, ref[1]] is not fully covered by the "shadow" of previous points?
            # Actually, if it's not dominated, it ALWAYS adds some positive area to the HV in 2D.
            # Is that true?
            # Yes. If a point is non-dominated with respect to the current set, it adds to the Pareto front.
            # The new Pareto front will have a larger or equal HV.
            # The only case it doesn't improve is if it's dominated.
            # Wait, what if it's non-dominated but inside the convex hull?
            # In 2D, any non-dominated point adds to the front.
            # The front is a step function. Adding a point to the front increases the area under the curve (towards ref).
            # So if it's non-dominated, it strictly improves HV (unless it's exactly on the line, which has measure 0).
            
            # So the check "is it dominated" is sufficient to determine "does it improve".
            # However, we need the *delta* for tracking.
            # If we don't need the exact delta for the algorithm logic (only boolean improves), we can just return True.
            # The current code uses `best_hv += hv_delta` but doesn't really use `best_hv` for decision making in the loop, 
            # except maybe for logging or final check. The DE loop uses `no_improve_count`.
            # So returning `True, 0.0` if non-dominated is safe for the control flow, 
            # but `calc_hv_2d` is used for final scoring by the evaluator, not by the optimizer's internal logic for selection 
            # (selection uses dominance and random replacement).
            
            # Wait, `get_top_population_indices` uses `get_hv_contributions`.
            # So we need accurate HV contributions for selection.
            # But `hv_incremental_update` is only used to update `no_improve_count`.
            # So we can simplify: if non-dominated, it improves.
            
            return True, 0.0

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
            
            nd_idx = get_pareto_front_indices_2d(F_valid)
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

        # --- Phase 1: Initialization ---
        init_points = []
        # Add corners
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + range_[0]*i,
                            lo[1] + range_[1]*j,
                            lo[2] + range_[2]*k,
                            lo[3] + range_[3]*l
                        ])
                        init_points.append(x)
        
        # Add stratified random points biased towards 'lo'
        n_init = 50
        n_rand = n_init - len(init_points)
        if n_rand > 0:
            r = np.random.random((n_rand, self.dim))
            r = r ** 2.0 # Bias towards 0
            rand_points = lo + range_ * r
            init_points.extend(rand_points)
        
        for x in init_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # --- Phase 2: DE Loop ---
        pop_size = 40
        F_mut = 0.8
        CR = 0.9
        no_improve = 0
        
        def get_population():
            if len(all_F) < pop_size:
                return list(range(len(all_F)))
            
            F_arr = np.array(all_F)
            contribs = get_hv_contributions(all_F, ref_point)
            
            if not contribs:
                return np.random.choice(len(all_F), pop_size, replace=False).tolist()
            
            sorted_indices = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
            selected = sorted_indices[:pop_size]
            
            if len(selected) < pop_size:
                remaining = pop_size - len(selected)
                available = [i for i in range(len(all_F)) if i not in selected]
                if available:
                    rand_fill = np.random.choice(available, min(remaining, len(available)), replace=False)
                    selected.extend(rand_fill.tolist())
            
            return selected

        pop_indices = get_population()
        de_limit = eval_count + int((self.budget - eval_count) * 0.7)

        while eval_count < de_limit and time.time() < deadline:
            # Refresh population every 10 iterations
            if eval_count % 10 == 0:
                pop_indices = get_population()
                
            if len(pop_indices) < 3:
                break

            # Adaptive parameters
            if no_improve > 10:
                F_mut = max(0.3, F_mut * 0.9)
                CR = min(0.95, CR * 1.05)
                no_improve = 0

            # Pick parents
            parents = np.random.choice(pop_indices, 3, replace=False)
            x1 = all_X[parents[0]]
            x2 = all_X[parents[1]]
            x3 = all_X[parents[2]]

            # Mutation
            v = x1 + F_mut * (x2 - x3)
            
            # Crossover
            u = np.random.random(self.dim) < CR
            y = np.where(u, v, x1)
            
            # Clip
            y = np.clip(y, lo, hi)

            fy = evaluate(y)
            if fy is None:
                break

            # Check improvement
            improves, _ = hv_incremental_update(fy, all_F, ref_point)
            if improves:
                no_improve = 0
            else:
                no_improve += 1

            # Selection
            if np.random.random() < 0.5: # Replace half the time if non-dominated or random
                # Replace a random member from the population
                target_idx = np.random.choice(pop_indices)
                ft = all_F[target_idx]
                
                # Simple dominance check
                new_dom = fy[0] <= ft[0] and fy[1] <= ft[1] and (fy[0] < ft[0] or fy[1] < ft[1])
                old_dom = ft[0] <= fy[0] and ft[1] <= fy[1] and (ft[0] < fy[0] or ft[1] < fy[1])
                
                if new_dom or (not old_dom and np.random.random() < 0.3):
                    # Find index of target_idx in pop_indices
                    try:
                        pos = pop_indices.index(target_idx)
                        pop_indices[pos] = eval_count - 1
                    except ValueError:
                        pass

        # --- Phase 3: Local Search Polish ---
        if eval_count < self.budget and time.time() < deadline:
            contribs = get_hv_contributions(all_F, ref_point)
            if contribs:
                top_candidates = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
                top_candidates = top_candidates[:10]
                
                remaining = self.budget - eval_count
                steps_per_cand = max(2, remaining // len(top_candidates))
                
                for idx in top_candidates:
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    x_cur = all_X[idx].copy()
                    f_cur = all_F[idx].copy()
                    
                    # Coordinate descent
                    step_sizes = range_ * np.array([0.1, 0.05, 0.01])
                    
                    for step_size in step_sizes:
                        improved = False
                        for _ in range(steps_per_cand):
                            if eval_count >= self.budget or time.time() > deadline:
                                break
                            
                            for d in range(self.dim):
                                if eval_count >= self.budget or time.time() > deadline:
                                    break
                                
                                for sign in [1, -1]:
                                    x_new = x_cur.copy()
                                    x_new[d] += sign * step_size[d]
                                    x_new = np.clip(x_new, lo, hi)
                                    
                                    if np.allclose(x_new, x_cur, atol=1e-10):
                                        continue
                                    
                                    f_new = evaluate(x_new)
                                    if f_new is None:
                                        continue
                                    
                                    # Accept if it dominates current
                                    dom = f_new[0] <= f_cur[0] and f_new[1] <= f_cur[1] and \
                                          (f_new[0] < f_cur[0] or f_new[1] < f_cur[1])
                                    
                                    # Or if it's non-dominated and adds to HV
                                    if not dom:
                                        imp, _ = hv_incremental_update(f_new, all_F, ref_point)
                                        if imp:
                                            dom = True # Treat as accept
                                    
                                    if dom:
                                        x_cur = x_new
                                        f_cur = f_new
                                        improved = True
                                        break
                                if improved:
                                    break
                            if not improved:
                                break
                        if not improved:
                            break

        # --- Phase 4: Scipy Polish ---
        if eval_count < self.budget and time.time() < deadline:
            contribs = get_hv_contributions(all_F, ref_point)
            if contribs:
                top_candidates = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
                top_candidates = top_candidates[:5]
                
                for idx in top_candidates:
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    x0 = all_X[idx].copy()
                    
                    def objective(x):
                        if eval_count >= self.budget or time.time() > deadline:
                            return 1e10
                        try:
                            f = problem(x)
                            f = np.array(f, dtype=float)
                            if not np.all(np.isfinite(f)):
                                return 1e10
                            # Soft max
                            beta = 100.0
                            f_max = (1.0/beta) * np.log(np.sum(np.exp(beta * f)))
                            return f_max
                        except Exception:
                            return 1e10

                    try:
                        result = minimize(
                            objective, x0, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 20, 'ftol': 1e-10}
                        )
                        if result.success or result.fun < 1e9:
                            f_result = evaluate(result.x)
                            if f_result is not None:
                                # Check if it improved
                                improves, _ = hv_incremental_update(f_result, all_F, ref_point)
                    except Exception:
                        pass

        return all_X
