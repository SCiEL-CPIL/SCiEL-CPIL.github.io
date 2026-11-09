# EVOLVE-BLOCK-START
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
        deadline = 320.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (given in task)
        ref_point = np.array([3.0, 5.0, 7.0])

        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        used = 0

        # Helper to evaluate a single point
        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f

        # Fast non-dominated sort (vectorized-friendly)
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].add(i)
                        n_dominated[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            for i in current_front:
                rank[i] = 0

            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        # Crowding distance
        def crowding_distance(F_arr, front_indices):
            if len(front_indices) == 0:
                return {}
            if len(front_indices) == 1:
                return {front_indices[0]: float('inf')}

            dist = {i: 0.0 for i in front_indices}
            n_objs = F_arr.shape[1]

            for m in range(n_objs):
                sorted_front = sorted(front_indices, key=lambda i: F_arr[i, m])
                dist[sorted_front[0]] = float('inf')
                dist[sorted_front[-1]] = float('inf')

                f_min = F_arr[sorted_front[0], m]
                f_max = F_arr[sorted_front[-1], m]
                if f_max == f_min:
                    continue

                for idx in range(1, len(sorted_front) - 1):
                    i = sorted_front[idx]
                    prev_i = sorted_front[idx - 1]
                    next_i = sorted_front[idx + 1]
                    dist[i] += (F_arr[next_i, m] - F_arr[prev_i, m]) / (f_max - f_min)

            return dist

        # Hypervolume calculation (2D projection approach for 3 objectives)
        def hypervolume_3d(points, ref):
            """Compute hypervolume of a set of 3D points dominated by ref."""
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than reference
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            
            # Convert to numpy
            pts = np.array(valid)
            
            # For 3 objectives, we can compute hypervolume using inclusion-exclusion
            # or by projecting onto 2D planes
            
            # Simple approach: sort by one objective, then compute 2D HV in the other two
            # Sort by objective 0
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            
            hv = 0.0
            prev_x = ref[0]
            
            for i in range(len(pts_sorted)):
                x, y, z = pts_sorted[i]
                # 2D hypervolume of points (y, z) dominated by (ref[1], ref[2])
                # with x being the current point's x value
                # Actually, we need to be more careful
                
            # Better: use a direct 3D hypervolume computation
            # For small populations, we can enumerate contributions
            
            # Filter to only points that can contribute (not dominated by others in the set)
            # Actually, let's just use a simple grid-based or analytical approach
            
            # For 3D, hypervolume = sum over each point of its contribution
            # A point contributes if it's not dominated by any other point in the set
            
            n = len(pts)
            hv = 0.0
            
            for i in range(n):
                pi = pts[i]
                # Check if pi is dominated by any other point
                dominated = False
                for j in range(n):
                    if j == i:
                        continue
                    pj = pts[j]
                    if np.all(pj <= pi) and np.any(pj < pi):
                        dominated = True
                        break
                if dominated:
                    continue
                
                # Compute the box dominated by pi and extending to ref
                # The contribution is the volume of the box [pi, ref] minus
                # volumes covered by other non-dominated points
                
                # Simple approximation: just take the box volume
                # This is an upper bound but works for relative comparison
                box_vol = (ref[0] - pi[0]) * (ref[1] - pi[1]) * (ref[2] - pi[2])
                
                # Subtract overlaps with other points
                # For simplicity, use a basic inclusion-exclusion for nearby points
                # Actually, let's use a proper 3D hypervolume algorithm
                
            # Use a proper 3D hypervolume computation
            # Sort by first objective
            pts_sorted = pts[np.argsort(pts[:, 0])]
            
            # Recursively compute
            def _hv_2d(points_2d, ref_2d):
                """2D hypervolume."""
                if len(points_2d) == 0:
                    return 0.0
                pts = np.array(points_2d)
                pts = pts[np.all(pts < ref_2d, axis=1)]
                if len(pts) == 0:
                    return 0.0
                # Sort by first coordinate
                pts = pts[np.argsort(pts[:, 0])]
                hv = 0.0
                max_y = 0.0
                for i in range(len(pts)):
                    x, y = pts[i]
                    if y > max_y:
                        hv += (ref_2d[0] - x) * (y - max_y)
                        max_y = y
                    else:
                        hv += (ref_2d[0] - x) * (ref_2d[1] - max_y) + (ref_2d[0] - x) * (max_y - y)
                # Fix: simpler 2D HV
                pts = pts[np.argsort(pts[:, 0])]
                hv = 0.0
                best_y = 0.0
                for i in range(len(pts)):
                    x, y = pts[i]
                    if y > best_y:
                        hv += (ref_2d[0] - x) * (y - best_y)
                        best_y = y
                hv += (ref_2d[0] - pts[-1, 0]) * (ref_2d[1] - best_y) if best_y < ref_2d[1] else 0
                return hv
            
            # For 3D: sort by obj 0, for each point compute 2D HV of the remaining objectives
            # where the "width" in obj 0 is the gap to the next point
            pts_sorted = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            for i in range(len(pts_sorted)):
                pi = pts_sorted[i]
                # Width in obj 0
                if i + 1 < len(pts_sorted):
                    width_0 = pts_sorted[i + 1, 0] - pi[0]
                else:
                    width_0 = ref[0] - pi[0]
                
                # 2D HV of points (obj1, obj2) for all points j >= i 
                # that are not dominated in obj1,obj2 by pi
                # Actually, the contribution of pi is width_0 * (2D HV of {pi[1:]} ∪ {pj[1:] for j > i} 
                # where the 2D reference is (ref[1], ref[2]))
                
                # Collect points j >= i
                sub_points = pts_sorted[i:, 1:]
                hv_2d = _hv_2d(sub_points, ref[1:])
                hv += width_0 * hv_2d
            
            return hv

        # Simplified hypervolume for selection (faster, approximate)
        def approximate_hv(F_arr, ref):
            """Approximate hypervolume for quick selection."""
            if len(F_arr) == 0:
                return 0.0
            valid = F_arr[np.all(F_arr < ref, axis=1)]
            if len(valid) == 0:
                return 0.0
            # Use sum of box volumes as a rough proxy
            total = 0.0
            for f in valid:
                total += (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            return total

        # Population size
        pop_size = 40
        if budget < 80:
            pop_size = max(10, budget // 5)
        pop_size = min(pop_size, budget)

        # Phase 1: Random initialization
        n_init = min(pop_size, budget)
        X_init = lo + (hi - lo) * np.random.random((n_init, dim))
        F_init = np.array([evaluate(x) for x in X_init])
        X = X_init
        F = F_init

        # Phase 2: Local refinement using Nelder-Mead on non-dominated points
        # Select top non-dominated points and refine them
        def local_refine(x0, max_evals=10):
            """Use Nelder-Mead to minimize a weighted sum of objectives."""
            # Create a scalar objective using weighted sum with random weights
            w = np.random.dirichlet(np.ones(n_obj))
            
            def scalar_obj(x):
                nonlocal used
                if used >= budget:
                    return 0.0
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                all_X.append(x.copy())
                all_F.append(f)
                used += 1
                return np.dot(w, f)
            
            try:
                result = minimize(
                    scalar_obj, x0, method='Nelder-Mead',
                    options={'maxiter': max_evals, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                return result.x
            except:
                return x0

        # Phase 3: Evolutionary loop with parent-child island dynamics
        eta_c = 15.0
        eta_m = 20.0
        p_c = 0.9
        p_m = 1.0 / dim

        def sbx_crossover(x1, x2):
            c1 = np.zeros_like(x1)
            c2 = np.zeros_like(x2)
            if np.random.rand() < p_c:
                for i in range(dim):
                    if np.random.rand() < 0.5:
                        if abs(x1[i] - x2[i]) < 1e-10:
                            c1[i] = x1[i]
                            c2[i] = x2[i]
                        else:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                            c1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                            c2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                    else:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
            else:
                c1 = x1.copy()
                c2 = x2.copy()
            return c1, c2

        def polynomial_mutation(x):
            x_mut = x.copy()
            for i in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                    xmut1 = x[i] + delta1 * (hi[i] - lo[i])
                    xmut2 = x[i] + delta2 * (hi[i] - lo[i])
                    if np.random.rand() < 0.5:
                        x_mut[i] = np.clip(xmut1, lo[i], hi[i])
                    else:
                        x_mut[i] = np.clip(xmut2, lo[i], hi[i])
            return x_mut

        def tournament_select(F_arr, ranks, crowding, size=3):
            n = len(F_arr)
            candidates = np.random.randint(0, n, size)
            best = candidates[0]
            best_rank = ranks[best]
            best_crowd = crowding.get(best, 0.0)
            for c in candidates[1:]:
                r = ranks[c]
                cr = crowding.get(c, 0.0)
                if r < best_rank or (r == best_rank and cr > best_crowd):
                    best = c
                    best_rank = r
                    best_crowd = cr
            return best

        # Main loop
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Compute ranks and crowding
            fronts, ranks = fast_non_dominated_sort(F)
            crowding = {}
            for front in fronts:
                cdist = crowding_distance(F, front)
                crowding.update(cdist)

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                if used >= budget:
                    break
                if time.time() - start_time > deadline:
                    break

                idx1 = tournament_select(F, ranks, crowding)
                idx2 = tournament_select(F, ranks, crowding)

                c1, c2 = sbx_crossover(X[idx1], X[idx2])
                c1 = polynomial_mutation(c1)

                # Evaluate
                f = evaluate(c1)
                offspring_X.append(c1)
                offspring_F.append(f)

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F)

            # Combine
            combined_X = np.vstack([X, np.array(offspring_X[:len(offspring_F)])])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            c_fronts, c_ranks = fast_non_dominated_sort(combined_F)

            next_X = []
            next_F = []

            for front in c_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    cdist = crowding_distance(combined_F, front)
                    remaining = pop_size - len(next_X)
                    sorted_front = sorted(front, key=lambda i: cdist.get(i, 0.0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Periodic local refinement on best points
            if len(F) > 0 and used < budget - 15:
                # Find non-dominated points
                fronts2, _ = fast_non_dominated_sort(F)
                if fronts2:
                    best_front = fronts2[0]
                    # Refine up to 3 points
                    n_refine = min(3, len(best_front), (budget - used) // 5)
                    for k in range(n_refine):
                        if used >= budget:
                            break
                        if time.time() - start_time > deadline:
                            break
                        idx = best_front[k % len(best_front)]
                        x_refined = local_refine(X[idx], max_evals=5)
                        # Evaluate refined point
                        f_ref = evaluate(x_refined)
                        # Replace if better in any objective
                        improved = False
                        for m in range(n_obj):
                            if f_ref[m] < F[idx, m]:
                                improved = True
                                break
                        if improved:
                            X[idx] = x_refined
                            F[idx] = f_ref

        # Final: record all points
        return np.array(all_X) if all_X else X
# EVOLVE-BLOCK-END
