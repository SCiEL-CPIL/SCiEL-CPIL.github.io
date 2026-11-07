import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

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
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_pareto_front(F_list):
            if not F_list:
                return np.array([])
            F = np.array(F_list)
            n = len(F)
            if n == 0:
                return np.array([])
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def calc_hv_2d(F_list, ref):
            """Calculate exact 2D hypervolume."""
            if not F_list:
                return 0.0
            F = np.array(F_list)
            # Filter points strictly better than ref
            mask = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
            if not np.any(mask):
                return 0.0
            F = F[mask]
            if len(F) == 0:
                return 0.0
            
            # Get non-dominated subset
            nd_idx = get_pareto_front(F)
            F_nd = F[nd_idx]
            if len(F_nd) == 0:
                return 0.0
            
            # Sort by f1 ascending
            F_nd = F_nd[np.argsort(F_nd[:, 0])]
            
            hv = 0.0
            for i in range(len(F_nd)):
                f1 = F_nd[i, 0]
                f2 = F_nd[i, 1]
                # Next point's f1 or ref[0]
                next_f1 = F_nd[i+1, 0] if i + 1 < len(F_nd) else ref[0]
                # The width is (next_f1 - f1)
                # The height is (ref[1] - f2)
                width = next_f1 - f1
                height = ref[1] - f2
                hv += width * height
            return hv

        def hv_contribution_2d(point, nd_points_sorted, ref):
            """
            Calculate the hypervolume contribution of a specific point.
            nd_points_sorted must be sorted by f1 ascending and contain only non-dominated points.
            """
            p1, p2 = point
            if p1 >= ref[0] or p2 >= ref[1]:
                return 0.0
            
            # Find the position of this point in the sorted list
            # We assume point is in nd_points_sorted
            idx = -1
            for i in range(len(nd_points_sorted)):
                if np.allclose(nd_points_sorted[i], point):
                    idx = i
                    break
            if idx == -1:
                # If not found exactly (shouldn't happen), return 0
                return 0.0
            
            next_f1 = nd_points_sorted[idx+1, 0] if idx + 1 < len(nd_points_sorted) else ref[0]
            width = next_f1 - p1
            height = ref[1] - p2
            return width * height

        # --- Phase 1: Initialization ---
        # Grid initialization to cover the space
        n_grid_per_dim = 2
        grid_points = []
        if self.dim == 4:
            for i in range(n_grid_per_dim):
                for j in range(n_grid_per_dim):
                    for k in range(n_grid_per_dim):
                        for l in range(n_grid_per_dim):
                            x = np.array([
                                lo[0] + bounds_range[0] * (i / (n_grid_per_dim - 1)) if n_grid_per_dim > 1 else lo[0],
                                lo[1] + bounds_range[1] * (j / (n_grid_per_dim - 1)) if n_grid_per_dim > 1 else lo[1],
                                lo[2] + bounds_range[2] * (k / (n_grid_per_dim - 1)) if n_grid_per_dim > 1 else lo[2],
                                lo[3] + bounds_range[3] * (l / (n_grid_per_dim - 1)) if n_grid_per_dim > 1 else lo[3]
                            ])
                            grid_points.append(x)
        
        # Add random points to fill up to 60 evaluations
        n_init = 60
        n_random = n_init - len(grid_points)
        if n_random > 0:
            rand_points = lo + bounds_range * np.random.random((n_random, self.dim))
            grid_points.extend(rand_points)
        
        for x in grid_points:
            if time.time() > deadline or eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget:
            return all_X

        # --- Phase 2: Differential Evolution with HV-guided selection ---
        
        # Maintain an archive of non-dominated points
        # We will use the full set of evaluated points for parent selection based on HV contribution
        # but limit the population size for DE to ~50 to keep computation tractable
        
        pop_size = 40
        F_mutation = 0.8
        CR = 0.9
        no_improve_count = 0
        best_hv = calc_hv_2d(all_F, ref_point)
        
        # Initialize population indices from current evaluations
        # Pick the top pop_size points by HV contribution
        def get_top_population_indices(size):
            if len(all_F) < size:
                return list(range(len(all_F)))
            
            nd_idx = get_pareto_front(all_F)
            F_nd = np.array(all_F)[nd_idx]
            
            # Filter strictly better than ref
            mask = (F_nd[:, 0] < ref_point[0]) & (F_nd[:, 1] < ref_point[1])
            if not np.any(mask):
                # If no points are better than ref, just pick random
                return np.random.choice(len(all_F), size, replace=False).tolist()
            
            F_nd = F_nd[mask]
            # Map back to original indices is tricky, so let's just use the F values for sorting
            # We need to associate F values with all_F indices
            
            # Create a list of (hv_contrib, original_idx)
            # First, get non-dominated points in all_F
            nd_original_idx = []
            for i in nd_idx:
                f = all_F[i]
                if f[0] < ref_point[0] and f[1] < ref_point[1]:
                    nd_original_idx.append(i)
            
            if len(nd_original_idx) < size:
                # Add random points to fill up
                remaining = size - len(nd_original_idx)
                other_indices = [i for i in range(len(all_F)) if i not in nd_original_idx]
                if other_indices:
                    rand_fill = np.random.choice(other_indices, min(remaining, len(other_indices)), replace=False)
                    nd_original_idx.extend(rand_fill.tolist())
                return nd_original_idx[:size]
            
            # Calculate HV contribution for each candidate
            # To do this efficiently, we sort the non-dominated points by f1
            sorted_nd = sorted(nd_original_idx, key=lambda i: all_F[i][0])
            
            # For each point, find its contribution
            # This requires knowing the "next" point in the sorted non-dominated front
            # Let's create the sorted list of F values for these indices
            sorted_F = np.array([all_F[i] for i in sorted_nd])
            
            contribs = {}
            for k in range(len(sorted_nd)):
                idx = sorted_nd[k]
                p1, p2 = sorted_F[k]
                next_f1 = sorted_F[k+1, 0] if k + 1 < len(sorted_F) else ref_point[0]
                width = next_f1 - p1
                height = ref_point[1] - p2
                contribs[idx] = width * height
            
            # Sort indices by contribution descending
            sorted_indices = sorted(contribs.keys(), key=lambda i: contribs[i], reverse=True)
            return sorted_indices[:size]

        pop_indices = get_top_population_indices(pop_size)

        while eval_count < self.budget and time.time() < deadline:
            # Update population if it has shrunk or if we want to refresh
            if len(pop_indices) < pop_size:
                pop_indices = get_top_population_indices(pop_size)
            elif eval_count % 20 == 0:
                # Periodically refresh population with best HV points
                pop_indices = get_top_population_indices(pop_size)

            current_hv = calc_hv_2d(all_F, ref_point)
            if current_hv > best_hv + 1e-9:
                best_hv = current_hv
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > 10:
                # Reduce mutation factor to exploit
                F_mutation *= 0.9
                no_improve_count = 0
                if F_mutation < 0.1:
                    # Switch to local search mode or break
                    break

            # Perform one DE step per iteration
            if len(pop_indices) < 3:
                break

            # Pick 3 distinct parents
            parents = np.random.choice(pop_indices, 3, replace=False)
            x1 = all_X[parents[0]]
            x2 = all_X[parents[1]]
            x3 = all_X[parents[2]]

            # Mutation
            v = x1 + F_mutation * (x2 - x3)
            
            # Crossover
            u = np.random.random(self.dim) < CR
            y = np.where(u, v, x1)
            
            # Clip to bounds
            y = np.clip(y, lo, hi)

            fy = evaluate(y)
            if fy is None:
                break

            # Tournament selection against a random member of the population
            target_idx = np.random.choice(pop_indices)
            ft = all_F[target_idx]

            if dominates(fy, ft):
                # Replace target
                pop_indices[pop_indices.index(target_idx)] = eval_count - 1
            elif not dominates(ft, fy):
                # Non-dominated, replace with some probability
                if np.random.random() < 0.5:
                    pop_indices[pop_indices.index(target_idx)] = eval_count - 1
                # Alternatively, add to population if space allows
                # But we keep fixed size for simplicity

        return all_X
