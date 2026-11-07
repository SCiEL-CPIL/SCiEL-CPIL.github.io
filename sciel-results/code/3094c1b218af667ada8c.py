import numpy as np
import time

class Optimizer:
    """
    Hypervolume-guided multi-objective optimizer for RE21.
    Uses stratified initialization + adaptive differential evolution with
    hypervolume-based selection and local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

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
                all_X.append(x.copy())
                all_F.append(f)
                eval_count += 1
                return f
            except Exception:
                return None

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_non_dominated_indices():
            if not all_F:
                return []
            F = np.array(all_F)
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        def hv_contribution(point, front):
            """Compute hypervolume contribution of a point given the current front (2D)."""
            if len(front) == 0:
                return max(0, ref_point[0] - point[0]) * max(0, ref_point[1] - point[1])
            
            # Sort front by first objective
            sorted_front = sorted(front, key=lambda p: p[0])
            
            # Check if point is dominated
            for fp in sorted_front:
                if fp[0] <= point[0] and fp[1] <= point[1] and (fp[0] < point[0] or fp[1] < point[1]):
                    return 0.0
            
            # Compute contribution by finding the region not covered by front
            # In 2D, the contribution is the area of the rectangle from point to ref
            # minus the parts already covered by front points that are "better" in one obj
            # but worse in the other.
            
            # Build sorted list including the point
            combined = sorted_front + [point]
            combined.sort(key=lambda p: p[0])
            
            # Find position of point
            # The contribution is the area of the "skyline" region
            # For 2D: contribution = (ref[0] - point[0]) * (ref[1] - point[1]) 
            #   - sum of overlaps with front points that are non-dominated
            # Simpler: compute HV of front, then HV of front+point, difference = contribution
            
            def hv_2d(points, ref):
                if not points:
                    return 0.0
                # Filter points that are within ref
                pts = [(p[0], p[1]) for p in points if p[0] < ref[0] and p[1] < ref[1]]
                if not pts:
                    return 0.0
                pts.sort(key=lambda p: p[0])
                hv = 0.0
                y_max = ref[1]
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2 = pts[i+1][0] if i + 1 < len(pts) else ref[0]
                    if y1 < y_max:
                        hv += (x2 - x1) * (y_max - y1)
                        y_max = min(y_max, y1)
                    else:
                        hv += (x2 - x1) * (ref[1] - y1)
                return hv

            hv_before = hv_2d(sorted_front, ref_point)
            hv_after = hv_2d(combined, ref_point)
            return max(0.0, hv_after - hv_before)

        def hypervolume_2d(points, ref):
            if not points:
                return 0.0
            pts = [(p[0], p[1]) for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if not pts:
                return 0.0
            pts.sort(key=lambda p: p[0])
            hv = 0.0
            y_max = ref[1]
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2 = pts[i+1][0] if i + 1 < len(pts) else ref[0]
                if y1 < y_max:
                    hv += (x2 - x1) * (y_max - y1)
                    y_max = min(y_max, y1)
                else:
                    hv += (x2 - x1) * (ref[1] - y1)
            return hv

        # Phase 1: Stratified initialization
        # Use Latin Hypercube Sampling for better coverage
        n_init = min(80, self.budget)
        
        # Create LHS points
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            for i in range(n_init):
                lhs_points[i, d] = lo[d] + bounds_range[d] * (perm[i] + np.random.random()) / n_init
        
        # Add corners
        corners = []
        for mask in range(2 ** self.dim):
            c = np.zeros(self.dim)
            for d in range(self.dim):
                c[d] = hi[d] if (mask >> d) & 1 else lo[d]
            corners.append(c)
        
        # Add center
        center = (lo + hi) / 2.0
        
        all_initial = np.vstack([lhs_points, np.array(corners), center.reshape(1, -1)])
        
        for x in all_initial:
            if eval_count >= self.budget or time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # Phase 2: Hypervolume-guided Adaptive DE with local search
        pop_size = 20
        F_mut = 0.7  # Mutation factor
        CR = 0.4     # Crossover rate
        n_generations = 0
        max_generations = 500
        no_improve_count = 0
        best_hv = 0.0

        # Get initial population
        nd_indices = get_non_dominated_indices()
        
        # Build population from non-dominated points, padded with random if needed
        if len(nd_indices) >= pop_size:
            # Select diverse subset: sort by f1, take evenly spaced
            nd_F = np.array([all_F[i] for i in nd_indices])
            sort_idx = np.argsort(nd_F[:, 0])
            sorted_nd = nd_indices[sort_idx]
            step = max(1, len(sorted_nd) // pop_size)
            selected = sorted_nd[::step]
            if len(selected) < pop_size:
                extra = np.random.choice(sorted_nd, size=pop_size - len(selected), replace=True)
                selected = np.concatenate([selected, extra])
            pop_indices = selected[:pop_size]
        else:
            pop_indices = list(nd_indices)
            extra_count = pop_size - len(pop_indices)
            extra_idx = np.random.choice(len(all_X), size=extra_count, replace=True)
            pop_indices = np.concatenate([pop_indices, extra_idx])
        
        population = [all_X[i].copy() for i in pop_indices]
        pop_F = [all_F[i].copy() for i in pop_indices]

        while (eval_count < self.budget and time.time() < deadline and 
               n_generations < max_generations and no_improve_count < 30):
            
            # Update non-dominated front
            nd_indices = get_non_dominated_indices()
            if len(nd_indices) == 0:
                break
            
            nd_F_list = [all_F[i] for i in nd_indices]
            current_hv = hypervolume_2d(nd_F_list, ref_point)
            
            if current_hv > best_hv + 1e-10:
                best_hv = current_hv
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Rebuild population from current non-dominated front
            if len(nd_indices) >= pop_size:
                nd_F_arr = np.array(nd_F_list)
                sort_idx = np.argsort(nd_F_arr[:, 0])
                sorted_nd = nd_indices[sort_idx]
                step = max(1, len(sorted_nd) // pop_size)
                selected = sorted_nd[::step]
                if len(selected) < pop_size:
                    extra = np.random.choice(sorted_nd, size=pop_size - len(selected), replace=True)
                    selected = np.concatenate([selected, extra])
                pop_indices = selected[:pop_size]
            else:
                pop_indices = list(nd_indices)
                extra_count = pop_size - len(pop_indices)
                extra_idx = np.random.choice(len(all_X), size=extra_count, replace=True)
                pop_indices = np.concatenate([pop_indices, extra_idx])
            
            population = [all_X[i].copy() for i in pop_indices]
            pop_F = [all_F[i].copy() for i in pop_indices]
            
            # Adaptive mutation factor: shrink as we converge
            progress = min(1.0, n_generations / max_generations)
            F_current = 0.8 * (1.0 - 0.5 * progress)  # Shrinks from 0.8 to 0.4
            CR_current = 0.3 + 0.4 * (1.0 - progress)  # Shrinks from 0.7 to 0.3
            
            # DE generation
            for i in range(pop_size):
                if eval_count >= self.budget or time.time() > deadline:
                    break
                
                # Select 3 distinct parents
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break
                p1, p2, p3 = np.random.choice(candidates, size=3, replace=False)
                
                # Mutation
                v = population[p1] + F_current * (population[p2] - population[p3])
                v = np.clip(v, lo, hi)
                
                # Crossover
                u = population[i].copy()
                for j in range(self.dim):
                    if np.random.random() < CR_current or j == np.random.randint(self.dim):
                        u[j] = v[j]
                
                # Evaluate
                f_u = evaluate(u)
                if f_u is None:
                    continue
                
                # Selection: keep if it improves or is non-dominated
                # Check if u dominates current i
                if dominates(f_u, pop_F[i]):
                    pop_F[i] = f_u
                    population[i] = u.copy()
                elif not dominates(pop_F[i], f_u):
                    # Both non-dominated: use HV contribution to decide
                    # Compute HV of current front without i, then with i, then with u
                    # For efficiency, just keep both if budget allows
                    # Simple: replace if u has better HV contribution
                    front_without_i = [pop_F[k] for k in range(pop_size) if k != i]
                    hv_with_i = hypervolume_2d(front_without_i + [pop_F[i]], ref_point)
                    hv_with_u = hypervolume_2d(front_without_i + [f_u], ref_point)
                    if hv_with_u > hv_with_i:
                        pop_F[i] = f_u
                        population[i] = u.copy()
                
                # Local search on the new point
                if f_u is not None and time.time() < deadline and eval_count < self.budget:
                    # Small local perturbation
                    for ls_iter in range(3):
                        if eval_count >= self.budget or time.time() > deadline:
                            break
                        step_size = 0.05 * bounds_range
                        x_ls = u + np.random.randn(self.dim) * step_size
                        x_ls = np.clip(x_ls, lo, hi)
                        f_ls = evaluate(x_ls)
                        if f_ls is None:
                            break
                        if dominates(f_ls, f_u):
                            u = x_ls.copy()
                            f_u = f_ls
                            population[i] = u.copy()
                            pop_F[i] = f_u
                        elif not dominates(f_u, f_ls):
                            # Keep the better one
                            front_without_u = [pop_F[k] for k in range(pop_size) if k != i]
                            hv_u = hypervolume_2d(front_without_u + [f_u], ref_point)
                            hv_ls = hypervolume_2d(front_without_u + [f_ls], ref_point)
                            if hv_ls > hv_u:
                                u = x_ls.copy()
                                f_u = f_ls
                                population[i] = u.copy()
                                pop_F[i] = f_u
            
            n_generations += 1

        return all_X
