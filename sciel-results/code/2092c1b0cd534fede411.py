"""Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 330.0 - 5.0  # 5s safety margin
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def within_deadline():
            return time.time() < deadline

        # --- Helpers ---
        def dominates(a, b):
            # a dominates b if a <= b in all and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            nd_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx

        def crowding_distance(F_nd, indices):
            """Compute crowding distances for non-dominated points."""
            m = len(indices)
            if m <= 2:
                return np.ones(m)
            cd = np.zeros(m)
            F_sub = F_nd[indices]
            for d in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, d])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F_sub[sorted_idx[0], d]
                f_max = F_sub[sorted_idx[-1], d]
                if f_max - f_min < 1e-12:
                    continue
                for k in range(1, m - 1):
                    cd[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], d] - F_sub[sorted_idx[k - 1], d]) / (f_max - f_min)
            return cd

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_list = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        dominated_list[i].append(j)
                    elif dominates(F[j], F[i]):
                        domination_count[i] += 1
            front = []
            for i in range(n):
                if domination_count[i] == 0:
                    front.append(i)
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in dominated_list[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        # --- Initial random sampling ---
        pop_size = min(40, budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points and their objectives
        all_X = list(X)
        all_F = list(F)

        def evaluate(x):
            nonlocal used
            fx = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(fx)
            return fx

        # --- NSGA-II style evolution with adaptive mutation ---
        # Population size for the evolutionary loop
        pop_size_evo = 30
        
        # Adaptive mutation: start with larger scale, reduce over time
        initial_mut_scale = 0.3
        final_mut_scale = 0.01
        total_iterations = max(1, (budget - used) // 2)
        iteration = 0
        
        while used < budget and within_deadline():
            iteration += 1
            # Get non-dominated fronts
            F_arr = np.array(all_F)
            fronts = fast_non_dominated_sort(F_arr)
            
            # Select parents from the best fronts using tournament selection
            # Prefer front 0, then front 1 if needed
            parent_pool = list(fronts[0])
            if len(parent_pool) < 10 and len(fronts) > 1:
                parent_pool += fronts[1][:10]
            
            if len(parent_pool) == 0:
                parent_pool = list(range(len(all_F)))
            
            # Binary tournament selection for two parents
            def tournament_select(pool):
                a = pool[rng.integers(len(pool))]
                b = pool[rng.integers(len(pool))]
                # Compare by front rank, then crowding distance
                F_a = all_F[a]
                F_b = all_F[b]
                # Simple: pick the one with better sum of objectives
                if np.sum(F_a) < np.sum(F_b):
                    return a
                else:
                    return b
            
            p1 = tournament_select(parent_pool)
            p2 = tournament_select(parent_pool)
            
            # Simple crossover (SBX-like)
            alpha = rng.random()
            c1 = alpha * all_X[p1] + (1 - alpha) * all_X[p2]
            c2 = (1 - alpha) * all_X[p1] + alpha * all_X[p2]
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            # Adaptive mutation: scale decreases over iterations
            progress = iteration / total_iterations
            mut_scale = initial_mut_scale - (initial_mut_scale - final_mut_scale) * progress
            c1 = np.clip(c1 + mut_scale * (hi - lo) * rng.standard_normal(dim), lo, hi)
            c2 = np.clip(c2 + mut_scale * (hi - lo) * rng.standard_normal(dim), lo, hi)
            
            f1 = evaluate(c1)
            if not within_deadline() or used >= budget:
                break
            f2 = evaluate(c2)
            
            # Elitism: maintain archive of non-dominated points
            # Remove dominated points from archive to keep memory manageable
            F_arr = np.array(all_F)
            nd_idx = non_dominated_indices(F_arr)
            if len(nd_idx) < len(all_F):
                # Remove some dominated points to keep memory manageable
                dominated_idx = [i for i in range(len(all_F)) if i not in set(nd_idx)]
                if dominated_idx:
                    # Keep a fraction of dominated points for diversity
                    remove_frac = min(0.15, len(dominated_idx) / max(1, len(all_F)))
                    n_remove = int(len(dominated_idx) * remove_frac)
                    to_remove = rng.choice(dominated_idx, size=min(n_remove, len(dominated_idx)), replace=False)
                    to_remove_set = set(to_remove)
                    all_X = [x for i, x in enumerate(all_X) if i not in to_remove_set]
                    all_F = [f for i, f in enumerate(all_F) if i not in to_remove_set]

        # --- Local refinement: grid-based scan in promising regions ---
        if within_deadline() and used < budget:
            # Find the Pareto front points and refine around them
            F_arr = np.array(all_F)
            nd_idx = non_dominated_indices(F_arr)
            nd_points_X = np.array([all_X[i] for i in nd_idx])
            nd_points_F = np.array([all_F[i] for i in nd_idx])
            
            # For each non-dominated point, do a local refinement
            for idx in range(len(nd_points_X)):
                if not within_deadline() or used >= budget:
                    break
                base = nd_points_X[idx]
                # Radial search with multiple scales
                for scale in [0.05, 0.02, 0.01]:
                    if not within_deadline() or used >= budget:
                        break
                    for _ in range(4):
                        if used >= budget or not within_deadline():
                            break
                        direction = rng.standard_normal(dim)
                        direction /= np.linalg.norm(direction) + 1e-12
                        for sign in [1, -1]:
                            if used >= budget or not within_deadline():
                                break
                            x_new = base + sign * scale * (hi - lo) * direction
                            x_new = np.clip(x_new, lo, hi)
                            evaluate(x_new)

        # --- Final: return all evaluated points ---
        return np.array(all_X)
# EVOLVE-BLOCK-END
