import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        # Center of the search space
        center = 0.5 * (lo + hi)
        # Radius of the initial hyper-sphere (half the diagonal)
        radius = 0.5 * np.linalg.norm(hi - lo)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_mat, n_total):
            ranks = np.zeros(n_total, dtype=int)
            S = [[] for _ in range(n_total)]
            n_dom = np.zeros(n_total, dtype=int)
            fronts = []

            for i in range(n_total):
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[i], F_mat[j]):
                        S[i].append(j)
                    elif dominates(F_mat[j], F_mat[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r

            return ranks, fronts

        def crowding_distance(F_mat, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k]
                if f_range > 1e-10:
                    for i in range(1, m - 1):
                        dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / f_range
            return dist

        # Store all evaluated points and their objective values
        all_X = []
        all_F = []
        evals_used = 0
        
        # Track best regions for exploitation
        best_front = []
        last_prune_eval = 0
        
        def fast_non_dominated_front(F_mat, n_total):
            """Return indices of the non-dominated front (rank 0)."""
            front = []
            for i in range(n_total):
                dominated = False
                for j in range(n_total):
                    if i == j:
                        continue
                    if dominates(F_mat[j], F_mat[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            return front
        
        # Main search loop
        n_batch = 20
        while evals_used < max_evals:
            if time.time() > deadline:
                break
            
            progress = evals_used / max_evals
            
            # Mix exploration and exploitation
            # Early: more exploration, later: more exploitation around good points
            exploit_ratio = min(0.7, progress * 0.8)
            
            for i in range(n_batch):
                if evals_used >= max_evals:
                    break
                if time.time() > deadline:
                    break
                
                if best_front and np.random.rand() < exploit_ratio:
                    # Exploit: perturb a point from the current front
                    idx = np.random.randint(len(best_front))
                    x_parent = best_front[idx]
                    # Gaussian perturbation with shrinking step size
                    step = radius * 0.1 * (1.0 - progress * 0.5)
                    x = x_parent + np.random.randn(self.dim) * step
                else:
                    # Explore: sample from shrinking sphere
                    current_radius = radius * (1.0 - progress * 0.8)
                    u = np.random.randn(self.dim)
                    u /= np.linalg.norm(u)
                    r = current_radius * np.random.rand() ** (1.0 / self.dim)
                    x = center + r * u
                
                x = np.clip(x, lo, hi)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                evals_used += 1
            
            # Prune archive periodically
            if len(all_F) > 80:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                
                # Find non-dominated front
                front_indices = fast_non_dominated_front(F_arr, len(all_F))
                
                # If front is too large, use crowding distance to select
                max_archive_size = 40
                if len(front_indices) > max_archive_size:
                    dists = crowding_distance(F_arr, front_indices)
                    keep_idx = np.argsort(dists)[::-1][:max_archive_size]
                    keep_set = set(front_indices[i] for i in keep_idx)
                else:
                    keep_set = set(front_indices)
                
                # Also keep some diverse points from other fronts for exploration
                other_indices = [i for i in range(len(all_F)) if i not in keep_set]
                if other_indices and len(keep_set) < max_archive_size:
                    # Add a few random points from other fronts
                    n_add = min(5, len(other_indices), max_archive_size - len(keep_set))
                    add_indices = np.random.choice(other_indices, n_add, replace=False)
                    keep_set.update(add_indices)
                
                all_X = [X_arr[i] for i in sorted(keep_set)]
                all_F = [F_arr[i] for i in sorted(keep_set)]
                
                # Update best front for exploitation
                if all_F:
                    F_new = np.array(all_F)
                    front_new = fast_non_dominated_front(F_new, len(all_F))
                    best_front = [all_X[i] for i in front_new]
                last_prune_eval = evals_used

        return np.array(all_X) if all_X else np.zeros((0, self.dim))
