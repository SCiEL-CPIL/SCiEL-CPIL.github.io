import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        # Center of the search space
        center = 0.5 * (lo + hi)
        # Radius of the initial hyper-sphere (half the diagonal)
        radius = 0.5 * np.linalg.norm(hi - lo)

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_non_dominated_front(F_mat, n_total):
            """Return indices of the non-dominated front (rank 0) only."""
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

        # Phase 1: Exploration with shrinking sphere (80% of budget)
        exploration_budget = int(0.8 * max_evals)
        n_batch = 10
        while evals_used < exploration_budget:
            if time.time() > deadline:
                break
            
            progress = evals_used / max_evals
            current_radius = radius * (1.0 - progress)
            
            for i in range(n_batch):
                if evals_used >= exploration_budget:
                    break
                if time.time() > deadline:
                    break
                    
                u = np.random.randn(self.dim)
                u /= np.linalg.norm(u)
                r = current_radius * np.random.rand() ** (1.0 / self.dim)
                x = center + r * u
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                evals_used += 1
                
            if len(all_F) > 150:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                front = get_non_dominated_front(F_arr, len(all_F))
                
                if len(front) > 80:
                    dists = crowding_distance(F_arr, front)
                    front_sorted = [front[i] for i in np.argsort(dists)[::-1]]
                    selected = front_sorted[:80]
                else:
                    selected = front
                
                all_X = [X_arr[i] for i in selected]
                all_F = [F_arr[i] for i in selected]

        # Phase 2: Exploitation around best points (20% of budget)
        if all_F and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Get non-dominated front
            front = get_non_dominated_front(F_arr, len(F_arr))
            
            # Sort front by crowding distance and pick diverse points
            if len(front) > 10:
                dists = crowding_distance(F_arr, front)
                front_sorted = [front[i] for i in np.argsort(dists)[::-1]]
                exploit_points = front_sorted[:10]
            else:
                exploit_points = front
            
            remaining_budget = max_evals - evals_used
            n_exploit_per_point = max(1, remaining_budget // max(1, len(exploit_points)))
            
            for idx in exploit_points:
                if evals_used >= max_evals or time.time() > deadline:
                    break
                
                base_x = X_arr[idx]
                for _ in range(n_exploit_per_point):
                    if evals_used >= max_evals or time.time() > deadline:
                        break
                    
                    # Small perturbation around the base point
                    step_size = radius * 0.05
                    x = base_x + np.random.randn(self.dim) * step_size
                    x = np.clip(x, lo, hi)
                    
                    f = problem(x)
                    all_X.append(x)
                    all_F.append(f)
                    evals_used += 1

        return np.array(all_X) if all_X else np.zeros((0, self.dim))
