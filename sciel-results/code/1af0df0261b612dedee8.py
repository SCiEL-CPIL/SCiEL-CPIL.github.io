"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        used = 0
        all_X = []
        all_F = []
        front_indices = []  # maintain running front

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fc)
            return fc

        def update_front():
            """Incrementally update the non-dominated front indices."""
            nonlocal front_indices
            n = len(all_F)
            if n == 0:
                front_indices = []
                return
            
            F = np.array(all_F, dtype=float)
            mask = np.all(F < ref, axis=1)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                front_indices = []
                return
            
            F_valid = F[valid_indices]
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            front_idx_sorted = [0]
            min_obj1 = F_sorted[0][1]
            for i in range(1, len(F_sorted)):
                if F_sorted[i][1] < min_obj1:
                    front_idx_sorted.append(i)
                    min_obj1 = F_sorted[i][1]
            
            front_indices = list(valid_indices[order[front_idx_sorted]])

        def compute_hv_contributions(front_idx):
            """Compute HV contribution for each point in front_idx."""
            if len(front_idx) == 0:
                return np.array([])
            F_front = np.array([all_F[i] for i in front_idx], dtype=float)
            order_f = np.argsort(F_front[:, 0])
            F_sorted = F_front[order_f]
            
            contributions = np.zeros(len(F_front))
            for i in range(len(F_sorted)):
                x_left = F_sorted[i][0]
                x_right = F_sorted[i + 1][0] if i + 1 < len(F_sorted) else ref[0]
                y_val = F_sorted[i][1]
                if y_val < ref[1] and x_right > x_left:
                    width = x_right - x_left
                    height = ref[1] - y_val
                    contributions[order_f[i]] = max(0.0, width * height)
            return contributions

        # Phase 1: Initial exploration with structured sampling
        init_budget = min(20, self.budget // 10)
        
        # Structured initialization: center, corners, and mid-edges
        init_fracs = []
        init_fracs.append(np.full(self.dim, 0.5))
        corners = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ]
        for c in corners:
            init_fracs.append(np.array(c, dtype=float))
        mid_edges = [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 1.0, 1.0],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 1.0, 0.5, 0.5],
        ]
        for m in mid_edges:
            init_fracs.append(np.array(m, dtype=float))
        extra_structured = [
            [0.25, 0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75, 0.75],
            [0.25, 0.75, 0.25, 0.75],
            [0.75, 0.25, 0.75, 0.25],
            [0.5, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0, 0.5],
            [1.0, 0.5, 0.0, 0.5],
        ]
        for e in extra_structured:
            init_fracs.append(np.array(e, dtype=float))
        
        rng.shuffle(init_fracs)
        init_fracs = init_fracs[:init_budget]
        
        for frac in init_fracs:
            if time.time() > deadline:
                break
            x = lo + frac * (hi - lo)
            evaluate(x)
        
        update_front()

        # Phase 2: Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            n_total = len(all_F)
            if n_total < 3:
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
                continue
            
            # Adaptive parameters based on progress
            progress = used / self.budget
            F_mut = 0.8 * (1.0 - progress) + 0.2  # 0.8 -> 0.2
            CR = 0.3 * (1.0 - progress) + 0.7    # 0.3 -> 0.7
            
            # Build selection pool: front points + near-front points
            front_set = set(front_indices)
            selection_pool = []
            
            # Add all front points
            for idx in front_indices:
                selection_pool.append(idx)
            
            # Add near-front points (dominated but close to front)
            if len(front_indices) > 0:
                F_front = np.array([all_F[i] for i in front_indices], dtype=float)
                for idx in range(n_total):
                    if idx in front_set:
                        continue
                    f = all_F[idx]
                    if np.all(f < ref):
                        # Check if this point is close to the front
                        # Find the front point that dominates it most
                        dominated_by = None
                        for fi in front_indices:
                            ff = all_F[fi]
                            if ff[0] <= f[0] and ff[1] <= f[1] and (ff[0] < f[0] or ff[1] < f[1]):
                                dominated_by = fi
                                break
                        if dominated_by is not None:
                            ff = all_F[dominated_by]
                            dist = np.sqrt((f[0] - ff[0])**2 + (f[1] - ff[1])**2)
                            # Include if within reasonable distance
                            scale = max(ref[0] - F_front[:, 0].min(), 1.0)
                            if dist < 0.1 * scale:
                                selection_pool.append(idx)
            
            # If selection pool is too small, add random points
            if len(selection_pool) < 3:
                remaining = [i for i in range(n_total) if i not in set(selection_pool)]
                if remaining:
                    selection_pool.extend(rng.choice(remaining, size=min(3, len(remaining)), replace=False).tolist())
                else:
                    selection_pool = list(range(n_total))
            
            # Tournament selection for 3 parents
            parents = []
            for _ in range(3):
                # Tournament of 3
                candidates = rng.choice(len(selection_pool), size=min(3, len(selection_pool)), replace=False)
                best_cand = candidates[0]
                best_hv = -1
                for c in candidates:
                    idx = selection_pool[c]
                    # Simple heuristic: prefer front points
                    if idx in front_set:
                        best_hv = 1.0
                        best_cand = c
                        break
                    else:
                        best_cand = c
                parents.append(selection_pool[best_cand])
            
            p1 = all_X[parents[0]]
            p2 = all_X[parents[1]]
            p3 = all_X[parents[2]]
            
            v = p1 + F_mut * (p2 - p3)
            mask = rng.random(self.dim) < CR
            child = np.where(mask, v, p1)
            
            # Adaptive Gaussian perturbation - reduce as we converge
            mut_scale = 0.05 * (1.0 - progress) + 0.005
            child = child + rng.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = np.clip(child, lo, hi)
            
            evaluate(child)
            
            # Update front periodically (every 5 evaluations)
            if used % 5 == 0:
                update_front()
            
            # Exploration: every 15 evals, try a random point
            if used % 15 == 0 and used < self.budget - 1 and time.time() < deadline:
                x_rand = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x_rand)

        update_front()
        return all_X
# EVOLVE-BLOCK-END
