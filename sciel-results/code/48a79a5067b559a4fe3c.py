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

    def _non_dominated_front(self, F):
        """Return indices of non-dominated points in F (2D minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        F = np.array(F, dtype=float)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _hv_contribution_2d(self, point, front, ref):
        """Compute HV contribution of a single point in a 2D front."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        p = np.array(point, dtype=float)
        if not np.all(p < ref):
            return 0.0
        
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        nd_front = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < nd_front[-1][1]:
                nd_front.append(pt)
        
        if len(nd_front) == 0:
            return 0.0
        
        all_pts = np.vstack([np.array(nd_front), p.reshape(1, -1)])
        order = np.argsort(all_pts[:, 0])
        all_pts = all_pts[order]
        
        nd_all = [all_pts[0]]
        for pt in all_pts[1:]:
            if pt[1] < nd_all[-1][1]:
                nd_all.append(pt)
        
        hv_with = 0.0
        for i in range(len(nd_all)):
            x_next = nd_all[i + 1][0] if i + 1 < len(nd_all) else ref[0]
            y_curr = nd_all[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - nd_all[i][0]
            height = ref[1] - y_curr
            hv_with += width * height
        
        hv_without = 0.0
        for i in range(len(nd_front)):
            x_next = nd_front[i + 1][0] if i + 1 < len(nd_front) else ref[0]
            y_curr = nd_front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - nd_front[i][0]
            height = ref[1] - y_curr
            hv_without += width * height
        
        return hv_with - hv_without

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        # Reference-point-aligned deterministic grid initialization
        # Since ref = [3000, 0.06], obj0 has huge range and obj1 tiny range.
        # Use stratified sampling in both objectives' implied ranges.
        n_grid = 30
        grid_X = []
        # Stratified grid in decision space
        for i in range(n_grid):
            for j in range(n_grid):
                x = np.zeros(self.dim)
                # Use Sobol-like deterministic points
                # Mix of uniform stratification
                x[0] = lo[0] + (hi[0] - lo[0]) * ((i + 0.5) / n_grid)
                x[1] = lo[1] + (hi[1] - lo[1]) * ((j + 0.5) / n_grid)
                x[2] = lo[2] + (hi[2] - lo[2]) * ((i * 7 + j * 3 + 1) % n_grid) / n_grid
                x[3] = lo[3] + (hi[3] - lo[3]) * ((i * 3 + j * 11 + 5) % n_grid) / n_grid
                grid_X.append(x)
        
        # Add boundary and center points
        grid_X.append(lo.copy())
        grid_X.append(hi.copy())
        grid_X.append((lo + hi) / 2)
        
        # Add random points to fill remaining budget
        n_random = min(self.budget - len(grid_X), 40)
        if n_random > 0:
            rand_X = lo + (hi - lo) * rng.random((n_random, self.dim))
            grid_X.extend(rand_X.tolist())
        
        # Cap at budget
        grid_X = grid_X[:self.budget]
        X_init = np.array(grid_X)
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        all_X = list(X_init)
        all_F = list(F_init)

        # Differential evolution parameters
        F_de = 0.8  # Mutation factor
        CR = 0.5    # Crossover rate
        pop_size = min(30, self.budget // 4)
        
        # Select initial population from all evaluated points
        # Prioritize non-dominated points
        front_idx = self._non_dominated_front(all_F)
        if len(front_idx) >= pop_size:
            pop_idx = front_idx[:pop_size]
        else:
            pop_idx = list(front_idx)
            remaining = pop_size - len(pop_idx)
            others = [i for i in range(len(all_F)) if i not in pop_idx]
            if others:
                pop_idx.extend(rng.choice(others, size=min(remaining, len(others)), replace=False).tolist())
        
        while len(pop_idx) < pop_size:
            pop_idx.append(rng.randint(len(all_F)))
        
        pop_X = [all_X[i].copy() for i in pop_idx]
        pop_F = [all_F[i].copy() for i in pop_idx]

        # Main DE loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find current non-dominated front for HV-guided selection
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) == 0:
                front_idx = np.array([0])
            front_points = [all_F[i] for i in front_idx]
            
            # Compute HV contributions for front points
            contributions = np.zeros(len(front_idx))
            for k, idx in enumerate(front_idx):
                contributions[k] = self._hv_contribution_2d(all_F[idx], front_points, ref)
            
            # DE generation
            for i in range(len(pop_X)):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct parents different from i
                candidates = [j for j in range(len(pop_X)) if j != i]
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Adaptive mutation factor: shrink over time
                progress = used / self.budget
                F_adapt = F_de * (1.0 - 0.5 * progress) + 0.3
                F_adapt = max(0.3, min(1.0, F_adapt))
                
                # Differential evolution mutation
                v = pop_X[r1] + F_adapt * (pop_X[r2] - pop_X[r3])
                
                # Crossover
                child = pop_X[i].copy()
                for d in range(self.dim):
                    if rng.random() < CR:
                        child[d] = v[d]
                
                # Small perturbation for exploration
                perturbation_scale = 0.05 * (1.0 - 0.8 * progress) + 0.01
                child += rng.standard_normal(self.dim) * perturbation_scale * (hi - lo)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                all_X.append(child)
                all_F.append(fc)
                
                # Update population: keep if non-dominated or improves HV
                new_front = self._non_dominated_front(all_F[-100:])  # Check recent
                if len(new_front) > 0:
                    # Replace worst individual in population
                    pop_F_arr = np.array(pop_F)
                    # Simple replacement: if child dominates any pop member
                    for j in range(len(pop_F)):
                        if np.all(fc <= pop_F[j]) and np.any(fc < pop_F[j]):
                            pop_X[j] = child.copy()
                            pop_F[j] = fc.copy()
                            break
                    else:
                        # Replace random individual
                        j = rng.randint(len(pop_X))
                        pop_X[j] = child.copy()
                        pop_F[j] = fc.copy()
        
        # Final local refinement around best HV contributors
        remaining = self.budget - used
        if remaining > 0:
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) > 0:
                front_points = [all_F[i] for i in front_idx]
                contributions = np.zeros(len(front_idx))
                for k, idx in enumerate(front_idx):
                    contributions[k] = self._hv_contribution_2d(all_F[idx], front_points, ref)
                
                top_k = min(8, len(front_idx))
                top_indices = front_idx[np.argsort(contributions)[-top_k:]]
                
                local_steps = min(remaining, 40)
                for step in range(local_steps):
                    if time.time() > deadline:
                        break
                    parent_idx = top_indices[step % len(top_indices)]
                    parent = all_X[parent_idx].copy()
                    
                    small_mut = 0.03 * (hi - lo)
                    child = parent + rng.standard_normal(self.dim) * small_mut
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    all_X.append(child)
                    all_F.append(fc)
                    
                    # Update top indices periodically
                    if step % 10 == 9:
                        new_front = self._non_dominated_front(all_F)
                        if len(new_front) > 0:
                            new_front_points = [all_F[i] for i in new_front]
                            new_contribs = np.zeros(len(new_front))
                            for k, idx in enumerate(new_front):
                                new_contribs[k] = self._hv_contribution_2d(all_F[idx], new_front_points, ref)
                            top_indices = new_front[np.argsort(new_contribs)[-top_k:]]

        return all_X
# EVOLVE-BLOCK-END
