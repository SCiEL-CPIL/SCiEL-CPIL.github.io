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

    def _hv_2d(self, front, ref):
        """Compute exact 2D hypervolume for a list of points (Pareto front)."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        # Filter strictly better than ref
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by obj0 ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # Keep only non-dominated points (obj1 must be strictly decreasing)
        front = [pts[0]]
        for p in pts[1:]:
            if p[1] < front[-1][1]:
                front.append(p)
        if len(front) == 0:
            return 0.0
        # Compute HV: sum of rectangles
        hv = 0.0
        for i in range(len(front)):
            x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
            y_curr = front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - front[i][0]
            height = ref[1] - y_curr
            hv += width * height
        return hv

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

        # Sort by obj0
        order = np.argsort(pts[:, 0])
        pts = pts[order]

        # Build non-dominated front
        nd_front = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < nd_front[-1][1]:
                nd_front.append(pt)

        if len(nd_front) == 0:
            return 0.0

        # Compute HV with and without the point
        front_with = [p] + list(front)
        hv_with = self._hv_2d(front_with, ref)
        hv_without = self._hv_2d(front, ref)

        return hv_with - hv_without

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

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        # Reference-point-aligned deterministic grid initialization
        # Since ref = [3000, 0.06], the first objective dominates HV contribution
        # We want to spread points to cover the objective space well

        # Initial population: diverse random points + structured grid
        pop_size = min(40, self.budget // 5)
        
        # Create a stratified grid in objective space mapped to decision space
        # Since we don't know the mapping, use stratified sampling in decision space
        X_list = []
        
        # Corner points
        X_list.append(lo.copy())
        X_list.append(hi.copy())
        
        # Center
        X_list.append((lo + hi) / 2.0)
        
        # Mid-edges
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                pt = (lo + hi) / 2.0
                pt[i] = lo[i]
                pt[j] = hi[j]
                X_list.append(pt.copy())
                pt2 = (lo + hi) / 2.0
                pt2[i] = hi[i]
                pt2[j] = lo[j]
                X_list.append(pt2.copy())
        
        # Random points to fill up to pop_size
        while len(X_list) < pop_size:
            X_list.append(lo + (hi - lo) * rng.random(self.dim))
        
        X = np.array(X_list[:pop_size])
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Track all evaluated points
        all_X = list(X)
        all_F = list(F)

        # Adaptive mutation scale
        mut_scale = 0.12

        # Differential evolution parameters
        F_mut = 0.8  # mutation factor
        CR = 0.5     # crossover rate
        pop_de = min(20, self.budget // 10)  # DE population size

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) == 0:
                front_idx = np.array([0])

            # Select parents using hypervolume contribution
            front_points = [all_F[i] for i in front_idx]

            # Compute contribution of each front point
            contributions = np.zeros(len(front_idx))
            for k, idx in enumerate(front_idx):
                contributions[k] = self._hv_contribution_2d(all_F[idx], front_points, ref)

            # Normalize contributions
            total_contrib = np.sum(contributions)
            if total_contrib > 0:
                probs = contributions / total_contrib
            else:
                probs = np.ones(len(front_idx)) / len(front_idx)

            # Select parents via weighted sampling
            if len(front_idx) >= 3:
                # Differential evolution: select 3 distinct parents
                parent_indices = rng.choice(len(front_idx), size=3, replace=False, p=probs)
                p1 = all_X[front_idx[parent_indices[0]]].copy()
                p2 = all_X[front_idx[parent_indices[1]]].copy()
                p3 = all_X[front_idx[parent_indices[2]]].copy()
                
                # DE mutation
                if rng.random() < 0.6:
                    v = p1 + F_mut * (p2 - p3)
                else:
                    # Blend crossover as alternative
                    alpha = rng.random()
                    v = alpha * p1 + (1 - alpha) * p2
                
                # Binomial crossover
                mask = rng.random(self.dim) < CR
                child = np.where(mask, v, p1)
            else:
                # Fallback for small fronts
                parent_idx = rng.choice(len(front_idx), p=probs)
                p1 = all_X[front_idx[parent_idx]].copy()
                child = p1.copy()

            # Mutate with adaptive scale
            progress = used / self.budget
            # Decrease scale as we progress
            current_mut = mut_scale * (1.0 - 0.5 * progress) + 0.02
            mutation = rng.standard_normal(self.dim) * current_mut * (hi - lo)
            child = child + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X.append(child)
            all_F.append(fc)

            # Periodic random exploration
            if used % 30 == 0 and used < self.budget - 5:
                for _ in range(2):
                    if used >= self.budget:
                        break
                    random_child = lo + (hi - lo) * rng.random(self.dim)
                    rc = problem(random_child)
                    used += 1
                    all_X.append(random_child)
                    all_F.append(rc)

        return all_X
# EVOLVE-BLOCK-END
