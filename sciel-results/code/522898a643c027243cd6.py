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

        def evaluate(x):
            nonlocal used
            fc = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fc)
            return fc

        def compute_front_and_contributions():
            """Compute non-dominated front and per-point HV contributions in one pass."""
            n = len(all_F)
            if n == 0:
                return np.array([], dtype=int), np.array([])
            F = np.array(all_F, dtype=float)
            mask = np.all(F < ref, axis=1)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return np.array([], dtype=int), np.array([])
            F_valid = F[valid_indices]
            
            # Sort by obj0 ascending
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            # Keep only non-dominated (obj1 must be strictly decreasing)
            front_mask = np.zeros(len(F_sorted), dtype=bool)
            min_obj1 = np.inf
            for i in range(len(F_sorted)):
                if F_sorted[i][1] < min_obj1:
                    front_mask[i] = True
                    min_obj1 = F_sorted[i][1]
            
            front_local_idx = np.where(front_mask)[0]
            if len(front_local_idx) == 0:
                return np.array([], dtype=int), np.array([])
            
            front_indices = valid_indices[order[front_local_idx]]
            
            # Compute HV contribution for each front point
            F_front = F_sorted[front_local_idx]
            contributions = np.zeros(len(F_front))
            for i in range(len(F_front)):
                x_left = F_front[i][0]
                x_right = F_front[i + 1][0] if i + 1 < len(F_front) else ref[0]
                y_val = F_front[i][1]
                if y_val < ref[1] and x_right > x_left:
                    width = x_right - x_left
                    height = ref[1] - y_val
                    contributions[i] = max(0.0, width * height)
            
            return front_indices, contributions

        # Phase 1: Initial exploration with structured sampling
        init_budget = min(20, self.budget // 10)
        
        # Structured initialization: center, corners, and mid-edges
        init_fracs = []
        # Center
        init_fracs.append(np.full(self.dim, 0.5))
        # 4 corners
        corners = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ]
        for c in corners:
            init_fracs.append(np.array(c, dtype=float))
        # 4 mid-edges (vary one pair)
        mid_edges = [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 1.0, 1.0],
            [0.0, 0.0, 0.5, 0.5],
            [1.0, 1.0, 0.5, 0.5],
        ]
        for m in mid_edges:
            init_fracs.append(np.array(m, dtype=float))
        # Additional structured points
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
        
        # Shuffle and take init_budget
        rng.shuffle(init_fracs)
        init_fracs = init_fracs[:init_budget]
        
        for frac in init_fracs:
            if time.time() > deadline:
                break
            x = lo + frac * (hi - lo)
            evaluate(x)

        # Phase 2: Main optimization loop
        F_mut = 0.5
        CR = 0.4
        
        # Precompute front membership set for fast lookup
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current front and contributions
            front_idx, contributions = compute_front_and_contributions()
            if len(front_idx) == 0:
                # Fallback: random exploration
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
                continue
            
            n_total = len(all_F)
            if n_total < 3:
                x = lo + (hi - lo) * rng.random(self.dim)
                evaluate(x)
                continue
            
            # Build selection weights efficiently
            weights = np.zeros(n_total)
            front_set = set(front_idx.tolist())
            for fi, idx in enumerate(front_idx):
                weights[idx] = contributions[fi] + 1e-6
            # Non-front points: give weight based on proximity to reference
            for idx in range(n_total):
                if idx not in front_set:
                    f = all_F[idx]
                    if np.all(f < ref):
                        # Distance to reference in objective space (normalized)
                        dist = (ref - f).min()
                        weights[idx] = 0.05 * (1.0 / (1.0 + dist))
                    else:
                        weights[idx] = 0.005
            
            total_w = weights.sum()
            if total_w > 1e-10:
                probs = weights / total_w
            else:
                probs = np.ones(n_total) / n_total
            
            # DE/rand/1/bin from all points
            parent_indices = rng.choice(n_total, size=3, replace=False, p=probs)
            p1 = all_X[parent_indices[0]]
            p2 = all_X[parent_indices[1]]
            p3 = all_X[parent_indices[2]]
            
            v = p1 + F_mut * (p2 - p3)
            mask = rng.random(self.dim) < CR
            child = np.where(mask, v, p1)
            
            # Adaptive Gaussian perturbation - reduced scale
            progress = used / self.budget
            mut_scale = 0.04 * (1.0 - progress) + 0.005
            child = child + rng.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = np.clip(child, lo, hi)
            
            evaluate(child)

            # Near the end, do local refinement around best front points
            if used > self.budget * 0.7 and used % 5 == 0 and used < self.budget - 1:
                # Pick a random front point and perturb locally
                fi = rng.randint(len(front_idx))
                base = all_X[front_idx[fi]]
                local_child = base + rng.standard_normal(self.dim) * 0.02 * (hi - lo)
                local_child = np.clip(local_child, lo, hi)
                evaluate(local_child)

        return all_X
# EVOLVE-BLOCK-END
