# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume (fixed by task)
        ref = np.array([1.1, 1.3, 1.1], dtype=np.float64)

        # Track all evaluations
        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            if not np.all(np.isfinite(f)):
                f = np.where(np.isfinite(f), f, 10.0)
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initial diverse sampling (Sobol-like via stratified random)
        n_init = min(24, self.budget)
        for i in range(n_init):
            # Stratified sampling in 4D
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)

        # Helper: compute non-dominated set
        def non_dominated(F):
            if len(F) == 0:
                return np.array([])
            mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        # Helper: hypervolume contribution of a point (simplified for 3D)
        def hypervolume_3d(points, ref_pt):
            """Compute hypervolume of points strictly dominated by ref_pt."""
            # Filter points that are strictly better than ref in all objectives
            valid = np.all(points < ref_pt, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Incremental hypervolume for 3D
            hv = 0.0
            prev_z1 = ref_pt[1]
            prev_z2 = ref_pt[2]
            for p in pts:
                z0, z1, z2 = p
                # Width in obj1
                w0 = ref_pt[0] - z0
                # Area in obj2-obj3 plane for points with obj1 <= z0
                # This is a simplification; for exact HV we need more care
                # For now, use a simple box approximation
                area = max(0, prev_z1 - z1) * max(0, prev_z2 - z2)
                hv += w0 * area
                if z1 < prev_z1:
                    prev_z1 = z1
                if z2 < prev_z2:
                    prev_z2 = z2
            return hv

        # Phase 2: Iterative improvement using weighted-sum local search
        # and gap-filling in objective space
        n_weights = 12
        # Generate weight vectors on a simplex
        weights = []
        for a in np.linspace(0.1, 0.9, 4):
            for b in np.linspace(0.1, 0.9 - a, 3):
                c = 1.0 - a - b
                if c > 0.05:
                    weights.append(np.array([a, b, c]))
        weights = np.array(weights[:n_weights])

        # For each weight direction, do a few local optimizations
        for w in weights:
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            # Start from best point in this direction
            if all_F:
                F_arr = np.array(all_F)
                scores = F_arr @ w
                start_idx = np.argmin(scores)
                x0 = all_X[start_idx].copy()
            else:
                x0 = lo + (hi - lo) * 0.5

            # Local search: small perturbations
            step_size = 0.15
            for it in range(8):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Random perturbation
                dx = rng.standard_normal(self.dim) * step_size
                x_new = x0 + dx
                f_new = evaluate(x_new)
                if f_new is None:
                    break

                # Accept if it improves the weighted sum
                if f_new @ w < (x0 is not None and all_F[-1] is f_new and False or 0):
                    # Actually, compare to current best in this direction
                    pass

                # Better: accept if non-dominated or improves HV
                # Simple acceptance: if it's better than x0 in weighted sum
                if len(all_F) >= 2:
                    f_prev = all_F[-2]
                    if f_new @ w < f_prev @ w:
                        x0 = x_new.copy()
                        step_size *= 0.7
                    else:
                        step_size *= 1.2
                        step_size = min(step_size, 0.3)
                else:
                    x0 = x_new.copy()
                    step_size *= 0.8

        # Phase 3: Fill gaps in objective space
        if eval_count < self.budget:
            # Find the non-dominated set
            F_arr = np.array(all_F)
            nd_idx = non_dominated(F_arr)
            nd_F = F_arr[nd_idx]

            # Try to find points in gaps between non-dominated points
            for i in range(len(nd_F)):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                for j in range(i + 1, len(nd_F)):
                    if eval_count >= self.budget:
                        break
                    # Midpoint in objective space
                    mid_f = (nd_F[i] + nd_F[j]) / 2.0
                    # Map back to decision space via random search
                    for attempt in range(5):
                        if eval_count >= self.budget:
                            break
                        x_rand = lo + (hi - lo) * rng.random(self.dim)
                        f_rand = evaluate(x_rand)
                        if f_rand is None:
                            break
                        # Check if close to mid_f
                        if np.all(f_rand < ref) and np.linalg.norm(f_rand - mid_f) < 0.15:
                            break

        # Final: ensure we haven't exceeded budget
        return all_X[-1] if all_X else np.zeros(self.dim)
# EVOLVE-BLOCK-END
