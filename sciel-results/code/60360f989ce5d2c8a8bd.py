"""Multi-objective optimizer for RE34 using HV-guided search with adaptive mutation and exact 3D HV."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using exact HV-guided parent selection with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hv_3d(self, F, ref):
        """Compute exact hypervolume for 3D points strictly below ref."""
        valid = np.all(F < ref, axis=1)
        Fv = F[valid]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        # Sort by f1 ascending
        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]
        n = len(Fv)

        # Remove dominated points
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if Fv[j, 1] >= Fv[i, 1] and Fv[j, 2] >= Fv[i, 2]:
                    keep[j] = False
        Fv = Fv[keep]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        # Sort by f1 ascending again
        Fv = Fv[np.argsort(Fv[:, 0])]
        n = len(Fv)

        hv = 0.0
        skyline = []  # list of [f2, f3], sorted by f2 asc, f3 strictly decreasing

        for i in range(n):
            f1_i = Fv[i, 0]
            f1_next = Fv[i + 1, 0] if i + 1 < n else ref[0]
            f1_next = min(f1_next, ref[0])

            if f1_next <= f1_i:
                continue

            f2_i = Fv[i, 1]
            f3_i = Fv[i, 2]

            # Update skyline
            new_skyline = []
            for pt in skyline:
                if not (pt[0] >= f2_i and pt[1] >= f3_i):
                    new_skyline.append(pt)
            new_skyline.append([f2_i, f3_i])
            new_skyline.sort(key=lambda x: x[0])
            final_skyline = []
            for pt in new_skyline:
                if not final_skyline or pt[1] < final_skyline[-1][1]:
                    final_skyline.append(pt)
            skyline = final_skyline

            # Compute 2D HV of skyline
            hv_2d = 0.0
            for j in range(len(skyline)):
                f2_j = skyline[j][0]
                f3_j = skyline[j][1]
                if f2_j >= ref[1] or f3_j >= ref[2]:
                    continue
                f2_next = skyline[j + 1][0] if j + 1 < len(skyline) else ref[1]
                f2_next = min(f2_next, ref[1])
                if f2_next > f2_j:
                    hv_2d += (f2_next - f2_j) * (ref[2] - f3_j)

            hv += (f1_next - f1_i) * hv_2d

        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        X_list = []
        F_list = []

        # Phase 1: Sobol-based deterministic sampling for first 128 evaluations
        n_sobol = min(128, self.budget)
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=self.dim, scramble=False)
            sobol_points = sobol.random(n=n_sobol)
        except ImportError:
            # Fallback: use Halton-like deterministic sequence
            sobol_points = np.zeros((n_sobol, self.dim))
            primes = [2, 3, 5, 7, 11]
            for i in range(n_sobol):
                for d in range(self.dim):
                    # Van der Corput sequence
                    f = 1.0
                    n_val = i + 1
                    while n_val > 0:
                        f /= primes[d]
                        n_val //= primes[d]
                    sobol_points[i, d] = f

        for i in range(n_sobol):
            x = lo + span * sobol_points[i]
            f = np.array(problem(x))
            X_list.append(x)
            F_list.append(f)

        used = len(X_list)

        # Phase 2: Local search on top HV-contributing points
        n_local = self.budget - used
        if n_local <= 0:
            return X_list

        step_size = 0.05  # Small axis-aligned steps for local refinement

        while used < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_list)
            n = len(F_arr)

            # Compute full HV
            full_hv = self._hv_3d(F_arr, ref)

            if full_hv <= 0:
                # No valid points; do random exploration
                x = lo + span * rng.random(self.dim)
                fc = np.array(problem(x))
                X_list.append(x)
                F_list.append(fc)
                used += 1
                continue

            # Compute marginal HV contributions for top candidates
            valid = np.all(F_arr < ref, axis=1)
            valid_indices = np.where(valid)[0]

            if len(valid_indices) == 0:
                x = lo + span * rng.random(self.dim)
                fc = np.array(problem(x))
                X_list.append(x)
                F_list.append(fc)
                used += 1
                continue

            # Sort valid points by f1 to find promising candidates
            Fv = F_arr[valid_indices]
            order = np.argsort(Fv[:, 0])
            # Take top 20 by f1 for contribution calculation
            top_candidates = valid_indices[order[:min(20, len(valid_indices))]]

            # Compute exact marginal contributions
            contrib = np.zeros(n)
            for idx in top_candidates:
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                F_without = F_arr[mask]
                hv_without = self._hv_3d(F_without, ref)
                contrib[idx] = full_hv - hv_without

            # Select top 10 points with highest contributions
            pos_mask = contrib > 0
            if np.sum(pos_mask) > 0:
                pos_indices = np.where(pos_mask)[0]
                sorted_contrib = np.sort(contrib[pos_indices])[::-1]
                top_k = min(10, len(pos_indices))
                # Get indices of top-k contributors
                threshold = sorted_contrib[top_k - 1] if top_k > 1 else sorted_contrib[0]
                top_k_indices = pos_indices[contrib[pos_indices] >= threshold]
            else:
                top_k_indices = valid_indices[:min(10, len(valid_indices))]

            # For each top-k point, try axis-aligned perturbations
            evaluated_this_iter = 0
            for idx in top_k_indices:
                if used >= self.budget or time.time() > deadline:
                    break

                parent = X_list[idx].copy()
                # Try each dimension with small perturbation
                for d in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Try positive and negative perturbation
                    for sign in [1, -1]:
                        if used >= self.budget or time.time() > deadline:
                            break

                        child = parent.copy()
                        child[d] += sign * step_size * span[d]
                        child = np.clip(child, lo, hi)

                        # Skip if too close to parent
                        if np.allclose(child, parent, atol=1e-10):
                            continue

                        fc = np.array(problem(child))
                        X_list.append(child)
                        F_list.append(fc)
                        used += 1
                        evaluated_this_iter += 1

                        # Adapt step size based on improvement
                        if np.all(fc < ref):
                            old_F = np.array(F_list[:-1])
                            dominated = np.any(np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1))
                            if not dominated:
                                step_size = max(0.02, step_size * 0.85)
                            else:
                                step_size = min(0.2, step_size * 1.1)
                        else:
                            step_size = min(0.2, step_size * 1.15)

                        break  # Only one perturbation per dimension per parent per iteration

        return X_list
# EVOLVE-BLOCK-END
