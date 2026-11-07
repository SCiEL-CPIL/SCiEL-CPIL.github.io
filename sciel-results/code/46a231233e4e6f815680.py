"""Hypervolume-driven adaptive sampling with Nelder-Mead refinement for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using HV-guided gap filling + NM refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        eval_count = [0]
        max_budget = self.budget

        def evaluate(x):
            eval_count[0] += 1
            if eval_count[0] > max_budget:
                raise ValueError("Budget exceeded")
            return np.array(problem(x), dtype=float)

        def time_ok():
            return time.time() < deadline

        def is_dominated(f, F_set):
            if len(F_set) == 0:
                return False
            for j in range(len(F_set)):
                if np.all(F_set[j] <= f) and np.any(F_set[j] < f):
                    return True
            return False

        def remove_dominated(F_set, X_set):
            n = len(F_set)
            if n <= 1:
                return F_set, X_set
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        keep[i] = False
                        break
            return F_set[keep], X_set[keep]

        def compute_hv(F_nd, ref_pt):
            """Exact 3D hypervolume of non-dominated set with respect to ref_pt."""
            if len(F_nd) == 0:
                return 0.0
            # Only points strictly better than ref in all objectives
            mask = np.all(F_nd < ref_pt, axis=1)
            if not np.any(mask):
                return 0.0
            F = F_nd[mask].copy()
            # Sort by first objective ascending
            order = np.argsort(F[:, 0])
            F = F[order]
            n = len(F)
            if n == 1:
                return float(np.prod(ref_pt - F[0]))
            # For 3D HV: sort by f0, sweep
            # Project onto (f1, f2) plane, compute 2D area under each "layer"
            total = 0.0
            prev_f0 = F[0, 0]
            for i in range(n - 1, -1, -1):
                # The layer for point i extends in f0 from F[i,0] to F[i+1,0] (or ref[0])
                f0_next = F[i+1, 0] if i < n - 1 else ref_pt[0]
                f0_width = max(0.0, f0_next - F[i, 0])
                if f0_width <= 0:
                    continue
                # 2D HV of points 0..i in (f1, f2)
                sub = F[:i+1, 1:3]
                sub_ref = ref_pt[1:3]
                # 2D HV: sort by f1, sweep
                order2 = np.argsort(sub[:, 0])
                sub = sub[order2]
                area = 0.0
                m = len(sub)
                if m == 0:
                    area = 0.0
                elif m == 1:
                    area = max(0.0, (sub_ref[0] - sub[0, 0]) * (sub_ref[1] - sub[0, 1]))
                else:
                    # Sweep from right
                    max_f2 = -np.inf
                    for k in range(m - 1, -1, -1):
                        f1_next = sub[k+1, 0] if k < m - 1 else sub_ref[0]
                        f1_w = max(0.0, f1_next - sub[k, 0])
                        if f1_w <= 0:
                            continue
                        max_f2 = max(max_f2, sub[k, 1])
                        f2_w = max(0.0, sub_ref[1] - max_f2)
                        area += f1_w * f2_w
                total += f0_width * area
            return float(total)

        def nelder_mead(f_obj, x0, max_iter=30, tol=1e-6):
            """Nelder-Mead minimization."""
            n = len(x0)
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.1 * (hi[i] - lo[i])

            fvals = [f_obj(s) for s in simplex]

            for it in range(max_iter):
                if not time_ok() or eval_count[0] >= max_budget:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < tol:
                    break

                centroid = np.mean(simplex[:n], axis=0)
                x_r = centroid + (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = f_obj(x_r)

                if f_r < fvals[0]:
                    x_e = centroid + 2 * (centroid - simplex[-1])
                    x_e = np.clip(x_e, lo, hi)
                    f_e = f_obj(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    x_c = centroid + 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = f_obj(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = 0.5 * (simplex[0] + simplex[i])
                            fvals[i] = f_obj(simplex[i])

            order = np.argmin(fvals)
            return simplex[order], fvals[order]

        def hv_marginal_gain(F_new, F_nd, ref_pt):
            """Estimate HV gain from adding F_new to non-dominated set F_nd."""
            if is_dominated(F_new, F_nd):
                return 0.0
            old_hv = compute_hv(F_nd, ref_pt)
            new_F = np.vstack([F_nd, F_new.reshape(1, -1)])
            new_F, _ = remove_dominated(new_F, np.zeros((len(new_F), 1)))
            new_hv = compute_hv(new_F, ref_pt)
            return max(0.0, new_hv - old_hv)

        # Phase 1: LHS initial sampling for coverage
        n_lhs = 36
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = (perm + np.random.random(n_lhs)) / n_lhs
        lhs_X = lo + (hi - lo) * lhs_X

        all_F = []
        all_X = []
        for x in lhs_X:
            if eval_count[0] >= max_budget or not time_ok():
                break
            f = evaluate(x)
            all_F.append(f)
            all_X.append(x)

        all_F = np.array(all_F)
        all_X = np.array(all_X)
        all_F, all_X = remove_dominated(all_F, all_X)

        # Phase 2: HV-guided adaptive sampling
        # Strategy: at each step, pick a candidate from a set of proposals
        # and evaluate the one with highest marginal HV gain
        n_adaptive = min(80, max_budget - eval_count[0])
        for step in range(n_adaptive):
            if eval_count[0] >= max_budget or not time_ok():
                break

            # Generate candidate points:
            # 1. Midpoints between pairs of non-dominated points in objective space
            # 2. Points along Pareto front interpolation
            # 3. Random points near the "gap" regions
            candidates = []

            n_nd = len(all_F)
            if n_nd >= 2:
                # Pairs of non-dominated points: midpoint in domain
                for i in range(n_nd):
                    for j in range(i + 1, n_nd):
                        x_mid = 0.5 * (all_X[i] + all_X[j])
                        candidates.append(x_mid)
                        # Also try asymmetric
                        x_a = 0.3 * all_X[i] + 0.7 * all_X[j]
                        x_b = 0.7 * all_X[i] + 0.3 * all_X[j]
                        candidates.append(x_a)
                        candidates.append(x_b)

            # Random candidates
            n_rand = 15
            for _ in range(n_rand):
                x_r = lo + np.random.random(dim) * (hi - lo)
                candidates.append(x_r)

            # Candidates from weight-vector directions (scalarized)
            weight_sets = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1/3, 1/3, 1/3]),
                np.array([0.7, 0.15, 0.15]),
                np.array([0.15, 0.7, 0.15]),
                np.array([0.15, 0.15, 0.7]),
                np.array([0.5, 0.25, 0.25]),
                np.array([0.25, 0.5, 0.25]),
                np.array([0.25, 0.25, 0.5]),
            ]
            for w in weight_sets:
                if n_nd > 0:
                    scalar_vals = all_F @ w
                    best_idx = np.argmin(scalar_vals)
                    x_start = all_X[best_idx].copy()
                    # Small perturbation
                    x_cand = x_start + 0.03 * (hi - lo) * np.random.standard_normal(dim)
                    x_cand = np.clip(x_cand, lo, hi)
                    candidates.append(x_cand)

            # Deduplicate candidates (approximate)
            unique_candidates = []
            for c in candidates:
                is_dup = False
                for uc in unique_candidates:
                    if np.allclose(c, uc, atol=0.005):
                        is_dup = True
                        break
                if not is_dup:
                    unique_candidates.append(c)

            # Pick best 2-3 candidates by estimated HV gain
            # Evaluate a few to get real HV gain
            n_eval = min(3, len(unique_candidates), max_budget - eval_count[0])
            if n_eval <= 0:
                break

            # Sort candidates by a heuristic: distance from existing points in objective space
            # We'll just evaluate a random subset and keep the best
            # Actually, let's be smarter: evaluate candidates that are far from existing in domain
            scored_candidates = []
            for c in unique_candidates[:min(50, len(unique_candidates))]:
                # Distance from all existing points
                if len(all_X) > 0:
                    dists = np.linalg.norm(all_X - c, axis=1)
                    min_dist = np.min(dists)
                else:
                    min_dist = 1.0
                scored_candidates.append((min_dist, c))

            scored_candidates.sort(key=lambda x: -x[0])
            top_candidates = [c for _, c in scored_candidates[:n_eval * 2]]

            # Evaluate top candidates and keep the one(s) with best HV gain
            best_gain = -1
            best_x = None
            best_f = None

            for c in top_candidates:
                if eval_count[0] >= max_budget or not time_ok():
                    break
                f = evaluate(c)
                if not is_dominated(f, all_F):
                    gain = hv_marginal_gain(f, all_F, ref)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = c
                        best_f = f

            if best_x is not None:
                all_X = np.vstack([all_X, best_x.reshape(1, -1)])
                all_F = np.vstack([all_F, best_f.reshape(1, -1)])
                all_F, all_X = remove_dominated(all_F, all_X)

        # Phase 3: Nelder-Mead refinement on weighted sums
        all_F, all_X = remove_dominated(all_F, all_X)
        remaining = max_budget - eval_count[0]

        weight_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1/3, 1/3, 1/3]),
            np.array([0.6, 0.2, 0.2]),
            np.array([0.2, 0.6, 0.2]),
            np.array([0.2, 0.2, 0.6]),
            np.array([0.5, 0.3, 0.2]),
            np.array([0.2, 0.5, 0.3]),
            np.array([0.2, 0.3, 0.5]),
            np.array([0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3]),
            np.array([0.3, 0.3, 0.4]),
            np.array([0.5, 0.2, 0.3]),
            np.array([0.3, 0.5, 0.2]),
        ]

        # Add some random weights
        for _ in range(10):
            w = np.random.random(n_obj)
            w /= w.sum()
            weight_vectors.append(w)

        # Deduplicate
        unique_w = []
        for w in weight_vectors:
            is_dup = False
            for uw in unique_w:
                if np.allclose(w, uw, atol=0.02):
                    is_dup = True
                    break
            if not is_dup:
                unique_w.append(w)
        weight_vectors = unique_w

        n_refinements = min(len(weight_vectors), remaining // 12)

        for wi in range(n_refinements):
            if eval_count[0] >= max_budget or not time_ok():
                break

            w = weight_vectors[wi]

            if len(all_F) > 0:
                scalar_vals = all_F @ w
                best_idx = np.argmin(scalar_vals)
                x_start = all_X[best_idx].copy()
            else:
                x_start = lo + 0.5 * (hi - lo)

            def weighted_obj(x, _w=w):
                if eval_count[0] >= max_budget:
                    return np.inf
                f = evaluate(x)
                return f @ _w

            x_opt, f_opt_val = nelder_mead(weighted_obj, x_start, max_iter=25, tol=1e-5)

            if eval_count[0] < max_budget and time_ok():
                f_opt = evaluate(x_opt)
                if not is_dominated(f_opt, all_F):
                    all_X = np.vstack([all_X, x_opt.reshape(1, -1)])
                    all_F = np.vstack([all_F, f_opt.reshape(1, -1)])

        # Phase 4: Final polish with local search around non-dominated points
        all_F, all_X = remove_dominated(all_F, all_X)
        remaining = max_budget - eval_count[0]

        if remaining > 5 and time_ok():
            n_nd = len(all_F)
            n_polish = min(n_nd, remaining // 6)

            for i in range(n_polish):
                if eval_count[0] >= max_budget or not time_ok():
                    break

                x_base = all_X[i].copy()
                f_base = all_F[i].copy()

                for trial in range(5):
                    if eval_count[0] >= max_budget or not time_ok():
                        break
                    scale = 0.03 * (0.7 ** trial) * (hi - lo)
                    x_try = x_base + scale * np.random.standard_normal(dim)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = evaluate(x_try)

                    if not is_dominated(f_try, all_F):
                        if np.all(f_try <= f_base) and np.any(f_try < f_base):
                            all_F[i] = f_try
                            all_X[i] = x_try
                            x_base = x_try
                            f_base = f_try
                        else:
                            all_X = np.vstack([all_X, x_try.reshape(1, -1)])
                            all_F = np.vstack([all_F, f_try.reshape(1, -1)])

        # Final cleanup
        all_F, all_X = remove_dominated(all_F, all_X)

        if len(all_F) > 0:
            return all_X
        else:
            x = lo + 0.5 * (hi - lo)
            f = evaluate(x)
            return np.array([x])
# EVOLVE-BLOCK-END
