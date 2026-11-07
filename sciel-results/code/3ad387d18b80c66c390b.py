"""Ridge-aware two-phase optimizer: active quadratic fit -> (1+1)-ES cross-ridge -> ridge line-search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Ridge-aware optimizer: identify ridge via active sampling, (1+1)-ES in orthogonal complement, ridge line-search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = rng.uniform(-4, 4, dim)
        best_f = func(best_x)
        calls_used = 1

        # ============ PHASE 1: Active sampling to identify ridge ============
        # Coarse scan to find a good center
        scan_points = [np.zeros(dim)]
        for i in range(dim):
            scan_points.append(np.full(dim, 2.0))
            scan_points.append(np.full(dim, -2.0))
            p = np.zeros(dim)
            p[i] = 3.0
            scan_points.append(p)
            p = np.zeros(dim)
            p[i] = -3.0
            scan_points.append(p)
        for _ in range(8):
            scan_points.append(rng.uniform(-4, 4, dim))

        scan_results = []
        for p in scan_points:
            if not time_ok() or calls_used >= budget - 100:
                break
            p = clip(p)
            f = func(p)
            calls_used += 1
            scan_results.append((f, p))
            if f < best_f:
                best_f = f
                best_x = p.copy()

        scan_results.sort(key=lambda x: x[0])
        center = scan_results[0][1].copy()
        center_f = scan_results[0][0]

        # Local quadratic fit around center
        h = 0.5
        quad_pts = [center.copy()]
        quad_fs = [center_f]
        for i in range(dim):
            for sign in [1, -1]:
                p = center.copy()
                p[i] += sign * h
                p = clip(p)
                f = func(p)
                calls_used += 1
                quad_pts.append(p)
                quad_fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = p.copy()

        quad_pts = np.array(quad_pts)
        quad_fs = np.array(quad_fs)

        # Add off-diagonal samples for full Hessian
        if time_ok() and calls_used < budget - 80:
            extra_pts = []
            extra_fs = []
            pairs_sampled = 0
            for i in range(dim):
                for j in range(i + 1, dim):
                    if pairs_sampled >= 10:
                        break
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            if pairs_sampled >= 10:
                                break
                            p = center.copy()
                            p[i] += si * h
                            p[j] += sj * h
                            p = clip(p)
                            f = func(p)
                            calls_used += 1
                            extra_pts.append(p)
                            extra_fs.append(f)
                            pairs_sampled += 1
                            if f < best_f:
                                best_f = f
                                best_x = p.copy()

            if len(extra_pts) > 0:
                all_pts = np.vstack([quad_pts, np.array(extra_pts)])
                all_fs = np.concatenate([quad_fs, np.array(extra_fs)])
            else:
                all_pts = quad_pts
                all_fs = quad_fs
        else:
            all_pts = quad_pts
            all_fs = quad_fs

        X_all = all_pts - center
        f_diff_all = all_fs - center_f
        n_all = X_all.shape[0]

        # Build full quadratic design matrix
        quad_terms = []
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    quad_terms.append(0.5 * X_all[:, i] ** 2)
                else:
                    quad_terms.append(X_all[:, i] * X_all[:, j])
        quad_terms = np.column_stack(quad_terms)

        design_full = np.hstack([
            np.ones((n_all, 1)),
            X_all,
            quad_terms
        ])

        try:
            coeffs_full, _, _, _ = np.linalg.lstsq(design_full, f_diff_all, rcond=None)
            H_full = np.zeros((dim, dim))
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    val = coeffs_full[idx]
                    idx += 1
                    if i == j:
                        H_full[i, i] = val
                    else:
                        H_full[i, j] = val
                        H_full[j, i] = val
        except np.linalg.LinAlgError:
            # Fallback: diagonal only
            H_full = np.eye(dim)

        # Find ridge direction: eigenvector with smallest |eigenvalue|
        try:
            eigvals, eigvecs = np.linalg.eigh(H_full)
            ridge_idx = np.argmin(np.abs(eigvals))
            ridge_dir = eigvecs[:, ridge_idx]
            ridge_dir = ridge_dir / (np.linalg.norm(ridge_dir) + 1e-12)
        except np.linalg.LinAlgError:
            ridge_dir = np.zeros(dim)
            ridge_dir[0] = 1.0

        # Build orthonormal basis for orthogonal complement of ridge_dir
        Q = np.eye(dim)
        Q[-1] = ridge_dir
        for i in range(dim - 1):
            Q[i] -= np.dot(Q[i], ridge_dir) * ridge_dir
            norm = np.linalg.norm(Q[i])
            if norm > 1e-10:
                Q[i] /= norm
            else:
                v = rng.standard_normal(dim)
                v -= np.dot(v, ridge_dir) * ridge_dir
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    Q[i] = v / norm
                    Q[i] -= np.dot(Q[i], ridge_dir) * ridge_dir
                    Q[i] /= (np.linalg.norm(Q[i]) + 1e-12)

        orth_basis = Q[:, :dim - 1]  # (dim, dim-1)

        # Project best_x onto ridge and orthogonal complement
        t_best = np.dot(best_x - center, ridge_dir)
        z_best = orth_basis.T @ (best_x - center - t_best * ridge_dir)

        # ============ PHASE 2: (1+1)-ES in the orthogonal complement ============
        n_orth = dim - 1
        if n_orth > 0:
            sigma_z = 1.0
            mean_z = z_best.copy()
            x_mean = center + orth_basis @ mean_z + t_best * ridge_dir
            x_mean = clip(x_mean)
            f_mean = func(x_mean)
            calls_used += 1
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

            successes = 0
            window = 0
            WIN = 5

            for _ in range(budget - calls_used - 30):
                if not time_ok() or calls_used >= budget - 30:
                    break

                z_trial = mean_z + sigma_z * rng.standard_normal(n_orth)
                x_trial = center + orth_basis @ z_trial + t_best * ridge_dir
                x_trial = clip(x_trial)
                f_trial = func(x_trial)
                calls_used += 1

                if f_trial < f_mean:
                    mean_z = z_trial
                    f_mean = f_trial
                    successes += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()

                window += 1
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma_z *= 1.3
                    else:
                        sigma_z *= 0.7
                    sigma_z = float(np.clip(sigma_z, 1e-8, 5.0))
                    successes = 0
                    window = 0

        # ============ PHASE 3: Ridge line-search ============
        # Fine-tune along the ridge direction, keeping orthogonal component fixed
        t_current = np.dot(best_x - center, ridge_dir)
        orth_component = best_x - t_current * ridge_dir

        t_lo = t_current - 3.0
        t_hi = t_current + 3.0
        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0

        for _ in range(min(30, budget - calls_used)):
            if not time_ok() or calls_used >= budget:
                break

            t1 = t_lo + inv_phi * (t_hi - t_lo)
            t2 = t_hi - inv_phi * (t_hi - t_lo)

            x1 = clip(orth_component + t1 * ridge_dir)
            x2 = clip(orth_component + t2 * ridge_dir)

            f1 = func(x1)
            calls_used += 1
            f2 = func(x2)
            calls_used += 1

            if f1 < best_f:
                best_f = f1
                best_x = x1.copy()
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()

            if f1 < f2:
                t_hi = t2
            else:
                t_lo = t1

            if abs(t_hi - t_lo) < 1e-6:
                break

        # ============ PHASE 4: Final fine-tune ============
        if time_ok() and calls_used < budget:
            sigma = 0.02
            x = best_x.copy()
            fx = best_f
            for _ in range(min(20, budget - calls_used)):
                if not time_ok() or calls_used >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
