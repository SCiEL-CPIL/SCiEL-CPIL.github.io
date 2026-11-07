"""Combined optimizer: multi-start Nelder-Mead + ridge-aware surrogate + adaptive ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with ridge-aware surrogate and adaptive (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        # Track best
        best_x = rng.uniform(-4, 4, dim)
        best_f = func(best_x)
        calls_used = 1

        def remaining():
            return budget - calls_used

        # ---- Phase 1: Multi-start Nelder-Mead ----
        delta = 3.0
        n_starts = 15
        for s in range(n_starts):
            if not time_ok() or calls_used >= budget - 50:
                break

            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = np.ones(dim) * 3.0
            elif s == 2:
                x0 = -np.ones(dim) * 3.0
            elif s == 3:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            x0 = clip(x0)
            f0 = func(x0)
            calls_used += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            if best_f < -50:
                break

            nm_budget = min(remaining(), 100)
            if nm_budget < 10:
                break

            # Create initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                point = x0.copy()
                point[i] += delta
                point = clip(point)
                simplex.append(point)

            fvals = []
            for pt in simplex:
                f = func(pt)
                calls_used += 1
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

            fvals = np.array(fvals)

            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5
            for _ in range(min(nm_budget, 80)):
                if not time_ok() or calls_used >= budget - 50:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = clip(centroid + alpha * (centroid - simplex[-1]))
                fr = func(xr)
                calls_used += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    xe = clip(centroid + gamma * (centroid - simplex[-1]))
                    fe = func(xe)
                    calls_used += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = clip(centroid + 0.5 * (xr - centroid))
                    else:
                        xc = clip(centroid - 0.5 * (centroid - simplex[-1]))
                    fc = func(xc)
                    calls_used += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            f = func(simplex[i])
                            calls_used += 1
                            fvals[i] = f
                            if f < best_f:
                                best_f = f
                                best_x = simplex[i].copy()
                        spread = np.max(fvals) - np.min(fvals)
                        if spread < 1e-8:
                            break

        # ---- Phase 2: Ridge-aware surrogate model ----
        # Fit a low-rank quadratic to identify ridge direction
        if time_ok() and calls_used < budget - 20:
            x_c = best_x.copy()
            f_c = best_f
            # Sample points around best to fit local quadratic
            n_samples = 2 * dim + 1
            pts = [x_c.copy()]
            fs = [f_c]
            h = 0.3
            for i in range(dim):
                for sign in [1, -1]:
                    p = x_c.copy()
                    p[i] += sign * h
                    p = clip(p)
                    f = func(p)
                    calls_used += 1
                    pts.append(p)
                    fs.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = p.copy()

            pts = np.array(pts)
            fs = np.array(fs)

            # Fit quadratic: f(x) ~ f0 + g^T(x-xc) + (x-xc)^T H (x-xc)
            # Use least squares on the differences
            X = pts - x_c
            f_diff = fs - f_c

            # Build design matrix for quadratic terms
            n_pts = X.shape[0]
            # Linear terms: X (n_pts x dim)
            # Quadratic terms: X_i * X_j for i <= j
            quad_terms = []
            for i in range(dim):
                for j in range(i, dim):
                    quad_terms.append(X[:, i] * X[:, j])
            quad_terms = np.column_stack(quad_terms) if quad_terms else np.zeros((n_pts, 0))

            design = np.hstack([np.ones((n_pts, 1)), X, quad_terms])
            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(design, f_diff, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = None

            if coeffs is not None:
                # Extract gradient and Hessian
                g = coeffs[1:1 + dim]
                # Reconstruct Hessian from quadratic coefficients
                H = np.zeros((dim, dim))
                idx = 1 + dim
                for i in range(dim):
                    for j in range(i, dim):
                        val = coeffs[idx]
                        idx += 1
                        if i == j:
                            H[i, i] = val
                        else:
                            H[i, j] = val / 2.0
                            H[j, i] = val / 2.0

                # Find ridge direction: eigenvector with smallest |eigenvalue| (flat direction)
                try:
                    eigvals, eigvecs = np.linalg.eigh(H)
                    # Ridge is the direction of smallest curvature
                    ridge_idx = np.argmin(np.abs(eigvals))
                    ridge_dir = eigvecs[:, ridge_idx]
                    # Normalize
                    ridge_dir = ridge_dir / (np.linalg.norm(ridge_dir) + 1e-12)

                    # Project best_x onto ridge and orthogonal complement
                    proj_onto_ridge = np.dot(best_x, ridge_dir) * ridge_dir
                    proj_orth = best_x - proj_onto_ridge

                    # Try moving along ridge direction
                    for step in [0.5, 0.3, 0.1, -0.5, -0.3, -0.1]:
                        if not time_ok() or calls_used >= budget - 10:
                            break
                        x_try = clip(best_x + step * ridge_dir)
                        f_try = func(x_try)
                        calls_used += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                except np.linalg.LinAlgError:
                    pass

        # ---- Phase 3: Adaptive (1+1)-ES polish ----
        if time_ok() and calls_used < budget - 5:
            sigma = 0.3
            x = best_x.copy()
            fx = best_f
            successes = 0
            window = 0
            WIN = 5
            for _ in range(budget - calls_used):
                if not time_ok() or calls_used >= budget:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls_used += 1
                window += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                if window >= WIN:
                    if successes / window > 0.2:
                        sigma *= 1.3
                    else:
                        sigma *= 0.7
                    sigma = float(np.clip(sigma, 1e-6, 1.0))
                    successes = 0
                    window = 0

        # ---- Phase 4: Fine-tune with very small steps ----
        if time_ok() and calls_used < budget:
            sigma = 0.01
            x = best_x.copy()
            fx = best_f
            for _ in range(min(30, budget - calls_used)):
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
