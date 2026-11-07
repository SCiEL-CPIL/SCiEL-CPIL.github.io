"""Rewritten optimizer: BOBYQA-style trust-region with quadratic model fitting."""
import time
import numpy as np


class Optimizer:
    """Derivative-free trust-region optimizer using local quadratic models (BOBYQA-style)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        deadline = time.time() + 270.0  # 30s safety margin before 300s kill

        # Call accounting
        calls = [0]
        best_x = [None]
        best_f = [np.inf]

        def safe_call(x):
            """Evaluate func(x) with budget and time guards."""
            if calls[0] >= total_budget:
                return best_f[0]
            if time.time() > deadline:
                return best_f[0]
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                return best_f[0]
            calls[0] += 1
            if v < best_f[0]:
                best_f[0] = v
                best_x[0] = np.array(x, dtype=float)
            return v

        # Trust-region radius
        delta = [1.0]
        # Model points: list of (x, f)
        model_pts = []

        def build_model(pts, center, radius):
            """
            Build a quadratic model from points near center within radius.
            Returns (a, b, c) such that model(x) = a + b @ (x-center) + (x-center) @ Q @ (x-center)
            We use a simplified approach: fit a quadratic via least squares.
            """
            m = len(pts)
            if m < n + 1:
                return None

            # Build design matrix for quadratic: [1, (x-c), (x-c)_i*(x-c)_j for i<=j]
            d = 1 + n + n * (n + 1) // 2
            A = np.zeros((m, d))
            bvec = np.zeros(m)

            for i, (x, fval) in enumerate(pts):
                dx = x - center
                A[i, 0] = 1.0
                A[i, 1:1 + n] = dx
                idx = 1 + n
                for j in range(n):
                    for k in range(j, n):
                        A[i, idx] = dx[j] * dx[k]
                        idx += 1
                bvec[i] = fval

            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
            except np.linalg.LinAlgError:
                return None

            a = coeffs[0]
            b = coeffs[1:1 + n]
            # Reconstruct Q (symmetric)
            Q = np.zeros((n, n))
            idx = 1 + n
            for j in range(n):
                for k in range(j, n):
                    val = coeffs[idx]
                    idx += 1
                    Q[j, k] = val
                    if j != k:
                        Q[k, j] = val

            return (a, b, Q)

        def minimize_quadratic(a, b, Q, center, radius):
            """
            Minimize the quadratic model a + b@(x-c) + (x-c)@Q@(x-c)
            subject to ||x - c|| <= radius.
            Uses a simple projected Newton / trust-region step.
            """
            # If Q is positive definite, the unconstrained minimizer is c - Q^{-1} b
            # Otherwise, we need to handle the trust region constraint.
            try:
                eigvals, eigvecs = np.linalg.eigh(Q)
            except np.linalg.LinAlgError:
                # Fallback: gradient descent step
                grad = b
                norm_grad = np.linalg.norm(grad)
                if norm_grad < 1e-12:
                    return center, a
                step = -grad / norm_grad * min(radius, 1.0)
                return center + step, a - b @ step

            # For the trust-region subproblem, use a simplified approach:
            # If Q is positive definite, find the unconstrained minimizer
            if np.all(eigvals > 1e-10):
                # Unconstrained minimizer: c - Q^{-1} b
                try:
                    Q_inv_b = np.linalg.solve(Q, b)
                except np.linalg.LinAlgError:
                    Q_inv_b = eigvecs @ (eigvecs.T @ b / eigvals)
                x_unc = center - Q_inv_b
                dist = np.linalg.norm(x_unc - center)
                if dist <= radius:
                    return x_unc, a - 0.5 * b @ Q_inv_b
                else:
                    # Project to trust region boundary
                    x_tr = center + (x_unc - center) / dist * radius
                    return x_tr, None  # evaluate at boundary
            else:
                # Indefinite or singular: use steepest descent direction
                grad = b
                norm_grad = np.linalg.norm(grad)
                if norm_grad < 1e-12:
                    return center, a
                step = -grad / norm_grad * min(radius, 1.0)
                return center + step, None

        # Initial random sampling to populate the model
        rng = np.random.default_rng(42)
        num_initial = min(n + 1, 15)
        for _ in range(num_initial):
            if calls[0] >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, size=n)
            fval = safe_call(x)
            model_pts.append((x, fval))

        # Main trust-region loop
        while calls[0] < total_budget and time.time() < deadline:
            # Find current center: best point in model
            if not model_pts:
                x = rng.uniform(-4.5, 4.5, size=n)
                fval = safe_call(x)
                model_pts.append((x, fval))
                continue

            center_idx = int(np.argmin([f for _, f in model_pts]))
            center = np.array(model_pts[center_idx][0], dtype=float)
            center_f = model_pts[center_idx][1]

            # Build model using all points within 2*delta of center
            relevant_pts = [(x, f) for x, f in model_pts if np.linalg.norm(x - center) <= 2.0 * delta[0]]
            if len(relevant_pts) < n + 1:
                relevant_pts = model_pts

            model = build_model(relevant_pts, center, delta[0])
            if model is None:
                # Fallback: take a random step
                x_new = center + rng.standard_normal(n) * delta[0] * 0.5
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = safe_call(x_new)
                model_pts.append((x_new, f_new))
                if f_new < center_f:
                    delta[0] = min(delta[0] * 1.5, 4.5)
                else:
                    delta[0] = max(delta[0] * 0.7, 0.05)
                continue

            a, b, Q = model

            # Solve the trust-region subproblem
            x_trial, f_model = minimize_quadratic(a, b, Q, center, delta[0])
            x_trial = np.clip(x_trial, -5.0, 5.0)

            # Evaluate at trial point
            f_actual = safe_call(x_trial)
            model_pts.append((x_trial, f_actual))

            # Compute predicted reduction
            if f_model is None:
                # Evaluate model at x_trial
                dx = x_trial - center
                f_model = a + b @ dx + dx @ Q @ dx
            rho = (center_f - f_actual) / (center_f - f_model + 1e-12) if center_f != f_model else 0.0

            # Update trust region
            if rho > 0.75:
                delta[0] = min(delta[0] * 2.0, 4.5)
            elif rho > 0.25:
                pass  # keep delta
            else:
                delta[0] = max(delta[0] * 0.5, 0.05)

            # Remove far points from model to keep it local
            if len(model_pts) > 50:
                # Keep only the closest 30 points to the current best
                model_pts.sort(key=lambda p: np.linalg.norm(p[0] - best_x[0]))
                model_pts = model_pts[:30]

            # If we've made good progress, try to refine around the best
            if best_f[0] < center_f - 1e-6:
                # Shift center toward best
                center = 0.5 * center + 0.5 * best_x[0]

        return best_x[0], best_f[0]
