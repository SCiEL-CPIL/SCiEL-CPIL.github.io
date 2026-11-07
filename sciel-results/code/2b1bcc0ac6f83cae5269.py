"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free Sequential Quadratic Programming (DF-SQP) with trust-region
    and adaptive restarts for 5-dimensional black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def check_time():
            return time.time() > deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse global search with random sampling + local refinement
        # We'll use a strategy: random starts, then local quadratic model optimization

        def local_optimize(x0, remaining_budget):
            """DF-SQP: build local quadratic model, solve trust region, iterate."""
            nonlocal calls, best_f, best_x
            if remaining_budget < 10:
                return

            # Build initial model with coordinate perturbations
            # Use a small local search with quadratic interpolation
            base = np.clip(x0.copy(), -5.0, 5.0)
            f_base = evaluate(base)

            # Adaptive step size
            delta = 1.0
            model_points = [base.copy()]
            model_fvals = [f_base]

            # Collect points around base
            for _ in range(min(2 * dim + 1, max(0, remaining_budget - 2))):
                if calls >= budget or check_time():
                    break
                # Try random perturbations with adaptive radius
                for attempt in range(5):
                    if calls >= budget or check_time():
                        break
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    x_try = base + delta * direction
                    x_try = np.clip(x_try, -5.0, 5.0)
                    # Avoid duplicates
                    if all(np.linalg.norm(x_try - p) > 1e-6 for p in model_points):
                        f_try = evaluate(x_try)
                        model_points.append(x_try.copy())
                        model_fvals.append(f_try)
                        if f_try < f_base:
                            base = x_try.copy()
                            f_base = f_try
                        break
                else:
                    # Shrink delta if no new point found
                    delta *= 0.5

            if len(model_points) < dim + 1:
                return

            # Fit quadratic model: f(x) ≈ c + g^T(x - x0) + 0.5 (x-x0)^T H (x-x0)
            # Use least squares to fit from model points
            x0_ref = base.copy()
            n_pts = len(model_points)
            # Design matrix for quadratic terms
            # Features: [1, (x-x0)_i for i in dim, (x-x0)_i*(x-x0)_j for i<=j]
            n_quad = dim * (dim + 1) // 2
            n_features = 1 + dim + n_quad

            A = np.zeros((n_pts, n_features))
            b = np.array(model_fvals)

            quad_idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        # (x-x0)_i^2
                        pass
                    quad_idx += 1

            # Build feature matrix
            for k in range(n_pts):
                d = model_points[k] - x0_ref
                A[k, 0] = 1.0
                for i in range(dim):
                    A[k, 1 + i] = d[i]
                qi = 0
                for i in range(dim):
                    for j in range(i, dim):
                        A[k, 1 + dim + qi] = d[i] * d[j]
                        qi += 1

            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return

            c = coeffs[0]
            g = coeffs[1:1 + dim].copy()
            # Reconstruct Hessian from quadratic coefficients
            H = np.zeros((dim, dim))
            qi = 0
            for i in range(dim):
                for j in range(i, dim):
                    H[i, j] = coeffs[1 + dim + qi]
                    H[j, i] = coeffs[1 + dim + qi]
                    qi += 1
            # The quadratic term is 0.5 * d^T H d, but our feature is d_i * d_j
            # So the actual Hessian contribution: for i==j, coeff = H_ii (since 0.5*H_ii*d_i^2)
            # Wait, let me reconsider. The model is c + g^T d + sum_{i<=j} q_ij * d_i * d_j
            # This equals c + g^T d + 0.5 * d^T H d where H_ii = 2*q_ii, H_ij = q_ij for i!=j
            # Actually: 0.5 * d^T H d = 0.5 * sum_i H_ii d_i^2 + sum_{i<j} H_ij d_i d_j
            # Our model: sum_{i<=j} q_ij d_i d_j = sum_i q_ii d_i^2 + sum_{i<j} q_ij d_i d_j
            # So H_ii = 2*q_ii, H_ij = q_ij for i != j
            for i in range(dim):
                H[i, i] *= 2.0

            # Regularize Hessian for stability
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.clip(eigvals, -10.0, 10.0)
            H = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Trust region optimization
            tr_radius = delta * 2.0
            x_current = base.copy()
            f_current = f_base

            # Iterate trust region
            for tr_iter in range(min(20, max(0, budget - calls))):
                if calls >= budget or check_time():
                    break

                # Solve: min 0.5*d^T H d + g^T d  s.t. ||d|| <= tr_radius
                # Use projected gradient or direct solve
                # Interior point: if unconstrained min is within radius, use it
                try:
                    if np.linalg.matrix_rank(H) >= dim:
                        d_unconstrained = np.linalg.solve(-H, g)
                    else:
                        d_unconstrained = -0.1 * g
                except np.linalg.LinAlgError:
                    d_unconstrained = -0.1 * g

                d_norm = np.linalg.norm(d_unconstrained)
                if d_norm > tr_radius:
                    d_step = d_unconstrained * (tr_radius / d_norm)
                else:
                    d_step = d_unconstrained

                x_new = x_current + d_step
                x_new = np.clip(x_new, -5.0, 5.0)

                f_new = evaluate(x_new)

                # Check if step was useful
                predicted = 0.5 * d_step @ H @ d_step + g @ d_step
                actual = f_new - f_current
                rho = actual / (abs(predicted) + 1e-12) if abs(predicted) > 1e-12 else 1.0

                if rho > 0.25:
                    x_current = x_new.copy()
                    f_current = f_new
                    tr_radius = min(tr_radius * 1.5, 3.0)
                    # Update model with new point
                    model_points.append(x_new.copy())
                    model_fvals.append(f_new)
                    # Refit model
                    if len(model_points) <= 30:
                        A2 = np.zeros((len(model_points), n_features))
                        b2 = np.array(model_fvals)
                        x0_ref = x_current.copy()
                        for k in range(len(model_points)):
                            d = model_points[k] - x0_ref
                            A2[k, 0] = 1.0
                            for i in range(dim):
                                A2[k, 1 + i] = d[i]
                            qi = 0
                            for i in range(dim):
                                for j in range(i, dim):
                                    A2[k, 1 + dim + qi] = d[i] * d[j]
                                    qi += 1
                        try:
                            coeffs, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)
                            c = coeffs[0]
                            g = coeffs[1:1 + dim].copy()
                            H = np.zeros((dim, dim))
                            qi = 0
                            for i in range(dim):
                                for j in range(i, dim):
                                    H[i, j] = coeffs[1 + dim + qi]
                                    H[j, i] = coeffs[1 + dim + qi]
                                    qi += 1
                            for i in range(dim):
                                H[i, i] *= 2.0
                            eigvals2, eigvecs2 = np.linalg.eigh(H)
                            eigvals2 = np.linalg.clip if hasattr(np.linalg, 'clip') else np.clip(eigvals2, -10.0, 10.0)
                            H = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T
                        except:
                            pass
                else:
                    tr_radius *= 0.5
                    if tr_radius < 1e-6:
                        break

        # Multi-start strategy
        n_starts = 5
        per_start_budget = budget // n_starts

        for s in range(n_starts):
            if calls >= budget or check_time():
                break

            if s == 0:
                x_start = np.zeros(dim)
            else:
                x_start = rng.uniform(-5.0, 5.0, dim)

            # Evaluate start
            if calls < budget and not check_time():
                f_start = evaluate(x_start)

            # Local refinement
            remaining = budget - calls
            local_optimize(x_start, remaining)

            # If we have budget left, do a few more random probes around best
            if calls < budget and not check_time() and best_x is not None:
                for _ in range(min(10, budget - calls)):
                    if calls >= budget or check_time():
                        break
                    x_probe = best_x + rng.standard_normal(dim) * 0.5
                    x_probe = np.clip(x_probe, -5.0, 5.0)
                    evaluate(x_probe)

        return best_x, best_f
# EVOLVE-BLOCK-END
