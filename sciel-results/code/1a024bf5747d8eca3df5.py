"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive-epsilon gradient + CMA-ES + Nelder-Mead for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def estimate_gradient_adaptive(x, f_current, h_start=1e-3, h_min=1e-6):
            """Adaptive-epsilon gradient estimation: start coarse, refine if noisy."""
            nonlocal calls
            h = h_start
            for attempt in range(3):
                if calls >= budget or time.time() > deadline:
                    return None, h
                grad = np.zeros(n)
                valid = True
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        valid = False
                        break
                    x_plus = x.copy()
                    x_plus[i] += h
                    x_minus = x.copy()
                    x_minus[i] -= h
                    f_plus = func(x_plus)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        valid = False
                        break
                    f_minus = func(x_minus)
                    calls += 1
                    grad[i] = (f_plus - f_minus) / (2 * h)
                if not valid:
                    return None, h
                # Check consistency: if gradient norm is very small relative to f,
                # we might be in a flat region; try smaller h
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12 and attempt < 2:
                    h *= 0.1
                    continue
                break
            return grad, h

        def gradient_descent_adaptive(x0, f0, max_iter=2000):
            """Gradient descent with adaptive epsilon and line search."""
            nonlocal calls, best_x, best_f

            x = x0.copy()
            f = f0
            step_size = 0.05
            max_step = 0.8
            min_step = 1e-9
            h_adaptive = 1e-3
            line_search_max = 15

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                grad, h_adaptive = estimate_gradient_adaptive(x, f, h_start=h_adaptive)
                if grad is None:
                    break

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12:
                    break

                # Backtracking line search
                alpha = step_size
                found = False
                for ls in range(line_search_max):
                    if calls >= budget or time.time() > deadline:
                        break
                    direction = -grad / grad_norm
                    x_trial = x + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1

                    # Armijo condition
                    if f_trial < f - 1e-4 * alpha * grad_norm ** 2:
                        found = True
                        break
                    alpha *= 0.5
                    if alpha < min_step:
                        break

                if not found or alpha < min_step:
                    # Try a smaller step
                    alpha = max(alpha, min_step)
                    direction = -grad / grad_norm
                    x_trial = x + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    if f_trial >= f:
                        break

                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    # Adjust step size based on line search result
                    step_size = min(alpha * 1.5, max_step)
                else:
                    step_size *= 0.5
                    if step_size < min_step:
                        break

            return x, f

        def cma_es(x0, f0, max_gen=100):
            """CMA-ES with small population for exploration."""
            nonlocal calls, best_x, best_f

            lam = 8
            mu = 4
            w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(lam, mu, -1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)

            sigma = 1.0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            Csqrt = np.eye(n)
            Csqrt_inv = np.eye(n)

            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            f_best_local = f0
            x_best_local = x0.copy()

            # Initial population
            pop_x = [x0.copy()]
            pop_f = [f0]

            for i in range(1, lam):
                if calls >= budget or time.time() > deadline:
                    return x_best_local, f_best_local
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (Csqrt @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                if fi < f_best_local:
                    f_best_local = fi
                    x_best_local = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                pop_x.append(xi)
                pop_f.append(fi)

            if calls >= budget:
                return x_best_local, f_best_local

            for gen in range(max_gen):
                if time.time() > deadline or calls >= budget:
                    break

                if sigma < 1e-8:
                    break

                order = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Update mean
                m = np.zeros(n)
                for i in range(mu):
                    m += w[i] * pop_x[i]

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma * chiN) * (Csqrt_inv @ (m - x_mean))

                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (n + 1)) * chiN:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1) / 2)
                else:
                    if np.linalg.norm(ps) / chiN < (1.4 + 2 / (n + 1)):
                        pc = (1 - cc) * pc + cc * (Csqrt_inv @ (m - x_mean) / sigma)
                    else:
                        pc = (1 - cc) * pc + (1 + cc) * (Csqrt_inv @ (m - x_mean) / sigma)

                    hsig = min(1, (np.linalg.norm(ps) / chiN) ** 2 / (1 - (1 - cs) ** (2 * (gen + 1))))
                    rank_one = np.outer(pc, pc)
                    rank_one[pc < 0] *= (1 + (2 / (n + 1) - 1) * hsig)
                    C = (1 + c1 * (hsig - 1)) * C + c1 * rank_one
                    rank_mu = np.zeros((n, n))
                    for i in range(mu):
                        z = Csqrt_inv @ (pop_x[i] - m) / sigma
                        rank_mu += w[i] * np.outer(z, z)
                    C += cmu * rank_mu

                    C = (C + C.T) / 2
                    C += 1e-12 * np.eye(n)

                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-15)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        Csqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                        Csqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        Csqrt = np.eye(n)
                        Csqrt_inv = np.eye(n)

                x_mean = m

                # Generate new population
                new_pop_x = []
                new_pop_f = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        return x_best_local, f_best_local
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * (Csqrt @ z)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls += 1
                    if fi < f_best_local:
                        f_best_local = fi
                        x_best_local = xi.copy()
                        if fi < best_f:
                            best_f = fi
                            best_x = xi.copy()
                    new_pop_x.append(xi)
                    new_pop_f.append(fi)

                pop_x = new_pop_x
                pop_f = new_pop_f

                # Convergence check
                if pop_f[-1] - pop_f[0] < 1e-10:
                    break

            return x_best_local, f_best_local

        def trust_region_quadratic(x0, f0, max_iter=200):
            """Derivative-free trust-region method with quadratic interpolation model.
            
            Builds a local quadratic model from sampled points to capture
            anisotropic curvature of the Rosenbrock valley more accurately
            than simplex-based methods.
            """
            nonlocal calls, best_x, best_f

            # Initialize with points around x0
            points = [x0.copy()]
            fvals = [f0]
            delta = 0.1  # trust region radius

            # Add initial points in coordinate directions
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] += delta
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                points.append(xi.copy())
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                xi = x0.copy()
                xi[i] -= delta
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                points.append(xi.copy())
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                m = len(points)
                if m < n + 1:
                    # Need at least n+1 points to fit gradient
                    break

                # Build quadratic model: f(x) ≈ c + g^T(x-x0) + 0.5*(x-x0)^T H (x-x0)
                # Use points relative to x0
                X = np.array([p - x0 for p in points])  # m x n
                f_rel = np.array(fvals) - f0

                # Solve for gradient and Hessian using least squares
                # Model: f_rel ≈ g^T * X + 0.5 * sum_ij H_ij * X_i * X_j
                # Build design matrix
                # Columns: X (n cols), then X_i*X_j for i<=j (n(n+1)/2 cols)
                quad_terms = []
                for i in range(n):
                    for j in range(i, n):
                        quad_terms.append(X[:, i] * X[:, j])
                quad_matrix = np.column_stack(quad_terms) if quad_terms else np.zeros((m, 0))
                
                A = np.column_stack([X, quad_matrix])
                
                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, f_rel, rcond=None)
                except np.linalg.LinAlgError:
                    break

                g_model = coeffs[:n]
                h_coeffs = coeffs[n:]
                
                # Reconstruct Hessian
                H_model = np.zeros((n, n))
                idx = 0
                for i in range(n):
                    for j in range(i, n):
                        H_model[i, j] = h_coeffs[idx]
                        if i != j:
                            H_model[j, i] = h_coeffs[idx]
                        idx += 1
                
                # Ensure H is symmetric and positive semi-definite
                H_model = (H_model + H_model.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(H_model)
                    eigvals = np.maximum(eigvals, -1e-10)  # allow small negative
                    H_model = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    H_model = np.eye(n) * 1e-3

                # Solve trust-region subproblem: min g^T s + 0.5 s^T H s s.t. ||s|| <= delta
                # Use eigendecomposition of H
                try:
                    eigvals, eigvecs = np.linalg.eigh(H_model)
                except:
                    eigvals = np.ones(n) * 1e-3
                    eigvecs = np.eye(n)

                # Transform to eigenbasis
                g_eig = eigvecs.T @ g_model

                # Find optimal step in eigenbasis
                # For each eigen-direction, optimal step is -g_i / lambda_i if lambda_i > 0
                # Subject to ||s_eig|| <= delta
                s_eig = np.zeros(n)
                for i in range(n):
                    if eigvals[i] > 1e-12:
                        s_eig[i] = -g_eig[i] / eigvals[i]
                    else:
                        s_eig[i] = 0

                # Scale to fit within trust region
                s_norm = np.linalg.norm(s_eig)
                if s_norm > delta:
                    s_eig *= delta / s_norm

                # Transform back
                s = eigvecs @ s_eig
                x_trial = x0 + s
                x_trial = np.clip(x_trial, -5, 5)
                
                # Evaluate trial point
                f_trial = func(x_trial)
                calls += 1

                # Check improvement
                # Predicted reduction from model
                s_local = x_trial - x0
                f_pred = f0 + g_model @ s_local + 0.5 * s_local @ H_model @ s_local
                f_actual = f_trial
                rho = (f0 - f_actual) / max(abs(f0 - f_pred), 1e-12)

                if rho > 0.25:
                    # Accept step, expand trust region
                    x0 = x_trial
                    f0 = f_trial
                    delta = min(delta * 2.0, 1.0)
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                    
                    # Add trial point to model points
                    points.append(x_trial.copy())
                    fvals.append(f_trial)
                    
                    # Keep only the best m points (remove worst)
                    if len(points) > 2 * n + 1:
                        # Remove the point with highest f value that's not x0
                        idxs = np.argsort(fvals)
                        # Keep x0 (index 0) and the best 2n points
                        keep = [0] + list(idxs[1:2*n])
                        points = [points[i] for i in keep]
                        fvals = [fvals[i] for i in keep]
                        x0 = points[0].copy()
                        f0 = fvals[0]
                elif rho > 0.1:
                    # Accept step, keep trust region same
                    x0 = x_trial
                    f0 = f_trial
                    if f0 < best_f:
                        best_f = f0
                        best_x = x0.copy()
                    points.append(x_trial.copy())
                    fvals.append(f_trial)
                    if len(points) > 2 * n + 1:
                        idxs = np.argsort(fvals)
                        keep = [0] + list(idxs[1:2*n])
                        points = [points[i] for i in keep]
                        fvals = [fvals[i] for i in keep]
                        x0 = points[0].copy()
                        f0 = fvals[0]
                else:
                    # Reject step, shrink trust region
                    delta *= 0.5
                    if delta < 1e-8:
                        break

            return x0, f0

        # Phase 1: Evaluate origin and a few random points
        f_origin = func(np.zeros(n))
        calls += 1
        best_x = np.zeros(n).copy()
        best_f = f_origin

        for i in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES for exploration (use ~30% of remaining budget)
        remaining = budget - calls
        cma_budget = int(0.3 * remaining)
        max_gen = max(15, cma_budget // 8)
        x_cma, f_cma = cma_es(best_x.copy(), best_f, max_gen=max_gen)

        if f_cma < best_f:
            best_f = f_cma
            best_x = x_cma.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Adaptive gradient descent for fast local convergence
        x_grad, f_grad = gradient_descent_adaptive(best_x.copy(), best_f, max_iter=1500)

        if f_grad < best_f:
            best_f = f_grad
            best_x = x_grad.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Trust-region quadratic model for final polish
        x_tr, f_tr = trust_region_quadratic(best_x.copy(), best_f, max_iter=200)

        if f_tr < best_f:
            best_f = f_tr
            best_x = x_tr.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 5: Multiple restarts with remaining budget
        restarts_done = 0
        max_restarts = 4
        while restarts_done < max_restarts and budget - calls > 80:
            # Try different restart strategies
            if restarts_done % 2 == 0:
                x_restart = best_x + 0.3 * rng.standard_normal(n)
            else:
                # Try a point in a different direction
                angle = rng.uniform(0, 2 * np.pi)
                offset = np.zeros(n)
                offset[0] = 0.5 * np.cos(angle)
                offset[1] = 0.5 * np.sin(angle)
                x_restart = best_x + offset
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = func(x_restart)
            calls += 1
            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()

            # Short CMA-ES exploration
            if budget - calls > 40:
                x_cma2, f_cma2 = cma_es(x_restart, f_restart, max_gen=10)
                if f_cma2 < best_f:
                    best_f = f_cma2
                    best_x = x_cma2.copy()

            # Gradient descent polish
            if calls < budget and time.time() < deadline and budget - calls > 30:
                x_grad2, f_grad2 = gradient_descent_adaptive(best_x.copy(), best_f, max_iter=200)
                if f_grad2 < best_f:
                    best_f = f_grad2
                    best_x = x_grad2.copy()

            # Trust region polish
            if calls < budget and time.time() < deadline and budget - calls > 20:
                x_tr2, f_tr2 = trust_region_quadratic(best_x.copy(), best_f, max_iter=50)
                if f_tr2 < best_f:
                    best_f = f_tr2
                    best_x = x_tr2.copy()

            restarts_done += 1

        # Final polish with whatever budget remains
        if budget - calls > 15 and time.time() < deadline:
            x_final, f_final = trust_region_quadratic(best_x.copy(), best_f, max_iter=30)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
