"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned) with PCA-initialized covariance."""
import numpy as np
import time
import scipy.stats


class Optimizer:
    """
    CMA-ES optimizer with PCA-initialized covariance for ill-conditioned landscapes.
    - Initial sampling to estimate search directions
    - PCA-based covariance initialization to align with funnel structure
    - Full CMA-ES with proper covariance adaptation
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin from 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def check():
            return calls_used < budget and time.time() < deadline - 3.0

        # Phase 1: Initial sampling for PCA-based covariance initialization
        # We sample a few points to estimate the landscape orientation.
        n_init = min(30, max(10, budget // 10))
        samples = []
        for i in range(n_init):
            if not check():
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            samples.append((fx, x.copy()))

        # Sort samples by fitness
        samples.sort(key=lambda t: t[0])

        # Initialize CMA-ES parameters
        pop_size = max(6, int(4 + 3 * np.log(dim)))
        if pop_size > (budget - calls_used) // 5:
            pop_size = max(6, (budget - calls_used) // 5)

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize mean at best point
        m = best_x.copy()

        # Initialize sigma and Covariance C using PCA on the best samples
        # We want C to reflect the anisotropy of the landscape.
        # We take the top K samples (best fitness) to estimate the local structure.
        K = min(dim, len(samples))
        if K >= 2:
            top_samples = np.array([s[1] for s in samples[:K]])
            # Center the samples
            center = top_samples.mean(axis=0)
            centered = top_samples - center
            
            # Compute covariance
            try:
                cov_mat = np.cov(centered.T)
                if cov_mat.shape == (dim, dim):
                    # Ensure positive definite
                    eigvals, eigvecs = np.linalg.eigh(cov_mat)
                    # Clip small eigenvalues to avoid numerical instability
                    eigvals = np.maximum(eigvals, 1e-6)
                    
                    # We want to initialize C such that it has a similar shape to the data covariance,
                    # but scaled to a reasonable search step size.
                    # The trace of the data covariance gives an idea of the spread.
                    # We normalize the covariance to have unit trace, then scale by an initial sigma^2.
                    # A reasonable initial sigma for [-5,5] is around 1.0-2.0.
                    # Let's use a base sigma of 1.0, so C should have trace ~ dim * 1.0^2? 
                    # Actually, in CMA-ES, y = m + sigma * C^{1/2} z.
                    # If C is unit trace, the effective step size in principal directions varies.
                    # Let's set C to be the normalized covariance matrix.
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    trace_C = np.trace(C)
                    if trace_C > 1e-8:
                        C /= trace_C
                    else:
                        C = np.eye(dim) / dim
                    sigma = 1.0
                else:
                    C = np.eye(dim)
                    sigma = 1.0
            except Exception:
                C = np.eye(dim)
                sigma = 1.0
        else:
            C = np.eye(dim)
            sigma = 1.0

        # Ensure C is positive definite for Cholesky
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            sigma = 1.0
            C_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Phase 2: CMA-ES main loop
        while calls_used + pop_size <= budget and time.time() < deadline - 5:
            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls_used >= budget or time.time() > deadline - 3:
                    break
                fy = func(y)
                calls_used += 1
                evaluated.append((fy, y.copy()))

            if len(evaluated) < pop_size:
                break

            # Sort by fitness
            evaluated.sort(key=lambda t: t[0])

            # Update best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()

            # Select top mu individuals
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * evaluated[i][1]

            # Calculate step in normalized space
            z_w = np.zeros(dim)
            for i in range(mu):
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                z_w += weights[i] * z_i

            # Update p_sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_w

            # Update sigma
            if h_sigma:
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 0.001, 5.0))

            # Update p_c
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c1) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((c1 / (2 - c1)) - 1)) else 0
            p_c = (1 - c1) * p_c + h_c * np.sqrt(c1 * (2 - c1) * mu_eff) * z_w

            # Update Covariance C
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_c) * c1 * (1 - c1) * C)

            for i in range(mu):
                try:
                    z_i = np.linalg.solve(C_sqrt, (evaluated[i][1] - m) / sigma)
                except np.linalg.LinAlgError:
                    z_i = (evaluated[i][1] - m) / sigma
                C += cmu * weights[i] * np.outer(z_i, z_i)

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update mean
            m = y_w
            m = np.clip(m, -5, 5)

            # Recompute C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

        # Phase 3: BBOB-DF-T style local refinement with quadratic model
        remaining = budget - calls_used
        if remaining >= 20 and best_fx < float('inf'):
            try:
                new_x, new_fx = self._df_t_refine(func, best_x, best_fx, remaining, deadline, rng)
                if new_fx < best_fx:
                    best_fx = new_fx
                    best_x = new_x.copy()
            except Exception:
                pass

        return best_x, best_fx

    def _df_t_refine(self, func, x0, f0, remaining, deadline, rng):
        """
        BBOB-DF-T style derivative-free trust-region refinement.
        Maintains a local quadratic model within a shrinking trust region.
        """
        dim = self.dim
        x = x0.copy()
        f = f0
        self._df_t_calls = 0

        # Trust region radius - start small since CMA-ES has already converged
        delta = 0.1  # initial trust region radius

        # Build initial model points: center + 2*dim axis points
        # This gives us 2*dim + 1 points for a full quadratic model in dim dims
        # Quadratic model: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x-x0)^T H (x-x0)
        # Need at least dim*(dim+3)/2 + 1 points for full quadratic, but we use
        # a simplified approach with fewer points and regularization

        # Use a smaller model: center + 2*dim axis points (2*dim+1 total)
        # This estimates gradient and diagonal Hessian
        model_points = [x.copy()]
        model_vals = [f]

        def eval_point(pt):
            nonlocal f
            if self._df_t_calls >= remaining or time.time() > deadline - 2:
                return None
            pt_clipped = np.clip(pt, -5, 5)
            fv = func(pt_clipped)
            self._df_t_calls += 1
            return (fv, pt_clipped)

        # Evaluate axis points to build local model
        axis_step = delta
        for d in range(dim):
            for sign in [1, -1]:
                pt = x.copy()
                pt[d] += sign * axis_step
                pt = np.clip(pt, -5, 5)
                result = eval_point(pt)
                if result is None:
                    break
                fv, pt_clipped = result
                model_points.append(pt_clipped)
                model_vals.append(fv)
                if fv < f:
                    f = fv
                    x = pt_clipped.copy()

        # If we ran out of budget, return
        if self._df_t_calls >= remaining or time.time() > deadline - 2:
            return x, f

        # Build local model from collected points
        # Use least squares to fit: f(x) ≈ c0 + c^T (x - x0) + 0.5 (x-x0)^T H_diag (x-x0)
        # For simplicity, use a full quadratic with regularization
        n_pts = len(model_points)
        if n_pts < dim + 1:
            return x, f

        # Center at x0 (original center)
        x0 = x.copy()
        # Build design matrix for quadratic model
        # Features: [1, x1-x0_1, ..., x_dim-x0_dim, (x-x0)_i^2, (x-x0)_i*(x-x0)_j for i<j]
        # Full quadratic has 1 + dim + dim + dim*(dim-1)/2 = 1 + 2*dim + dim*(dim-1)/2 params
        # For dim=5: 1 + 10 + 10 = 21 params, need at least 21 points
        # We have at most 2*dim+1 = 11 points, so use regularized least squares

        # Simpler approach: use scipy's least squares on the quadratic model
        # with Tikhonov regularization
        try:
            # Build feature matrix
            X = np.array(model_points)
            dX = X - x0  # (n_pts, dim)
            n_features = 1 + dim + dim  # constant, linear, diagonal quadratic
            # Features: [1, dX_1, ..., dX_dim, dX_1^2, ..., dX_dim^2]
            A = np.hstack([
                np.ones((n_pts, 1)),
                dX,
                dX ** 2
            ])
            b = np.array(model_vals)

            # Tikhonov regularization
            lam = 1e-4
            A_reg = A.copy()
            b_reg = b.copy()
            # Regularize coefficients (not the constant)
            for j in range(1, n_features):
                A_reg[:, j] += lam * (1 if j <= dim else 0) * A_reg[:, j]  # mild reg on linear
                if j > dim:
                    A_reg[:, j] += lam * A_reg[:, j]

            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
            except:
                return x, f

            # Extract model parameters
            c0 = coeffs[0]
            g = coeffs[1:1+dim]  # gradient
            h_diag = coeffs[1+dim:1+2*dim]  # diagonal Hessian (times 0.5)

            # The model is: f(x) ≈ c0 + g^T (x-x0) + 0.5 * sum(h_diag_i * (x-x0)_i^2)
            # Note: h_diag here is actually 2*H_ii from the quadratic term

            # Trust-region subproblem: minimize model subject to ||x - x0|| <= delta
            # For diagonal Hessian, this is separable
            # Try Newton step with trust region clipping

            for tr_iter in range(20):
                if self._df_t_calls >= remaining or time.time() > deadline - 2:
                    break

                # Check if model is still accurate: evaluate at current x
                # If model prediction is far from actual, shrink trust region
                model_pred = c0 + g @ (x - x0) + 0.5 * np.sum(h_diag * (x - x0) ** 2)

                # Compute Newton step: solve H s = -g where H = diag(h_diag)
                # With regularization for ill-conditioning
                H = np.diag(h_diag)
                # Regularize Hessian
                eigvals, eigvecs = np.linalg.eigh(H)
                eigvals_reg = np.maximum(eigvals, 1e-6)
                H_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

                try:
                    s = np.linalg.solve(H_reg, -g)
                except:
                    s = -g / (np.linalg.norm(g) + 1e-10)

                # Clip step to trust region
                s_norm = np.linalg.norm(s)
                if s_norm > delta:
                    s = s * (delta / s_norm)

                # Try the step
                x_new = x + s
                x_new = np.clip(x_new, -5, 5)
                result = eval_point(x_new)
                if result is None:
                    break
                f_new, x_new_clipped = result

                # Accept if improvement
                if f_new < f:
                    # Update center
                    x0 = x
                    x = x_new_clipped
                    f = f_new
                    # Expand trust region slightly
                    delta = min(delta * 1.5, 1.0)
                    # Rebuild model around new center
                    # Keep it simple: just shift x0 and re-evaluate a few points
                    # For efficiency, just continue with updated x0
                    x0 = x.copy()
                    # Re-estimate gradient and Hessian quickly
                    # Use finite differences with small step
                    eps = max(delta * 0.1, 1e-4)
                    g_est = np.zeros(dim)
                    h_est = np.zeros(dim)
                    for d in range(dim):
                        pt_p = x.copy()
                        pt_p[d] += eps
                        pt_p = np.clip(pt_p, -5, 5)
                        res_p = eval_point(pt_p)
                        if res_p is None:
                            break
                        fp, _ = res_p

                        pt_m = x.copy()
                        pt_m[d] -= eps
                        pt_m = np.clip(pt_m, -5, 5)
                        res_m = eval_point(pt_m)
                        if res_m is None:
                            break
                        fm, _ = res_m

                        g_est[d] = (fp - fm) / (2 * eps)
                        h_est[d] = (fp - 2 * f + fm) / (eps ** 2)

                    g = g_est
                    h_diag = h_est

                    # Update model points
                    model_points = [x.copy()]
                    model_vals = [f]
                    for d in range(dim):
                        for sign in [1, -1]:
                            pt = x.copy()
                            pt[d] += sign * eps
                            pt = np.clip(pt, -5, 5)
                            res = eval_point(pt)
                            if res is None:
                                break
                            fv, pt_clipped = res
                            model_points.append(pt_clipped)
                            model_vals.append(fv)
                            if fv < f:
                                f = fv
                                x = pt_clipped.copy()
                else:
                    # Reject: shrink trust region
                    delta *= 0.5
                    if delta < 1e-6:
                        break

        except Exception:
            pass

        return x, f
