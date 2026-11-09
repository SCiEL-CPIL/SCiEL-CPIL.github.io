"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid quadratic-model trust-region + Nelder-Mead for anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # Safety margin
        budget = self.budget
        dim = self.dim

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(dim)

        def safe_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Dense initial sampling to build a quadratic model
        # Sample points on a grid to capture anisotropy
        n_init = min(2 * dim + 1, budget // 4)
        points = []
        values = []

        # Origin
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        points.append(x0.copy())
        values.append(f0)

        # Axis-aligned points (positive and negative)
        step_init = 2.0
        for i in range(dim):
            for sign in [1.0, -1.0]:
                xi = np.zeros(dim)
                xi[i] = sign * step_init
                fi = safe_func(xi)
                points.append(xi.copy())
                values.append(fi)
                if calls >= budget:
                    break
            if calls >= budget:
                break

        # Fit quadratic model: f(x) ≈ c + g'x + 0.5*x'H*x
        def fit_quadratic(pts, vals):
            n_pts = len(pts)
            if n_pts < dim + 1:
                return None, None, None
            X = np.array(pts)
            v = np.array(vals)
            # Design matrix: [1, x_1, ..., x_d, x_1*x_1, x_1*x_2, ..., x_d*x_d]
            # Number of quadratic terms: 1 + d + d*(d+1)/2
            n_quad = dim * (dim + 1) // 2
            n_params = 1 + dim + n_quad
            
            if n_pts < n_params:
                # Underdetermined; use least squares anyway
                pass
            
            # Build design matrix
            A = np.zeros((n_pts, n_params))
            A[:, 0] = 1.0
            A[:, 1:1+dim] = X
            
            # Quadratic terms
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    A[:, idx] = X[:, i] * X[:, j]
                    idx += 1
            
            # Solve least squares
            try:
                coeff, residuals, rank, sv = np.linalg.lstsq(A, v, rcond=None)
            except Exception:
                return None, None, None
            
            # Extract parameters
            c = coeff[0]
            g = coeff[1:1+dim]
            
            # Build Hessian from quadratic coefficients
            H = np.zeros((dim, dim))
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        H[i, i] = coeff[idx]  # 0.5*H_ii*x_i^2 = coeff*x_i^2 => H_ii = 2*coeff
                        H[i, i] = 2.0 * coeff[idx]
                    else:
                        H[i, j] = coeff[idx]  # 0.5*H_ij*x_i*x_j + 0.5*H_ji*x_j*x_i = H_ij*x_i*x_j
                        H[j, i] = coeff[idx]
                    idx += 1
            
            # Symmetrize and ensure positive semi-definite
            H = 0.5 * (H + H.T)
            # Regularize for numerical stability
            H += 1e-8 * np.eye(dim)
            
            return c, g, H

        c, g, H = fit_quadratic(points, values)

        # Phase 2: Trust-region optimization using quadratic model
        if c is not None and time.time() < deadline - 5.0:
            # Initial trust region radius
            delta = 1.0
            max_delta = 4.5
            min_delta = 1e-6
            
            # Trust-region iterations
            for tr_iter in range(30):
                if calls >= budget - 10 or time.time() > deadline - 5.0:
                    break
                
                # Solve trust-region subproblem: min c + g'x + 0.5*x'H*x s.t. ||x|| <= delta
                # Use scipy's trust-region solver
                def model_func(x):
                    return c + g @ x + 0.5 * x @ H @ x
                
                def model_grad(x):
                    return g + H @ x
                
                x_center = best_x.copy()
                
                # Shift model to be centered at x_center
                # f(x_center + y) ≈ c' + g'y + 0.5*y'H*y
                # where c' = f(x_center), g' = grad at x_center, H' = H
                # But our model is global, so we need to re-evaluate or shift
                # For simplicity, solve in original coordinates with constraint ||x - x_center|| <= delta
                
                # Use scipy minimize with trust-constr
                try:
                    from scipy.optimize import minimize as scipy_min
                    # Define objective and gradient for the model
                    res = scipy_min(
                        model_func,
                        x_center,
                        jac=model_grad,
                        method='trust-constr',
                        bounds=[(-5.0, 5.0)] * dim,
                        constraints=[{'type': 'ineq', 'fun': lambda x: delta**2 - np.sum((x - x_center)**2)}],
                        options={'maxiter': 20, 'gtol': 1e-4}
                    )
                    x_trial = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    # Fallback: simple gradient step with line search
                    grad = g + H @ x_center
                    norm_grad = np.linalg.norm(grad)
                    if norm_grad < 1e-12:
                        break
                    # Try step in negative gradient direction
                    step_dir = -grad / norm_grad
                    # Find best step size via backtracking
                    alpha = 1.0
                    x_trial = x_center
                    f_trial = model_func(x_trial)
                    for _ in range(20):
                        x_new = x_center + alpha * step_dir
                        x_new = np.clip(x_new, -5.0, 5.0)
                        if np.linalg.norm(x_new - x_center) > delta:
                            alpha *= 0.5
                            continue
                        f_new = model_func(x_new)
                        if f_new < f_trial:
                            x_trial = x_new
                            f_trial = f_new
                            break
                        alpha *= 0.5
                    if np.linalg.norm(x_trial - x_center) < 1e-10:
                        break

                # Evaluate actual function at trial point
                f_actual = safe_func(x_trial)
                
                if not np.isfinite(f_actual):
                    delta *= 0.5
                    continue
                
                # Compute predicted reduction
                f_pred = model_func(x_trial)
                f_actual_center = model_func(x_center)
                predicted_reduction = f_actual_center - f_pred
                actual_reduction = f_actual_center - f_actual
                
                # Acceptance ratio
                if abs(predicted_reduction) > 1e-12:
                    rho_ratio = actual_reduction / predicted_reduction
                else:
                    rho_ratio = 1.0 if actual_reduction > 0 else 0.0
                
                if rho_ratio > 0.75:
                    # Success: expand trust region
                    delta = min(2.0 * delta, max_delta)
                    x_center = x_trial.copy()
                    # Update model by adding new point
                    points.append(x_trial.copy())
                    values.append(f_actual)
                    # Re-fit model
                    c, g, H = fit_quadratic(points, values)
                    if c is None:
                        break
                elif rho_ratio > 0.25:
                    # Moderate: keep same trust region
                    delta = delta
                    x_center = x_trial.copy()
                    points.append(x_trial.copy())
                    values.append(f_actual)
                    c, g, H = fit_quadratic(points, values)
                    if c is None:
                        break
                else:
                    # Failure: shrink trust region
                    delta *= 0.5
                    if delta < min_delta:
                        break

        # Phase 3: Nelder-Mead polish
        if calls < budget - (dim + 1) and time.time() < deadline - 2.0:
            # Initialize simplex around best point
            simplex = [best_x.copy()]
            f_values = [best_fx]
            
            step_nm = max(0.1, 0.5 * delta) if 'delta' in dir() else 0.5
            for i in range(dim):
                xi = best_x.copy()
                xi[i] += step_nm
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_func(xi)
                simplex.append(xi)
                f_values.append(fi)
                if calls >= budget:
                    break
            
            if len(simplex) < dim + 1:
                return best_x, best_fx
            
            # Nelder-Mead parameters
            alpha_nm = 1.0
            gamma_nm = 2.0
            rho_nm = 0.5
            sigma_nm = 0.5
            n = dim + 1
            
            while calls < budget - n and time.time() < deadline - 1.0:
                # Sort simplex
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Convergence check
                if f_values[-1] - f_values[0] < 1e-12:
                    break
                
                x_worst = simplex[-1]
                x_best = simplex[0]
                
                # Centroid
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha_nm * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = safe_func(x_ref)
                
                if f_values[0] <= f_ref < f_values[-2]:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                elif f_ref < f_values[0]:
                    # Expansion
                    x_exp = x_centroid + gamma_nm * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        x_con = x_centroid + rho_nm * (x_ref - x_centroid)
                    else:
                        x_con = x_centroid - rho_nm * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = safe_func(x_con)
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_values[i] = safe_func(simplex[i])
                            if calls >= budget:
                                break
                        if calls >= budget:
                            break

        return best_x, best_fx
# EVOLVE-BLOCK-END
