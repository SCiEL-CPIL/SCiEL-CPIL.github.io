import numpy as np
import time

class Optimizer:
    """
    BOBYQA-style trust-region optimizer with quadratic model fitting.
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    
    The sharp ridge has anisotropic curvature: narrow across the ridge, 
    elongated along it. Quadratic models can capture this anisotropy 
    and take large informed steps along the valley floor while constraining 
    movement across the narrow width.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f_val = func(x)
                if np.isfinite(f_val):
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global random search to find a good starting point
        n_random = 20
        random_points = []
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break
            random_points.append((f_val, x.copy()))
        
        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0
            random_points.append((f0, x0.copy()))

        # Sort random points by fitness
        random_points.sort(key=lambda t: t[0])
        x_start = random_points[0][1].copy()
        f_start = random_points[0][0]
        
        # Phase 2: BOBYQA-style trust-region optimization
        # We maintain a set of interpolation points and fit a quadratic model
        
        # Initialize interpolation points around x_start
        interp_points = [x_start.copy()]
        interp_values = [f_start]
        
        # Add coordinate points to build initial model
        initial_delta = 0.5
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            for sign in [1, -1]:
                xi = x_start.copy()
                xi[i] += sign * initial_delta
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                interp_points.append(xi.copy())
                interp_values.append(fi)
        
        # Trust region radius
        rho = 0.5
        rho_min = 1e-6
        rho_max = 5.0
        
        max_iter = 400
        iter_count = 0
        
        while iter_count < max_iter and calls < budget and time.time() < deadline:
            # Need at least dim+1 points for a quadratic model in dim dimensions
            # Actually for a full quadratic we need (dim+1)(dim+2)/2 points
            # But we can use a reduced model if we have fewer
            n_points = len(interp_points)
            if n_points < dim + 1:
                # Add more random points
                x_new = x_start + rng.normal(0, rho, size=dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = safe_call(x_new)
                if f_new is None:
                    break
                interp_points.append(x_new.copy())
                interp_values.append(f_new)
                continue
            
            # Build quadratic model: q(x) = c + g^T(x - x0) + 0.5 (x-x0)^T H (x-x0)
            # where x0 is the center of the trust region (current best)
            x0 = best_x.copy()
            f0 = best_f
            
            # Translate points relative to x0
            X = np.array([p - x0 for p in interp_points])  # (n, dim)
            F = np.array(interp_values)
            
            # Design matrix for quadratic model
            # q(x) = c + sum_i g_i * x_i + sum_{i<=j} h_ij * x_i * x_j
            n_params = 1 + dim + dim * (dim + 1) // 2
            
            # Build design matrix
            design = np.ones((n_points, n_params))
            design[:, 1:1+dim] = X
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    design[:, idx] = X[:, i] * X[:, j]
                    idx += 1
            
            # Fit model parameters using least squares
            if n_points >= n_params:
                # Use least squares
                try:
                    params, residuals, rank, sv = np.linalg.lstsq(design, F, rcond=None)
                except np.linalg.LinAlgError:
                    params = np.zeros(n_params)
            else:
                # Use ridge regression for underdetermined case
                try:
                    A = design.T @ design + 1e-6 * np.eye(n_params)
                    b = design.T @ F
                    params = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    params = np.zeros(n_params)
            
            # Extract model parameters
            c = params[0]
            g = params[1:1+dim].copy()
            H_flat = params[1+dim:].copy()
            
            # Reconstruct Hessian (symmetric)
            H = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    val = H_flat[idx]
                    H[i, j] = val
                    if i != j:
                        H[j, i] = val
                    idx += 1
            
            # Make H positive semi-definite for stable optimization
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.maximum(eigvals, 0)
            H = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Solve trust-region subproblem:
            # min q(x) = c + g^T d + 0.5 d^T H d  s.t. ||d|| <= rho
            # where d = x - x0
            
            # Use a simple approach: try gradient descent with trust region constraint
            # First, find the unconstrained minimizer
            if np.linalg.norm(g) < 1e-10 and np.allclose(H, 0):
                # Model is flat
                d_new = np.zeros(dim)
            else:
                try:
                    # Solve H d = -g
                    if np.linalg.matrix_rank(H) >= dim:
                        d_unconstrained = np.linalg.solve(H, -g)
                    else:
                        d_unconstrained = np.linalg.lstsq(H, -g, rcond=None)[0]
                except np.linalg.LinalgError:
                    d_unconstrained = -g / (np.linalg.norm(g) + 1e-10)
                
                # Project to trust region
                norm_d = np.linalg.norm(d_unconstrained)
                if norm_d > rho:
                    d_new = d_unconstrained * (rho / norm_d)
                else:
                    d_new = d_unconstrained
            
            # Ensure step is within bounds
            x_trial = x0 + d_new
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            # Evaluate trial point
            f_trial = safe_call(x_trial)
            if f_trial is None:
                break
            
            # Update trust region based on actual vs predicted improvement
            predicted = c + g @ d_new + 0.5 * d_new @ H @ d_new
            actual = f_trial
            predicted_reduction = f0 - predicted
            actual_reduction = f0 - actual
            
            if predicted_reduction > 1e-10:
                ratio = actual_reduction / predicted_reduction
            else:
                ratio = 1.0 if actual_reduction > 0 else 0.0
            
            if ratio > 0.75:
                # Good agreement, expand trust region
                rho = min(rho * 2.0, rho_max)
            elif ratio > 0.25:
                # Acceptable, keep trust region
                pass
            else:
                # Poor agreement, shrink trust region
                rho = max(rho * 0.5, rho_min)
            
            # Add trial point to interpolation set
            interp_points.append(x_trial.copy())
            interp_values.append(f_trial)
            
            # Remove worst point if set is too large (keep at most 2*dim+1 points)
            max_interp = 2 * dim + 1
            if len(interp_points) > max_interp:
                # Remove the point with the worst function value that is not the best
                idxs = np.argsort(interp_values)
                # Keep best and worst few, remove middle
                remove_idx = idxs[-1]  # Remove worst
                interp_points.pop(remove_idx)
                interp_values.pop(remove_idx)
            
            # Check convergence
            if rho < rho_min:
                # Try restarting with a perturbation
                if calls < budget - 10 and time.time() < deadline:
                    perturbation = rng.normal(0, 0.5, size=dim)
                    x_perturbed = best_x + perturbation
                    x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                    f_perturbed = safe_call(x_perturbed)
                    if f_perturbed is not None:
                        if f_perturbed < best_f:
                            best_x = x_perturbed.copy()
                            best_f = f_perturbed
                        interp_points = [x_perturbed.copy()]
                        interp_values = [f_perturbed]
                        rho = 0.5
                        # Rebuild interpolation set
                        for i in range(dim):
                            if calls >= budget or time.time() > deadline:
                                break
                            for sign in [1, -1]:
                                xi = x_perturbed.copy()
                                xi[i] += sign * rho
                                xi = np.clip(xi, -5.0, 5.0)
                                fi = safe_call(xi)
                                if fi is None:
                                    break
                                interp_points.append(xi.copy())
                                interp_values.append(fi)
                    else:
                        break
                else:
                    break
            
            iter_count += 1
        
        return best_x, best_f
