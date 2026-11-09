import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: The evaluator kills at 330s, but we want to return well before.
        # Let's aim to finish within 280s to be safe.
        start_time = time.time()
        deadline = start_time + 280
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Ensure x is a numpy array of correct shape and within bounds
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Build initial sample set for quadratic model
        # We need at least dim+1 points for a linear model, but for quadratic we want more
        # Let's start with a small set around origin and expand if needed
        
        # Initial point: try origin first, then random points if it's bad
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        
        # Build initial sample set: x0 plus points in coordinate directions
        delta_init = 1.0
        samples_x = [x0.copy()]
        samples_f = [f0]
        
        for i in range(dim):
            if time.time() > deadline or calls >= budget:
                break
            xi = x0.copy()
            xi[i] += delta_init
            xi = np.clip(xi, -5.0, 5.0)
            fi = safe_func(xi)
            samples_x.append(xi)
            samples_f.append(fi)
            
            xi_neg = x0.copy()
            xi_neg[i] -= delta_init
            xi_neg = np.clip(xi_neg, -5.0, 5.0)
            if time.time() > deadline or calls >= budget:
                break
            fi_neg = safe_func(xi_neg)
            samples_x.append(xi_neg)
            samples_f.append(fi_neg)
        
        # If we have too few samples, add random points
        min_samples = 2 * dim + 1
        while len(samples_x) < min_samples and calls < budget and time.time() < deadline:
            x_rand = rng.uniform(-4.0, 4.0, dim)
            x_rand = np.clip(x_rand, -5.0, 5.0)
            f_rand = safe_func(x_rand)
            samples_x.append(x_rand)
            samples_f.append(f_rand)
        
        # Trust-Region Reflective Algorithm
        # Maintain a trust region radius and a quadratic model
        trust_radius = 2.0  # Initial trust region radius
        max_trust_radius = 9.0  # Max trust region (covers [-5,5])
        min_trust_radius = 1e-6
        
        # Find best point in current samples
        best_idx = np.argmin(samples_f)
        x_center = samples_x[best_idx].copy()
        f_center = samples_f[best_idx]
        
        # Build quadratic model from samples
        # m(x) = c + g^T(x - x_center) + 0.5 * (x - x_center)^T H (x - x_center)
        # We'll fit this using least squares
        
        def fit_quadratic_model(samples_x, samples_f, x_center):
            n_samples = len(samples_x)
            dim = len(x_center)
            
            # Build design matrix for quadratic model
            # Features: 1, (x-xc)_i, (x-xc)_i*(x-xc)_j for i<=j
            # Number of features: 1 + dim + dim*(dim+1)/2
            n_quad = dim * (dim + 1) // 2
            n_features = 1 + dim + n_quad
            
            X = np.zeros((n_samples, n_features))
            y = np.zeros(n_samples)
            
            for k in range(n_samples):
                dx = samples_x[k] - x_center
                X[k, 0] = 1.0
                X[k, 1:1+dim] = dx
                idx = 1 + dim
                for i in range(dim):
                    for j in range(i, dim):
                        X[k, idx] = dx[i] * dx[j]
                        idx += 1
                y[k] = samples_f[k]
            
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback to linear model
                X_lin = np.zeros((n_samples, 1 + dim))
                for k in range(n_samples):
                    dx = samples_x[k] - x_center
                    X_lin[k, 0] = 1.0
                    X_lin[k, 1:] = dx
                try:
                    coeffs_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
                    # Pad with zeros for quadratic terms
                    coeffs = np.zeros(n_features)
                    coeffs[:1+dim] = coeffs_lin
                    return coeffs
                except:
                    return None
            
            return coeffs
        
        def predict_model(coeffs, x, x_center):
            if coeffs is None:
                return float('inf')
            dx = x - x_center
            n_quad = dim * (dim + 1) // 2
            n_features = 1 + dim + n_quad
            
            # Build feature vector
            feat = np.zeros(n_features)
            feat[0] = 1.0
            feat[1:1+dim] = dx
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    feat[idx] = dx[i] * dx[j]
                    idx += 1
            
            return float(feat @ coeffs)
        
        def minimize_model_in_trust_region(coeffs, x_center, radius):
            if coeffs is None:
                return None, float('inf')
            
            dim = len(x_center)
            n_quad = dim * (dim + 1) // 2
            n_features = 1 + dim + n_quad
            
            # Extract gradient and Hessian from coefficients
            g = coeffs[1:1+dim].copy()
            
            # Reconstruct Hessian from quadratic coefficients
            # H[i,j] = coeff[i,j] + coeff[j,i] for i != j, H[i,i] = 2*coeff[i,i]
            H = np.zeros((dim, dim))
            idx = 1 + dim
            for i in range(dim):
                for j in range(i, dim):
                    val = coeffs[idx]
                    if i == j:
                        H[i, i] = 2.0 * val
                    else:
                        H[i, j] = val
                        H[j, i] = val
                    idx += 1
            
            # Ensure H is symmetric
            H = (H + H.T) / 2.0
            
            # Solve for unconstrained minimizer of model: H*p = -g
            try:
                p_uncon = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                p_uncon = np.linalg.lstsq(H, -g, rcond=None)[0]
            
            p_norm = np.linalg.norm(p_uncon)
            
            if p_norm <= radius:
                # Unconstrained minimizer is inside trust region
                return p_uncon, predict_model(coeffs, x_center + p_uncon, x_center)
            else:
                # Constrained minimizer: minimize model subject to ||p|| <= radius
                # Use Lagrange multiplier method
                # H*p + g = lambda*p => (H - lambda*I)*p = -g
                # Find lambda such that ||p|| = radius
                
                # Bisection on lambda
                lambda_low = 0.0
                lambda_high = np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-6
                
                for _ in range(50):
                    lambda_mid = (lambda_low + lambda_high) / 2.0
                    A = H - lambda_mid * np.eye(dim)
                    try:
                        p = np.linalg.solve(A, -g)
                    except np.linalg.LinAlgError:
                        p = np.linalg.lstsq(A, -g, rcond=None)[0]
                    p_norm = np.linalg.norm(p)
                    if p_norm > radius:
                        lambda_low = lambda_mid
                    else:
                        lambda_high = lambda_mid
                
                lambda_opt = (lambda_low + lambda_high) / 2.0
                A = H - lambda_opt * np.eye(dim)
                try:
                    p = np.linalg.solve(A, -g)
                except np.linalg.LinAlgError:
                    p = np.linalg.lstsq(A, -g, rcond=None)[0]
                
                # Scale to exactly radius if needed
                p_norm = np.linalg.norm(p)
                if p_norm > 0:
                    p = p * (radius / p_norm)
                
                return p, predict_model(coeffs, x_center + p, x_center)
        
        # Main optimization loop
        max_iter = 100
        
        for iter in range(max_iter):
            if time.time() > deadline or calls >= budget:
                break
            
            # Fit quadratic model at current center
            coeffs = fit_quadratic_model(samples_x, samples_f, x_center)
            
            if coeffs is None:
                # Can't fit model, break
                break
            
            # Minimize model within trust region
            p, f_model = minimize_model_in_trust_region(coeffs, x_center, trust_radius)
            
            if p is None:
                break
            
            # Evaluate actual function at proposed point
            x_new = x_center + p
            x_new = np.clip(x_new, -5.0, 5.0)
            
            # Check if we're actually moving
            if np.linalg.norm(p) < 1e-10:
                # Model suggests we're at minimum, but let's try a small step
                # or reduce trust region
                trust_radius *= 0.5
                if trust_radius < min_trust_radius:
                    break
                continue
            
            f_new = safe_func(x_new)
            
            if not np.isfinite(f_new):
                # Evaluation failed, shrink trust region
                trust_radius *= 0.5
                if trust_radius < min_trust_radius:
                    break
                continue
            
            # Compute actual reduction vs predicted reduction
            actual_reduction = f_center - f_new
            predicted_reduction = f_center - f_model
            
            # Compute rho (efficiency ratio)
            if predicted_reduction > 1e-10:
                rho = actual_reduction / predicted_reduction
            else:
                rho = 0.0
            
            # Update trust region based on rho
            if rho < 0.25:
                # Poor fit, shrink trust region
                trust_radius *= 0.5
                if trust_radius < min_trust_radius:
                    break
            elif rho < 0.75:
                # Accept step, keep trust region same
                pass
            else:
                # Good fit, expand trust region
                trust_radius = min(trust_radius * 2.0, max_trust_radius)
            
            # Update center and samples
            if actual_reduction > -1e-10:
                # Accept the step
                x_center = x_new.copy()
                f_center = f_new
                
                # Add new point to samples
                samples_x.append(x_new.copy())
                samples_f.append(f_new)
                
                # Keep only the most recent samples to avoid numerical issues
                max_samples = 3 * dim + 10
                if len(samples_x) > max_samples:
                    # Remove oldest points, but keep the best few
                    # Sort by function value and keep best 2*dim + points near center
                    indices = np.argsort(samples_f)
                    keep_indices = list(indices[:2*dim])
                    # Also keep points close to current center
                    for i, x in enumerate(samples_x):
                        if i not in keep_indices and np.linalg.norm(x - x_center) < trust_radius:
                            keep_indices.append(i)
                            if len(keep_indices) >= max_samples:
                                break
                    keep_indices = sorted(set(keep_indices))
                    samples_x = [samples_x[i] for i in keep_indices]
                    samples_f = [samples_f[i] for i in keep_indices]
            else:
                # Reject step, add to samples but don't move center
                samples_x.append(x_new.copy())
                samples_f.append(f_new)
                
                # Keep samples manageable
                max_samples = 3 * dim + 10
                if len(samples_x) > max_samples:
                    indices = np.argsort(samples_f)
                    keep_indices = list(indices[:2*dim])
                    samples_x = [samples_x[i] for i in keep_indices]
                    samples_f = [samples_f[i] for i in keep_indices]
            
            # Check convergence
            if trust_radius < min_trust_radius:
                break
        
        return best_x, best_f
