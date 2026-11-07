import numpy as np
import time
import sys

class Optimizer:
    """
    Optimizer using a Quadratic Surrogate Model for the Bent Cigar function.
    Fits a local quadratic model to recent evaluations to estimate curvature
    and take efficient steps towards the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 290.0  # Safety margin: 300s limit, stop at 290s
        self.best_x = None
        self.best_f = np.inf
        self.calls = 0

    def _fit_quadratic(self, X, Y):
        """
        Fits a quadratic model: f(x) = c + g^T x + 0.5 x^T H x
        Returns: c, g, H
        """
        n, d = X.shape
        # Number of parameters: 1 (const) + d (linear) + d*(d+1)/2 (quadratic)
        # We construct the design matrix A such that A * params = Y
        # params = [c, g_1...g_d, H_11, H_12, ..., H_dd]
        
        # Quadratic terms: x_i * x_j for i <= j
        quad_cols = []
        for i in range(d):
            for j in range(i, d):
                quad_cols.append(X[:, i] * X[:, j])
        
        if quad_cols:
            A_quad = np.column_stack(quad_cols)
        else:
            A_quad = np.zeros((n, 0))
            
        A_lin = X
        A_const = np.ones((n, 1))
        
        A = np.hstack([A_const, A_lin, A_quad])
        
        # Solve least squares
        try:
            params, residuals, rank, sv = np.linalg.lstsq(A, Y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback if singular (should not happen with full rank design usually)
            params = np.zeros(A.shape[1])
            
        c = params[0]
        g = params[1:1+d]
        
        # Reconstruct Hessian H from quadratic coefficients
        # The term in the model is sum_{i<=j} coeff_ij * x_i * x_j
        # This corresponds to 0.5 * x^T H x where H is symmetric.
        # If i==j, coeff_ii corresponds to H_ii / 2? No.
        # Let's be precise.
        # Model: c + sum g_i x_i + sum_{i<=j} q_{ij} x_i x_j
        # Gradient: g_i + sum_j H_ij x_j = 0
        # Hessian: H
        # For i != j: term q_ij x_i x_j contributes 2 q_ij to H_ij and H_ji?
        # Actually, standard quadratic form 0.5 x^T H x = 0.5 sum_i H_ii x_i^2 + sum_{i<j} H_ij x_i x_j
        # So if we have coefficients q_ij for i<=j:
        # If i==j, H_ii = 2 * q_ii
        # If i<j, H_ij = H_ji = 2 * q_ij (since the term in 0.5 x^T H x is H_ij x_i x_j, 
        # but our sum has q_ij x_i x_j. Wait.
        # 0.5 * (H_ij x_i x_j + H_ji x_j x_i) = H_ij x_i x_j (since H is symmetric).
        # So the coefficient of x_i x_j in the expansion is H_ij.
        # Thus q_ij = H_ij for i<j.
        # And for i=j, term is 0.5 H_ii x_i^2. So q_ii = 0.5 H_ii => H_ii = 2 q_ii.
        
        H = np.zeros((d, d))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                q_val = params[1 + d + idx]
                if i == j:
                    H[i, j] = 2.0 * q_val
                else:
                    H[i, j] = q_val
                    H[j, i] = q_val
                idx += 1
                
        return c, g, H

    def _get_quadratic_minimizer(self, c, g, H):
        """
        Finds x minimizing c + g^T x + 0.5 x^T H x.
        Gradient: g + H x = 0 => H x = -g
        """
        # Regularize H to ensure it's positive definite for minimization
        # If H is not PD, the quadratic is unbounded or saddle. 
        # We add a small ridge to H to make it invertible and PD.
        H_reg = H + 1e-6 * np.eye(self.dim)
        
        try:
            # Check if PD
            # For small dim, eigenvalues are cheap
            eigvals = np.linalg.eigvalsh(H_reg)
            if np.min(eigvals) <= 0:
                # Not PD, increase ridge
                H_reg = H + 1e-3 * np.eye(self.dim)
                eigvals = np.linalg.eigvalsh(H_reg)
                if np.min(eigvals) <= 0:
                    # Still not PD, use identity metric (steepest descent direction)
                    # x = -g / lambda_max? Or just -g scaled.
                    # Let's just return the steepest descent direction with a fixed step
                    norm_g = np.linalg.norm(g)
                    if norm_g < 1e-8:
                        return None
                    step = -g / norm_g * 1.0
                    return step
            
            x_min = np.linalg.solve(H_reg, -g)
            return x_min
        except np.linalg.LinAlgError:
            return None

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(42)
        
        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        self.calls += 1
        self.best_x = x.copy()
        self.best_f = fx
        
        history_X = [x]
        history_Y = [fx]
        
        # Adaptive step size for fallback
        sigma = 1.0
        
        # Phase 1: Gather initial points for quadratic fit
        # Need at least dim+1 points for linear, but for quadratic we want more.
        # Let's start with a few random points to establish a local model.
        # We will use a sliding window of the last 2*dim + 1 points for the fit.
        window_size = 2 * dim + 1
        
        for i in range(self.budget - 1):
            if time.time() > self.deadline:
                break
            
            # Check if we have enough points to fit a quadratic
            # A full quadratic in d dims has d(d+1)/2 + d + 1 parameters.
            # For d=5, that's 15+5+1 = 21 parameters.
            # We should use at least 21 points.
            # Let's accumulate until we have 21 points, then switch to surrogate-guided search.
            
            if len(history_X) < 2 * dim + 1:
                # Exploration phase: random steps
                step = rng.standard_normal(dim) * sigma
                x_new = np.clip(x + step, -5, 5)
                fx_new = func(x_new)
                self.calls += 1
                
                if fx_new < self.best_f:
                    self.best_f = fx_new
                    self.best_x = x_new.copy()
                    x = x_new
                    fx = fx_new
                    # Reduce sigma on success? Or keep? 
                    # In exploration, maybe reduce slightly to converge locally
                    sigma *= 0.9
                else:
                    # Stay at current x? Or move?
                    # For (1+1) style, we stay. For model building, we might want to move anyway?
                    # If we stay, we don't add new diversity. 
                    # Let's move to x_new to gather data, but keep best_x separate.
                    # However, if we move to a worse point, the next step is from there.
                    # This is a "random walk" exploration.
                    x = x_new
                    fx = fx_new
                    sigma *= 1.1 # Expand if failed?
                
                history_X.append(x_new)
                history_Y.append(fx_new)
                
                # Trim history if it gets too long (keep most recent)
                if len(history_X) > 2 * dim + 1:
                    history_X.pop(0)
                    history_Y.pop(0)
                    
            else:
                # Exploitation phase: Use quadratic surrogate
                X_hist = np.array(history_X)
                Y_hist = np.array(history_Y)
                
                c, g, H = self._fit_quadratic(X_hist, Y_hist)
                
                # Get suggested step from quadratic minimizer
                # The minimizer of the surrogate is x_surr = x_current - H^{-1} g
                # Wait, the model is centered at 0 in the fit? 
                # No, the fit is in the absolute coordinates.
                # So x_surr is the absolute coordinate of the minimizer.
                # The step is x_surr - x_current.
                
                step = self._get_quadratic_minimizer(c, g, H)
                
                if step is None:
                    # Fallback to random
                    step = rng.standard_normal(dim) * sigma
                else:
                    # Limit step size to avoid jumping too far outside the local model validity
                    # If the step is huge, the quadratic approximation is likely bad.
                    step_norm = np.linalg.norm(step)
                    max_step = 2.0 # Cap the step
                    if step_norm > max_step:
                        step = step / step_norm * max_step
                        
                    # Also, if the step is very small, add some noise to escape flat regions
                    if step_norm < 0.1:
                        step += rng.standard_normal(dim) * 0.1 * sigma

                x_new = np.clip(x + step, -5, 5)
                fx_new = func(x_new)
                self.calls += 1
                
                if fx_new < self.best_f:
                    self.best_f = fx_new
                    self.best_x = x_new.copy()
                    x = x_new
                    fx = fx_new
                    # Success: shrink sigma
                    sigma *= 0.8
                else:
                    # Failure: 
                    # Move to new point to update model? 
                    # If we don't move, we repeat the same point? 
                    # We should move to x_new to gather data, even if worse.
                    x = x_new
                    fx = fx_new
                    # Increase sigma
                    sigma *= 1.2
                    # Cap sigma
                    sigma = min(sigma, 5.0)

                history_X.append(x_new)
                history_Y.append(fx_new)
                
                # Trim history
                if len(history_X) > 2 * dim + 1:
                    history_X.pop(0)
                    history_Y.pop(0)

        return self.best_x, self.best_f
