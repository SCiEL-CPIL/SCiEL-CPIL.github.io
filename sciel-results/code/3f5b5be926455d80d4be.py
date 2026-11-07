"""Derivative-Free Trust-Region optimizer with local quadratic surrogates for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DFTR optimizer: builds local quadratic models from directional probes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 275  # safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- Phase 1: Coarse exploration to find a good starting point ----
        # Use a small set of well-spread points to locate the basin
        n_explore = 40
        # Stratified random points in [-5, 5]^dim
        rng = np.random.RandomState(42)
        explore_pts = np.zeros((n_explore, dim))
        for i in range(n_explore):
            for d in range(dim):
                # Stratified: divide each dim into 5 strata, pick one
                stratum = (i * dim + d) % 5
                explore_pts[i, d] = -5.0 + 2.0 * (stratum + rng.rand())
        
        for pt in explore_pts:
            if total_calls >= budget or time.time() > deadline:
                break
            safe_call(pt)

        if total_calls >= budget or time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: DFTR with local quadratic surrogates ----
        x0 = best_x.copy()
        f0 = best_f
        delta = 2.0  # initial trust-region radius

        # Build an orthonormal basis for the trust region
        # Use coordinate directions plus some random directions
        n_dirs = 2 * dim  # number of directional probes
        # Start with coordinate directions
        dirs = np.eye(dim)
        # Add random directions
        rand_dirs = rng.randn(n_dirs - dim, dim)
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        dirs = np.vstack([dirs, rand_dirs])
        # Orthonormalize using QR
        Q, _ = np.linalg.qr(dirs)
        dirs = Q[:n_dirs]  # take first n_dirs orthonormal directions

        # Trust-region iteration
        max_trust_iters = 30
        for tr_iter in range(max_trust_iters):
            if total_calls >= budget or time.time() > deadline:
                break

            # Build local model using n_dirs + 1 evaluations (center + n_dirs points)
            # We already have f0 at x0; evaluate n_dirs points at x0 + delta * dirs
            model_pts = np.zeros((n_dirs, dim))
            model_fs = np.zeros(n_dirs)
            for i in range(n_dirs):
                if total_calls >= budget or time.time() > deadline:
                    break
                model_pts[i] = x0 + delta * dirs[i]
                model_fs[i] = safe_call(model_pts[i])

            if total_calls >= budget or time.time() > deadline:
                break

            # Fit a quadratic model: f(x) ≈ f0 + g^T (x - x0) + (x - x0)^T H (x - x0)
            # In the trust region, let y = (x - x0) / delta, so y is in unit ball
            # f(x0 + delta*y) ≈ f0 + delta * g^T y + delta^2 * y^T H y
            # We have n_dirs data points: for each i, y_i = dirs[i], f_i = model_fs[i]
            # f_i ≈ f0 + delta * g^T dirs[i] + delta^2 * dirs[i]^T H dirs[i]
            #
            # This is underdetermined (n_dirs = 2*dim, but unknowns are g (dim) + H (dim*dim/2 + dim) = dim + dim*(dim+1)/2)
            # For dim=5: 10 unknowns for g, and 15 for H symmetric -> 25 total unknowns, but only 10 equations.
            # 
            # Instead, use a simpler model: linear + isotropic quadratic
            # f(x) ≈ f0 + g^T (x-x0) + c * ||x-x0||^2
            # Unknowns: g (dim) + c (1) = dim+1 = 6 for dim=5, with 10 data points.
            # This is well-posed.
            
            # Set up least squares: A * params = b
            # For each i: f_i - f0 = delta * (g^T dirs[i]) + delta^2 * c * ||dirs[i]||^2
            # Since dirs are orthonormal, ||dirs[i]||^2 = 1
            # So: (f_i - f0) / delta = g^T dirs[i] + delta * c
            # 
            # Let b_i = (f_i - f0) / delta
            # b_i = g^T dirs[i] + delta * c
            # 
            # We can write this as: [dirs[i]^T, delta] * [g; c] = b_i
            # Stack for all i: A is n_dirs x (dim+1), params = [g; c]
            
            b = (model_fs - f0) / delta
            A = np.zeros((n_dirs, dim + 1))
            A[:, :dim] = dirs
            A[:, dim] = delta
            
            # Solve least squares
            params, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            g = params[:dim]
            c = params[dim]
            
            # The model is: m(y) = f0 + delta * g^T y + delta^2 * c * ||y||^2
            # Minimize m(y) subject to ||y|| <= 1
            # Gradient of m w.r.t. y: delta * g + 2 * delta^2 * c * y
            # If c > 0, the unconstrained minimizer is y* = -g / (2 * delta * c)
            # If c <= 0, the model is concave, minimize on the boundary
            
            if c > 1e-10:
                y_star = -g / (2.0 * delta * c)
                norm_y = np.linalg.norm(y_star)
                if norm_y > 1.0:
                    y_star = y_star / norm_y  # project to boundary
                step = delta * y_star
            else:
                # Concave or flat: move in direction of -g to the boundary
                norm_g = np.linalg.norm(g)
                if norm_g < 1e-10:
                    # Flat model, try a small random step
                    y_star = rng.randn(dim)
                    y_star /= np.linalg.norm(y_star)
                    step = delta * y_star
                else:
                    step = -delta * g / norm_g
            
            # Try the step
            x_new = x0 + step
            f_new = safe_call(x_new)
            
            if f_new < f0:
                # Accept: update x0, f0, and possibly expand trust region
                x0 = x_new
                f0 = f_new
                # Expand trust region if the step was successful and went to boundary
                if np.linalg.norm(step) > 0.9 * delta:
                    delta = min(delta * 1.5, 5.0)
                else:
                    delta = min(delta * 1.2, 5.0)
            else:
                # Reject: shrink trust region
                delta *= 0.5
                if delta < 1e-4:
                    break
                # Try a different direction: random perturbation
                y_rand = rng.randn(dim)
                y_rand /= np.linalg.norm(y_rand)
                x_try = x0 + delta * y_rand
                f_try = safe_call(x_try)
                if f_try < f0:
                    x0 = x_try
                    f0 = f_try
                    delta = min(delta * 1.5, 5.0)
                else:
                    if delta < 1e-5:
                        break

            # Check if we're close to the boundary of the box
            if np.any(x0 < -4.9) or np.any(x0 > 4.9):
                delta = min(delta, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
