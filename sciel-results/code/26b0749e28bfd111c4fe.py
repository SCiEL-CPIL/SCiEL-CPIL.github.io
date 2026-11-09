"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Trust-Region Derivative-Free Optimizer (Bobyqa-inspired) for f11 Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _build_quadratic_model(self, points, values):
        """
        Build a quadratic model from scattered points.
        points: (n, d) array
        values: (n,) array
        Returns: (b, A, c) such that model(x) = b + c^T x + 0.5 * x^T A x
        We use a least-squares fit with a ridge penalty for stability.
        """
        n = len(points)
        d = self.dim
        
        # Basis functions: [1, x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2]
        # Number of quadratic terms: d*(d+1)/2
        n_quad = d * (d + 1) // 2
        n_params = 1 + d + n_quad
        
        # Build design matrix
        M = np.zeros((n, n_params))
        M[:, 0] = 1.0
        M[:, 1:1+d] = points
        
        # Quadratic terms
        quad_idx = 1 + d
        for i in range(d):
            for j in range(i, d):
                M[:, quad_idx] = points[:, i] * points[:, j]
                quad_idx += 1
        
        # Solve least squares with ridge regularization
        # Add small ridge to handle ill-conditioning
        ridge = 1e-6 * np.eye(n_params)
        try:
            # Use np.linalg.lstsq with ridge
            A_aug = M.T @ M + ridge
            b_aug = M.T @ values
            coeffs = np.linalg.solve(A_aug, b_aug)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(M, values, rcond=None)[0]
        
        # Extract coefficients
        b = coeffs[0]
        c = coeffs[1:1+d]
        
        # Reconstruct Hessian A from quadratic coefficients
        # The quadratic part is sum_{i<=j} coeffs_quad[i,j] * x_i * x_j
        # This corresponds to 0.5 * x^T H x where H is symmetric
        H = np.zeros((d, d))
        quad_idx = 0
        for i in range(d):
            for j in range(i, d):
                coeff = coeffs[1 + d + quad_idx]
                if i == j:
                    H[i, i] = coeff
                else:
                    H[i, j] = coeff / 2.0
                    H[j, i] = coeff / 2.0
                quad_idx += 1
        
        return b, c, H

    def _find_trust_region_step(self, b, c, H, center, radius):
        """
        Find the step that minimizes the quadratic model within the trust region.
        model(x) = b + c^T x + 0.5 * x^T H x
        We want to minimize over ||x - center|| <= radius.
        Let y = x - center. Then x = center + y.
        model(center + y) = b + c^T (center + y) + 0.5 * (center + y)^T H (center + y)
        = [b + c^T center + 0.5 * center^T H center] 
          + [c^T + center^T H] y 
          + 0.5 * y^T H y
        
        So we minimize: g^T y + 0.5 * y^T H y where g = c + H @ center
        subject to ||y|| <= radius.
        
        This is a trust-region subproblem. We solve it using a simple approach:
        1. Try the unconstrained minimum: y* = -H^{-1} g if H is PD
        2. If ||y*|| <= radius, use y*
        3. Otherwise, use a Cauchy point or boundary point
        """
        g = c + H @ center
        
        # Check if H is positive definite (approximately)
        try:
            eigvals = np.linalg.eigvalsh(H)
            if np.min(eigvals) > 1e-8:
                # PD: unconstrained minimum exists
                try:
                    y_star = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    y_star = -radius * g / (np.linalg.norm(g) + 1e-10)
                
                norm_y = np.linalg.norm(y_star)
                if norm_y <= radius:
                    return y_star
                else:
                    # Scale to boundary
                    return radius * y_star / norm_y
            else:
                # Not PD: use Cauchy point (direction of steepest descent)
                norm_g = np.linalg.norm(g)
                if norm_g < 1e-10:
                    return np.zeros(self.dim)
                
                # Cauchy point: y = -alpha * g where alpha = min(radius / norm_g, ...)
                # Simple: move along -g by radius
                return -radius * g / norm_g
        except Exception:
            # Fallback: steepest descent
            norm_g = np.linalg.norm(g)
            if norm_g < 1e-10:
                return np.zeros(self.dim)
            return -radius * g / norm_g

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Initial random sampling to get diverse points
        n_init = min(50, budget // 4)
        points = []
        values = []
        
        for i in range(n_init):
            if time.time() > deadline or calls_made >= budget:
                break
            if i == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-5, 5, d)
            f = evaluate(x)
            points.append(x.copy())
            values.append(f)
        
        if len(points) < d + 2:
            # Not enough points, do more random search
            while calls_made < budget and time.time() < deadline:
                x = rng.uniform(-5, 5, d)
                f = evaluate(x)
                points.append(x.copy())
                values.append(f)
            return best_x, best_f
        
        # Phase 2: Trust-region iterations
        # Start with the best point as center
        idx_best = np.argmin(values)
        center = points[idx_best].copy()
        f_center = values[idx_best]
        
        # Initial trust region radius
        radius = 1.0
        
        # Keep a set of points for model building (limit to ~2*d + 5 points)
        max_points = 2 * d + 10
        
        while calls_made < budget and time.time() < deadline:
            # Select points for model building: closest to center + best few
            # Sort by distance to center
            dists = np.array([np.linalg.norm(p - center) for p in points])
            order = np.argsort(dists)
            
            # Take closest max_points points
            selected_indices = order[:max_points]
            sel_points = np.array([points[i] for i in selected_indices])
            sel_values = np.array([values[i] for i in selected_indices])
            
            # Build quadratic model
            b, c, H = self._build_quadratic_model(sel_points, sel_values)
            
            # Find trust region step
            y = self._find_trust_region_step(b, c, H, center, radius)
            x_new = center + y
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            f_new = evaluate(x_new)
            points.append(x_new.copy())
            values.append(f_new)
            
            # Check improvement
            rho_actual = (f_center - f_new) / max(radius, 1e-8)
            
            if rho_actual > 0.25:
                # Good step: accept and increase radius
                center = x_new.copy()
                f_center = f_new
                radius = min(2.0 * radius, 5.0)  # Cap radius at 5
            elif rho_actual > 0.1:
                # Accept but don't increase
                center = x_new.copy()
                f_center = f_new
            else:
                # Reject: decrease radius
                radius = 0.5 * radius
                if radius < 0.01:
                    # Trust region too small, try a random restart
                    if calls_made + 10 > budget:
                        break
                    # Random perturbation
                    x_rand = center + rng.uniform(-0.5, 0.5, d)
                    x_rand = np.clip(x_rand, -5, 5)
                    f_rand = evaluate(x_rand)
                    points.append(x_rand.copy())
                    values.append(f_rand)
                    if f_rand < f_center:
                        center = x_rand.copy()
                        f_center = f_rand
                    radius = 0.5
            
            # Periodically add random points to maintain diversity
            if len(points) % 10 == 0 and calls_made < budget - 20:
                x_rand = rng.uniform(-5, 5, d)
                f_rand = evaluate(x_rand)
                points.append(x_rand.copy())
                values.append(f_rand)
            
            # Limit the number of stored points to avoid memory issues
            if len(points) > 200:
                # Keep best 100 points
                order_vals = np.argsort(values)
                keep = order_vals[:100]
                points = [points[i] for i in keep]
                values = [values[i] for i in keep]
        
        return best_x, best_f

# EVOLVE-BLOCK-END
