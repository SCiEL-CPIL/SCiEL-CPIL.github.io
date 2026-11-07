"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic fitting optimizer for Sphere function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        
        d = self.dim
        
        # Phase 1: Exact quadratic fit using finite differences
        # For a quadratic f(x) = (x-c)^T Q (x-c) + f0, we can recover c and Q
        # using 2d+1 points: x0, x0 ± h*e_i for i=1..d
        # This gives us the gradient and Hessian exactly (for a true quadratic).
        
        h = 1.0  # Step size for finite differences
        
        x0 = x.copy()
        f0 = fx
        
        # Evaluate at x0 ± h*e_i
        # We need 2d additional evaluations
        if self.budget < 2 * d + 2:
            return self._fallback_es(func, x, fx, calls, deadline)
        
        # Store gradient and Hessian estimates
        grad = np.zeros(d)
        hess = np.zeros((d, d))
        
        # First pass: evaluate x0 + h*e_i
        f_plus = np.zeros(d)
        for i in range(d):
            x_plus = x0.copy()
            x_plus[i] += h
            x_plus = np.clip(x_plus, -5, 5)
            f_plus[i] = func(x_plus)
            calls += 1
            if time.time() > deadline:
                return x, fx
        
        # Second pass: evaluate x0 - h*e_i
        f_minus = np.zeros(d)
        for i in range(d):
            x_minus = x0.copy()
            x_minus[i] -= h
            x_minus = np.clip(x_minus, -5, 5)
            f_minus[i] = func(x_minus)
            calls += 1
            if time.time() > deadline:
                return x, fx
        
        # Central differences for gradient
        grad = (f_plus - f_minus) / (2 * h)
        
        # Central differences for Hessian diagonal
        hess_diag = (f_plus + f_minus - 2 * f0) / (h * h)
        np.fill_diagonal(hess, hess_diag)
        
        # For off-diagonal Hessian, we need more points or use the fact that
        # for a quadratic, we can determine it from 2d+1 points if we use
        # a different design. However, for Sphere, the Hessian is constant,
        # so we can estimate it more carefully.
        
        # Alternative: Use the 2d+1 point design to fit the full quadratic.
        # f(x0 + u) = u^T A u + g^T u + f0
        # With u = ±h*e_i, we get:
        # f(x0 + h*e_i) = h^2 * A[i,i] + h * g[i] + f0
        # f(x0 - h*e_i) = h^2 * A[i,i] - h * g[i] + f0
        # Adding: 2*h^2*A[i,i] = f_plus[i] + f_minus[i] - 2*f0
        # Subtracting: 2*h*g[i] = f_plus[i] - f_minus[i]
        
        # This only gives diagonal of A and g. For off-diagonal, we need
        # additional points. Let's add d(d-1)/2 points for cross terms.
        
        # Actually, let's use a smarter approach: since it's a Sphere,
        # the Hessian is 2*Q where Q is constant. We can estimate Q from
        # the gradient at two different points.
        
        # Better approach: Use the quadratic model fit with 2d+1 points
        # to get an initial estimate, then refine.
        
        # Let's do a proper quadratic fit with 2d+1 points.
        # We already have f0, f_plus, f_minus.
        # From these:
        # g[i] = (f_plus[i] - f_minus[i]) / (2*h)
        # A[i,i] = (f_plus[i] + f_minus[i] - 2*f0) / (2*h^2)
        
        g_est = (f_plus - f_minus) / (2 * h)
        A_diag = (f_plus + f_minus - 2 * f0) / (2 * h * h)
        
        # For off-diagonal A[i,j], we need f at x0 + h/2*(e_i + e_j)
        # f(x0 + h/2*(e_i+e_j)) = (h/2)^2 * (A[i,i] + 2*A[i,j] + A[j,j]) + (h/2)*(g[i]+g[j]) + f0
        # So A[i,j] = [f_ij - f0 - (h/2)*(g[i]+g[j]) - (h/2)^2*(A[i,i]+A[j,j])] / (2*(h/2)^2)
        
        # Let's evaluate these cross-term points
        A = np.zeros((d, d))
        np.fill_diagonal(A, A_diag)
        
        num_cross = d * (d - 1) // 2
        if self.budget >= 2 * d + 2 + num_cross:
            for i in range(d):
                for j in range(i + 1, d):
                    if time.time() > deadline:
                        break
                    x_cross = x0.copy()
                    x_cross[i] += h / 2
                    x_cross[j] += h / 2
                    x_cross = np.clip(x_cross, -5, 5)
                    f_cross = func(x_cross)
                    calls += 1
                    
                    # f_cross = (h/2)^2 * (A[i,i] + 2*A[i,j] + A[j,j]) + (h/2)*(g[i]+g[j]) + f0
                    # Solve for A[i,j]:
                    # A[i,j] = [f_cross - f0 - (h/2)*(g[i]+g[j]) - (h/2)^2*(A[i,i]+A[j,j])] / (2*(h/2)^2)
                    A_ij = (f_cross - f0 - (h / 2) * (g_est[i] + g_est[j]) - (h / 2) ** 2 * (A_diag[i] + A_diag[j])) / (2 * (h / 2) ** 2)
                    A[i, j] = A_ij
                    A[j, i] = A_ij
        
        # Now find the minimum of the quadratic model:
        # f(x0 + u) ≈ u^T A u + g_est^T u + f0
        # Minimum at u* = -0.5 * A^{-1} * g_est
        try:
            u_star = -0.5 * np.linalg.solve(A, g_est)
        except:
            u_star = -0.5 * np.linalg.pinv(A) @ g_est
        
        x_star = np.clip(x0 + u_star, -5, 5)
        f_star = func(x_star)
        calls += 1
        
        if f_star < fx:
            x, fx = x_star, f_star
        
        # Phase 2: Newton-like polish using gradient and Hessian
        # Since it's a quadratic, one Newton step should converge.
        # But due to numerical errors, let's do a few iterations.
        remaining_calls = self.budget - calls
        for _ in range(min(5, remaining_calls)):
            if time.time() > deadline:
                break
            
            # Estimate gradient and Hessian at current x using central differences
            h_small = max(1e-4, 0.01 * np.linalg.norm(x))
            h_small = min(h_small, 0.1)  # Don't make it too large
            
            grad_new = np.zeros(d)
            hess_new = np.zeros((d, d))
            
            f_current = func(x)
            calls += 1
            if time.time() > deadline:
                break
            
            for i in range(d):
                if time.time() > deadline:
                    break
                
                x_p = x.copy()
                x_p[i] += h_small
                x_p = np.clip(x_p, -5, 5)
                f_p = func(x_p)
                calls += 1
                
                x_m = x.copy()
                x_m[i] -= h_small
                x_m = np.clip(x_m, -5, 5)
                f_m = func(x_m)
                calls += 1
                
                grad_new[i] = (f_p - f_m) / (2 * h_small)
                hess_new[i, i] = (f_p + f_m - 2 * f_current) / (h_small * h_small)
            
            # Estimate off-diagonal Hessian
            for i in range(d):
                for j in range(i + 1, d):
                    if time.time() > deadline:
                        break
                    
                    x_pp = x.copy()
                    x_pp[i] += h_small / 2
                    x_pp[j] += h_small / 2
                    x_pp = np.clip(x_pp, -5, 5)
                    f_pp = func(x_pp)
                    calls += 1
                    
                    # f(x + h/2*e_i + h/2*e_j) ≈ f(x) + h/2*(g[i]+g[j]) + (h/2)^2*(A[i,i]+2*A[i,j]+A[j,j])
                    # A[i,j] = [f_pp - f_current - h/2*(g[i]+g[j]) - (h/2)^2*(A[i,i]+A[j,j])] / (2*(h/2)^2)
                    A_ij = (f_pp - f_current - (h_small / 2) * (grad_new[i] + grad_new[j]) - (h_small / 2) ** 2 * (hess_new[i, i] + hess_new[j, j])) / (2 * (h_small / 2) ** 2)
                    hess_new[i, j] = A_ij
                    hess_new[j, i] = A_ij
            
            # Newton step: x_new = x - H^{-1} * grad
            try:
                step = np.linalg.solve(hess_new, grad_new)
            except:
                step = np.linalg.pinv(hess_new) @ grad_new
            
            # Damping: ensure the step doesn't go too far
            step_norm = np.linalg.norm(step)
            if step_norm > 1.0:
                step = step / step_norm
            
            x_new = np.clip(x + step, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            if f_new < fx:
                x, fx = x_new, f_new
        
        # Phase 3: Final polish with adaptive step size
        remaining_calls = self.budget - calls
        if remaining_calls > 0:
            sigma = 0.01
            for _ in range(remaining_calls):
                if time.time() > deadline:
                    break
                sigma *= 0.9
                if sigma < 1e-12:
                    sigma = 1e-12
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
        
        return x, fx

    def _fallback_es(self, func, x, fx, calls, deadline):
        """Fallback (1+1)-ES if fitting fails or budget is too low."""
        d = self.dim
        rng = np.random.default_rng(1)
        sigma = 1.0
        remaining = self.budget - calls
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
        return x, fx
# EVOLVE-BLOCK-END
