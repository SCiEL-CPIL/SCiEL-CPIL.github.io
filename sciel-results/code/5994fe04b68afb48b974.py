"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Trust-Region Quadratic Model Optimizer for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: Evaluator kills at 330s. We aim for 300s hard limit.
        # Leave 30s buffer for safety to ensure we return a valid result.
        t0 = time.time()
        deadline = t0 + 270.0

        # State
        x = np.zeros(dim)
        f = func(x)
        calls = 1
        best_x = x.copy()
        best_f = f
        
        # Initial trust region radius
        # Start small to ensure we are in the local quadratic regime, 
        # but large enough to make progress.
        # Box is [-5, 5], so radius 1.0 is reasonable.
        rho = 1.0 
        # Adaptive factor for trust region
        rho_min = 1e-4
        rho_max = 10.0
        
        # We will estimate gradient and Hessian using a finite difference stencil.
        # For a quadratic function, this estimate is exact (ignoring floating point error).
        # Stencil: Center + 2*dim points (plus/minus along each axis).
        # Total points needed: 1 + 2*dim.
        
        # We will maintain a model: f(x) ~ f0 + g.T * (x-x0) + 0.5 * (x-x0).T * H * (x-x0)
        # We update the model whenever we move the center.
        
        # Helper to build model at current center x
        def build_model(x_center, f_center):
            h = 1e-3 # Step size for finite differences
            # Ensure h is small enough to be within the trust region and box
            # but large enough to avoid numerical noise.
            # Since we are in a trust region, h should be << rho.
            
            # Calculate Gradient
            g = np.zeros(dim)
            H = np.zeros((dim, dim))
            
            # Evaluate points for gradient and diagonal of Hessian
            # f(x+h*e_i), f(x-h*e_i)
            # This gives 2*dim evaluations.
            
            f_plus = np.zeros(dim)
            f_minus = np.zeros(dim)
            
            for i in range(dim):
                if time.time() > deadline:
                    return None, None, calls
                # Step
                step = np.zeros(dim)
                step[i] = h
                x_p = x_center + step
                x_m = x_center - step
                
                # Clip to box to ensure valid calls, though model assumes unconstrained
                # If x_center is near boundary, this might distort the model slightly,
                # but Sphere is smooth.
                f_p = func(np.clip(x_p, -5, 5))
                f_m = func(np.clip(x_m, -5, 5))
                
                calls += 2
                f_plus[i] = f_p
                f_minus[i] = f_m
                
            # Gradient: (f(x+h) - f(x-h)) / 2h
            g = (f_plus - f_minus) / (2 * h)
            
            # Hessian Diagonal: (f(x+h) + f(x-h) - 2f(x)) / h^2
            H_diag = (f_plus + f_minus - 2 * f_center) / (h * h)
            
            # For off-diagonals, we need more points or assume diagonal dominance initially.
            # For Sphere, the Hessian is constant and positive definite.
            # A full finite difference Hessian requires 2*dim^2 + 1 points, which is 51 for dim=5.
            # We have budget 500. 51 is affordable.
            # Let's compute the full Hessian if we have enough budget left.
            
            if calls + 2 * dim * dim < budget and time.time() < deadline:
                # Compute off-diagonals
                # f(x + h(e_i + e_j))
                for i in range(dim):
                    for j in range(i + 1, dim):
                        if time.time() > deadline:
                            break
                        step = np.zeros(dim)
                        step[i] = h
                        step[j] = h
                        x_ij = x_center + step
                        
                        f_ij = func(np.clip(x_ij, -5, 5))
                        calls += 1
                        
                        # Mixed derivative:
                        # f_ij - f_plus[i] - f_plus[j] + f_center  (approx, needs care)
                        # Standard 2nd order central difference for mixed:
                        # f(x+h(e_i+e_j)) - f(x+h(e_i-e_j)) - f(x-h(e_i+e_j)) + f(x-h(e_i-e_j)) / (4h^2)
                        # This requires 4 more points per pair. Too expensive.
                        
                        # Alternative: Use the fact that it's a quadratic.
                        # If we assume the function is strictly quadratic, the Hessian is constant.
                        # We can estimate H_ij using:
                        # f(x + h(e_i+e_j)) - f(x) - h*g_i - h*g_j - 0.5*h^2*(H_ii + H_jj) = h^2 * H_ij
                        # So H_ij = (f_ij - f_center - h*g_i - h*g_j - 0.5*h^2*(H_ii + H_jj)) / h^2
                        
                        val = (f_ij - f_center - h * g[i] - h * g[j] - 0.5 * h * h * (H_diag[i] + H_diag[j])) / (h * h)
                        H[i, j] = val
                        H[j, i] = val
            
            np.fill_diagonal(H, H_diag)
            
            # Ensure H is symmetric and positive definite
            H = (H + H.T) / 2
            # Add small regularization to ensure PD
            H += 1e-8 * np.eye(dim)
            
            return g, H, calls

        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Build model at current x
            g, H, calls = build_model(x, f)
            if g is None:
                break
            
            # Solve for Newton step: H * d = -g
            try:
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent step if singular
                d = -g * 0.1
            
            # Scale step to trust region
            norm_d = np.linalg.norm(d)
            if norm_d > rho:
                d = d * (rho / norm_d)
            
            # Proposed point
            x_new = x + d
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate new point
            f_new = func(x_new)
            calls += 1
            
            # Calculate predicted improvement
            # f(x+d) - f(x) ~ g.T * d + 0.5 * d.T * H * d
            pred_improve = g @ d + 0.5 * d @ H @ d
            actual_improve = f - f_new
            
            # Calculate rho (reduction ratio)
            if abs(pred_improve) < 1e-10:
                # If predicted improvement is tiny, accept if actual is not worse
                if actual_improve >= -1e-8:
                    ratio = 1.0
                else:
                    ratio = 0.0
            else:
                ratio = actual_improve / pred_improve
            
            # Update trust region and state
            if ratio > 0.75:
                # Good step, expand trust region
                rho = min(rho * 2.0, rho_max)
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            elif ratio > 0.25:
                # Acceptable step, keep trust region
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            else:
                # Poor step, shrink trust region
                rho = max(rho * 0.5, rho_min)
            
            # Check convergence
            if np.linalg.norm(d) < 1e-6 and rho < 1e-3:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
