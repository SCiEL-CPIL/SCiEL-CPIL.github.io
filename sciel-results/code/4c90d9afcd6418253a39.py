"""Derivative-free trust-region BFGS optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """DF-TBFGS optimizer with finite-difference gradients and quadratic models."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 295  # Safety margin: 330s kill, 35s buffer
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            x = np.asarray(x).flatten()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial sampling to find a good starting point
        init_points = [np.zeros(d)]
        for _ in range(12):
            init_points.append(rng.uniform(-5, 5, d))

        for p in init_points:
            fx = try_eval(p)
            if fx is None:
                break

        if best_x is None:
            best_x = np.zeros(d)
            best_f = try_eval(best_x)

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: DF-TBFGS (Derivative-Free Trust-Region BFGS)
        x = best_x.copy()
        fx = best_f

        # Finite-difference gradient estimation
        def estimate_gradient(x, f0, h):
            """Estimate gradient using central differences."""
            nonlocal calls_used, best_f, best_x
            grad = np.zeros(d)
            for i in range(d):
                if calls_used >= max_calls or time.time() > deadline - 2:
                    return None
                x_plus = x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5, 5)
                fp = try_eval(x_plus)
                if fp is None:
                    return None
                x_minus = x.copy()
                x_minus[i] -= h
                x_minus = np.clip(x_minus, -5, 5)
                fm = try_eval(x_minus)
                if fm is None:
                    return None
                grad[i] = (fp - fm) / (2 * h)
            return grad

        # Solve trust-region subproblem: min 0.5*z'*H*z + g'*z s.t. ||z|| <= delta
        def solve_trust_region(g, H, delta):
            """Solve the trust-region subproblem using eigen-decomposition."""
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            # Ensure H is positive semi-definite by clipping negative eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Try unconstrained solution first
            try:
                z_unconstrained = np.linalg.solve(H, -g)
                if np.linalg.norm(z_unconstrained) <= delta:
                    return z_unconstrained
            except np.linalg.LinAlgError:
                pass
            
            # Constrained solution: z = delta * eigenvector corresponding to smallest eigenvalue
            # Actually, for the trust region problem, the solution lies on the boundary
            # We use the formula: z = (delta / (sqrt(lambda_k + rho))) * v_k
            # where lambda_k is the smallest eigenvalue and rho is chosen so that
            # sum_i (v_i^T g)^2 / (lambda_i + rho) = delta^2
            
            # Simple approach: project the unconstrained solution onto the sphere
            norm_z = np.linalg.norm(z_unconstrained)
            if norm_z > delta:
                z_constrained = delta * z_unconstrained / norm_z
                return z_constrained
            
            return z_unconstrained

        # BFGS update
        def bfgs_update(H, s, y):
            """Standard BFGS update."""
            sy = s @ y
            if sy <= 1e-10:
                return H
            H_new = H + np.outer(s, s) / sy - np.outer(H @ s, s @ H) / (s @ H @ s + 1e-10)
            # Symmetrize
            H_new = 0.5 * (H_new + H_new.T)
            return H_new

        # Main DF-TBFGS loop
        delta = 1.0  # Initial trust region radius
        H = np.eye(d)  # Initial Hessian approximation
        stagnation_count = 0
        max_stagnation = 30  # Max iterations without improvement before restarting

        for iteration in range(500):
            if calls_used >= max_calls or time.time() > deadline - 5:
                break

            # Estimate gradient
            h = max(delta * 1e-4, 1e-5)
            g = estimate_gradient(x, fx, h)
            if g is None:
                break

            # If gradient is very small, we're near a minimum
            if np.linalg.norm(g) < 1e-10:
                break

            # Solve trust-region subproblem
            z = solve_trust_region(g, H, delta)
            
            # Step size: take full step within trust region
            x_new = x + z
            x_new = np.clip(x_new, -5, 5)

            # Evaluate new point
            fx_new = try_eval(x_new)
            if fx_new is None:
                break

            # Compute actual reduction
            actual_reduction = fx - fx_new
            predicted_reduction = -0.5 * z @ H @ z - g @ z

            # Update trust region
            if actual_reduction > 0.1 * predicted_reduction and predicted_reduction > 0:
                # Good step: increase trust region
                delta = min(2 * delta, 5.0)
                x = x_new
                fx = fx_new
                stagnation_count = 0
                
                # BFGS update
                s = x - (x - z)  # = z
                y = estimate_gradient(x, fx, h)
                if y is not None:
                    H = bfgs_update(H, z, y - g)
            elif actual_reduction > 0:
                # Acceptable step: keep trust region
                delta = delta
                x = x_new
                fx = fx_new
                stagnation_count = 0
            else:
                # Rejected step: decrease trust region
                delta = 0.5 * delta
                stagnation_count += 1

            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check for stagnation
            if stagnation_count > max_stagnation:
                # Restart with a new random point nearby
                x = best_x + rng.normal(0, 0.5, d)
                x = np.clip(x, -5, 5)
                fx = try_eval(x)
                if fx is None:
                    break
                delta = 0.5
                H = np.eye(d)
                stagnation_count = 0

            # Convergence check
            if delta < 1e-8:
                break

        # Phase 3: Final local refinement with coordinate descent
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            
            for coord_iter in range(20):
                if calls_used >= max_calls or time.time() > deadline - 1:
                    break
                
                improved = False
                for i in range(d):
                    if calls_used >= max_calls or time.time() > deadline - 1:
                        break
                    
                    # Try to improve along dimension i
                    step = max(delta * 0.1, 1e-3)
                    best_local_f = fx
                    best_local_x = x.copy()
                    
                    for t in np.linspace(-1, 1, 11):
                        if calls_used >= max_calls or time.time() > deadline - 1:
                            break
                        x_trial = x.copy()
                        x_trial[i] += t * step
                        x_trial = np.clip(x_trial, -5, 5)
                        ft = try_eval(x_trial)
                        if ft is None:
                            break
                        if ft < best_local_f:
                            best_local_f = ft
                            best_local_x = x_trial.copy()
                    
                    if best_local_f < fx - 1e-10:
                        x = best_local_x
                        fx = best_local_f
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
