"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
Trust-Region Quadratic Surrogate for anisotropic refinement.
"""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            try:
                wrapped_obj(x0)
            except Exception:
                return None, np.inf
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Trust-Region Quadratic Surrogate ---
        # Build a local quadratic model from recent evaluations
        # The model: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
        # We estimate g and H from finite differences of recent points
        
        current_x = x0.copy()
        current_fx = best_fx
        
        # Trust region radius
        trust_radius = 1.0
        
        # History of recent evaluations for model building
        # Store (x, f) pairs
        history = [(current_x.copy(), current_fx)]
        max_history = 30  # Keep last 30 points for model
        
        # Gradient estimation via central differences
        def estimate_gradient(x_center, f_center):
            """Estimate gradient using finite differences."""
            grad = np.zeros(dim)
            eps = 1e-4
            for i in range(dim):
                x_plus = x_center.copy()
                x_plus[i] += eps
                x_minus = x_center.copy()
                x_minus[i] -= eps
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                try:
                    fp = wrapped_obj(x_plus)
                    fm = wrapped_obj(x_minus)
                    grad[i] = (fp - fm) / (2 * eps)
                except Exception:
                    # If we hit the limit, use one-sided difference
                    try:
                        fp = wrapped_obj(x_plus)
                        grad[i] = (fp - f_center) / eps
                    except Exception:
                        grad[i] = 0.0
            return grad
        
        # Hessian estimation via second-order finite differences
        def estimate_hessian(x_center, f_center, grad):
            """Estimate Hessian using finite differences."""
            H = np.zeros((dim, dim))
            eps = 1e-3
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        # Second derivative
                        x_plus = x_center.copy()
                        x_plus[i] += eps
                        x_minus = x_center.copy()
                        x_minus[i] -= eps
                        x_center2 = x_center.copy()
                        x_plus = np.clip(x_plus, -5, 5)
                        x_minus = np.clip(x_minus, -5, 5)
                        
                        try:
                            fp = wrapped_obj(x_plus)
                            fm = wrapped_obj(x_minus)
                            H[i, i] = (fp - 2 * f_center + fm) / (eps ** 2)
                        except Exception:
                            H[i, i] = 0.0
                    else:
                        # Cross derivative
                        x_pp = x_center.copy()
                        x_pp[i] += eps
                        x_pp[j] += eps
                        x_pm = x_center.copy()
                        x_pm[i] += eps
                        x_pm[j] -= eps
                        x_mp = x_center.copy()
                        x_mp[i] -= eps
                        x_mp[j] += eps
                        x_mm = x_center.copy()
                        x_mm[i] -= eps
                        x_mm[j] -= eps
                        
                        x_pp = np.clip(x_pp, -5, 5)
                        x_pm = np.clip(x_pm, -5, 5)
                        x_mp = np.clip(x_mp, -5, 5)
                        x_mm = np.clip(x_mm, -5, 5)
                        
                        try:
                            fpp = wrapped_obj(x_pp)
                            fpm = wrapped_obj(x_pm)
                            fmp = wrapped_obj(x_mp)
                            fmm = wrapped_obj(x_mm)
                            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)
                            H[j, i] = H[i, j]
                        except Exception:
                            H[i, j] = 0.0
                            H[j, i] = 0.0
            return H
        
        # Main trust-region loop
        no_improve_count = 0
        max_no_improve = 15
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # Estimate gradient and Hessian at current point
            try:
                grad = estimate_gradient(current_x, current_fx)
            except Exception:
                break
            
            # Check if gradient is small enough
            if np.linalg.norm(grad) < 1e-6:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Try to escape by random perturbation
                    if calls_used < budget - 10:
                        perturbation = np.random.randn(dim) * 0.1
                        new_x = np.clip(current_x + perturbation, -5, 5)
                        try:
                            new_fx = wrapped_obj(new_x)
                            if new_fx < current_fx:
                                current_x = new_x
                                current_fx = new_fx
                                no_improve_count = 0
                                trust_radius = 0.5
                            else:
                                break
                        except Exception:
                            break
                    else:
                        break
                continue
            
            # Estimate Hessian (only if we have enough budget)
            if calls_used < budget - 20:
                try:
                    H = estimate_hessian(current_x, current_fx, grad)
                    # Regularize Hessian for stability
                    H += 1e-4 * np.eye(dim)
                except Exception:
                    H = np.eye(dim) * 1e-2
            else:
                H = np.eye(dim) * 1e-2
            
            # Solve for trust-region step: minimize model within trust region
            # min 0.5 d^T H d + g^T d  s.t. ||d|| <= trust_radius
            
            # Use eigenvalue decomposition for stability
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(H)
                
                # Clamp eigenvalues to be positive (for convexity)
                eigenvalues_clamped = np.maximum(eigenvalues, 1e-6)
                
                # Transform to eigenbasis
                g_eig = eigenvectors.T @ grad
                H_eig = np.diag(eigenvalues_clamped)
                
                # Solve in eigenbasis: H_eig d_eig = -g_eig
                d_eig = -g_eig / eigenvalues_clamped
                
                # Check if step exceeds trust region
                step_norm = np.linalg.norm(d_eig)
                if step_norm > trust_radius:
                    d_eig *= trust_radius / step_norm
                
                # Transform back to original basis
                d = eigenvectors @ d_eig
            except Exception:
                # Fallback: steepest descent with trust region
                d = -grad
                step_norm = np.linalg.norm(d)
                if step_norm > trust_radius:
                    d *= trust_radius / step_norm
            
            # Try the proposed step
            new_x = np.clip(current_x + d, -5, 5)
            
            try:
                new_fx = wrapped_obj(new_x)
            except Exception:
                break
            
            # Accept or reject step
            if new_fx < current_fx:
                # Accept: update current point
                current_x = new_x
                current_fx = new_fx
                no_improve_count = 0
                
                # Expand trust region if step was good
                actual_reduction = current_fx - new_fx
                predicted_reduction = 0.5 * d @ H @ d + grad @ d
                if predicted_reduction > 0:
                    rho = actual_reduction / predicted_reduction
                    if rho > 0.75:
                        trust_radius = min(trust_radius * 1.5, 5.0)
                    elif rho > 0.25:
                        trust_radius *= 0.9
                    else:
                        trust_radius *= 0.5
                else:
                    trust_radius *= 0.9
            else:
                # Reject: shrink trust region
                trust_radius *= 0.5
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    if trust_radius < 1e-4:
                        break
                    # Try a random kick to escape
                    if calls_used < budget - 5:
                        perturbation = np.random.randn(dim) * trust_radius
                        new_x2 = np.clip(current_x + perturbation, -5, 5)
                        try:
                            new_fx2 = wrapped_obj(new_x2)
                            if new_fx2 < current_fx:
                                current_x = new_x2
                                current_fx = new_fx2
                                no_improve_count = 0
                                trust_radius = 0.5
                            else:
                                break
                        except Exception:
                            break
                    else:
                        break
        
        # Final update of best if TR found a better point
        if current_fx < best_fx:
            best_fx = current_fx
            best_x = current_x

        return best_x, best_fx
