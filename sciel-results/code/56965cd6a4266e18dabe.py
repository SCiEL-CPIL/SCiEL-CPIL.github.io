"""Ridge-aware optimizer for BBOB f13 (Sharp ridge) with manifold projection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Ridge-aware optimizer with manifold projection for sharp ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with adaptive (1+1)-ES
        x = np.zeros(dim)
        f = func(x)
        best_x = x.copy()
        best_f = f
        calls_used = 1
        sigma = 2.5
        stall_count = 0
        max_stall = 30

        # Phase 1: Adaptive (1+1)-ES until budget exhausted or convergence
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            z = rng.standard_normal(dim)
            x_new = np.clip(x + sigma * z, -5, 5)
            f_new = func(x_new)
            calls_used += 1

            if f_new < f:
                f = f_new
                x = x_new
                best_x = x.copy()
                best_f = f
                sigma *= 1.2  # Success: increase step
                stall_count = 0
            else:
                sigma *= 0.85  # Failure: decrease step
                stall_count += 1

            # If stalled for too long, try larger jump
            if stall_count >= max_stall:
                z_large = rng.standard_normal(dim) * 3
                x_jump = np.clip(best_x + z_large, -5, 5)
                f_jump = func(x_jump)
                calls_used += 1
                if f_jump < best_f:
                    best_f = f_jump
                    best_x = x_jump.copy()
                    x = x_jump.copy()
                    f = f_jump
                    sigma = 1.5
                stall_count = 0

            # Early termination if very converged
            if sigma < 1e-6:
                break

        # Phase 2: Ridge-aware exploitation using gradient estimation
        # Estimate gradient at best point via central differences
        def estimate_gradient(x0, h=1e-4):
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x0.copy()
                x_minus = x0.copy()
                x_plus[i] = min(5.0, x0[i] + h)
                x_minus[i] = max(-5.0, x0[i] - h)
                grad[i] = (func(x_plus) - func(x_minus)) / (x_plus[i] - x_minus[i])
                calls_used += 2
            return grad

        # Estimate Hessian diagonal via second differences
        def estimate_hessian_diag(x0, h=1e-3):
            hess_diag = np.zeros(dim)
            for i in range(dim):
                x_plus = x0.copy()
                x_minus = x0.copy()
                x0_copy = x0.copy()
                x_plus[i] = min(5.0, x0[i] + h)
                x_minus[i] = max(-5.0, x0[i] - h)
                hess_diag[i] = (func(x_plus) - 2 * func(x0_copy) + func(x_minus)) / (h * h)
                calls_used += 3
            return hess_diag

        # Ridge direction: direction of smallest curvature (null space)
        # Use power iteration on inverse Hessian approximation
        if calls_used < budget - 20:
            grad = estimate_gradient(best_x)
            if np.linalg.norm(grad) > 1e-8:
                # Normalize gradient
                grad_norm = np.linalg.norm(grad)
                grad = grad / grad_norm

                # Estimate Hessian diagonal to find ridge direction
                hess_diag = estimate_hessian_diag(best_x)
                # Ridge direction is orthogonal to gradient, in direction of low curvature
                # Simple approach: use gradient direction for descent, 
                # and random orthogonal directions for ridge exploration
                
                # Phase 2a: Steepest descent with backtracking
                x_cur = best_x.copy()
                f_cur = best_f
                step_size = 0.5
                
                for _ in range(min(50, budget - calls_used)):
                    if time.time() > deadline:
                        break
                    if calls_used >= budget:
                        break
                    
                    # Re-estimate gradient
                    grad = estimate_gradient(x_cur)
                    if np.linalg.norm(grad) < 1e-8:
                        break
                    
                    # Backtracking line search
                    alpha = step_size
                    for _ in range(20):
                        x_trial = np.clip(x_cur - alpha * grad, -5, 5)
                        f_trial = func(x_trial)
                        calls_used += 1
                        if f_trial < f_cur - 1e-4 * alpha * np.linalg.norm(grad) ** 2:
                            break
                        alpha *= 0.5
                    
                    if f_trial < f_cur:
                        f_cur = f_trial
                        x_cur = x_trial
                        step_size = min(1.0, step_size * 1.1)
                    else:
                        step_size *= 0.7
                    
                    if f_cur < best_f:
                        best_f = f_cur
                        best_x = x_cur.copy()
                    
                    if step_size < 1e-6:
                        break

                # Phase 2b: Ridge-aligned random search in null space
                # Find directions orthogonal to gradient
                if calls_used < budget - 10:
                    grad_final = estimate_gradient(best_x)
                    if np.linalg.norm(grad_final) > 1e-8:
                        grad_final = grad_final / np.linalg.norm(grad_final)
                        
                        # Generate random vectors and project to null space
                        ridge_sigma = 0.3
                        for _ in range(min(30, budget - calls_used)):
                            if time.time() > deadline:
                                break
                            if calls_used >= budget:
                                break
                            
                            z = rng.standard_normal(dim)
                            # Project onto null space of gradient
                            z_proj = z - np.dot(z, grad_final) * grad_final
                            z_norm = np.linalg.norm(z_proj)
                            if z_norm < 1e-8:
                                continue
                            z_proj /= z_norm
                            
                            x_trial = np.clip(best_x + ridge_sigma * z_proj, -5, 5)
                            f_trial = func(x_trial)
                            calls_used += 1
                            
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()
                                ridge_sigma *= 1.1
                            else:
                                ridge_sigma *= 0.8

        # Phase 3: Final Nelder-Mead polish
        if calls_used < budget - 5:
            # Initialize simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                pt = best_x.copy()
                pt[i] += 0.1
                simplex.append(np.clip(pt, -5, 5))
            
            # Evaluate initial simplex
            simplex_f = []
            for i, pt in enumerate(simplex):
                if i == 0:
                    simplex_f.append(best_f)
                else:
                    f_pt = func(pt)
                    calls_used += 1
                    simplex_f.append(f_pt)
            
            # Nelder-Mead iterations
            nm_max_iter = min(50, (budget - calls_used) // (dim + 1))
            for _ in range(nm_max_iter):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = np.clip(2 * centroid - simplex[-1], -5, 5)
                f_r = func(x_r)
                calls_used += 1
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = np.clip(centroid + 2 * (x_r - centroid), -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            f_i = func(simplex[i])
                            calls_used += 1
                            simplex_f[i] = f_i
                
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
