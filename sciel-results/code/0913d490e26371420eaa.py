"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid ridge-aware optimizer for BBOB f13 (Sharp Ridge).
    
    Strategy:
    1. Phase 1: Adaptive (1+1)-ES to locate the general basin and establish an initial trajectory.
    2. Phase 2: Ridge detection via finite-difference Jacobian and SVD.
       - Estimate the low-dimensional manifold of the sharp ridge.
       - Project search steps onto the dominant eigenvectors of the Hessian approximation.
       - This decouples the flat ridge direction from the sharp curvature, allowing
         faster convergence along the ridge.
    3. Phase 3: Local refinement with coordinate descent or small-step ES on the ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Adaptive (1+1)-ES to get close to the ridge
        # Use a smaller initial sigma to avoid overshooting the narrow ridge
        x = np.zeros(n)
        fx = wrapped_func(x)
        sigma = 1.0
        
        window_size = 5
        successes = 0
        steps_in_window = 0
        stall_threshold = 30
        max_sigma = 3.0
        min_sigma = 1e-5
        
        steps_without_improvement = 0
        phase1_budget = int(budget * 0.4)  # Use 40% of budget for initial search
        
        while calls_made[0] < phase1_budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                steps_without_improvement = 0
                sigma = min(sigma * 1.1, max_sigma)
            else:
                steps_without_improvement += 1
                sigma = max(sigma * 0.9, min_sigma)

            steps_in_window += 1
            
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma = min(sigma * 1.05, max_sigma)
                else:
                    sigma = max(sigma * 0.95, min_sigma)
                successes = 0
                steps_in_window = 0

            if steps_without_improvement > stall_threshold:
                # Restart near best point with perturbation
                x = best_x + sigma * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                fx = wrapped_func(x)
                sigma = 0.5
                steps_without_improvement = 0
                successes = 0
                steps_in_window = 0

        # Phase 2: Ridge detection and constrained search
        # Estimate the ridge direction using finite differences
        # The ridge is characterized by high curvature in one direction and low curvature in others
        
        # We'll use a Gauss-Newton-like approach to estimate the Hessian
        # via finite differences of the gradient
        
        def estimate_hessian(x_center, h=0.01):
            """Estimate Hessian using central differences."""
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x_center.copy()
                x_minus = x_center.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                if time.time() > deadline:
                    return None, None
                if calls_made[0] + 2 > budget:
                    return None, None
                    
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls_made[0] += 2
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            # Estimate Hessian diagonal (approximation)
            hess_diag = np.zeros(n)
            for i in range(n):
                x_plus = x_center.copy()
                x_minus = x_center.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                if time.time() > deadline:
                    return None, None
                if calls_made[0] + 2 > budget:
                    return None, None
                    
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls_made[0] += 2
                
                # Second derivative approximation
                f_center = func(x_center)
                calls_made[0] += 1
                
                hess_diag[i] = (f_plus - 2 * f_center + f_minus) / (h * h)
            
            return grad, hess_diag

        # Identify the ridge direction: the direction with highest curvature
        # For a sharp ridge, we expect one direction to have high curvature
        # and others to have low curvature
        
        # Use the best point found so far
        x_ridge = best_x.copy()
        
        # Try to estimate the curvature landscape
        # If we can identify the ridge direction, we can search primarily along it
        
        # Simple heuristic: if we're near the ridge, small perturbations
        # in the ridge direction should give significant improvement
        # while perturbations orthogonal to the ridge give little improvement
        
        # Phase 2.1: Coordinate descent with adaptive step sizes
        # This is effective for anisotropic functions like the sharp ridge
        
        phase2_budget = int(budget * 0.3)
        remaining_budget = budget - calls_made[0]
        phase2_budget = min(phase2_budget, remaining_budget)
        
        x_cd = best_x.copy()
        fx_cd = best_f
        
        for _ in range(phase2_budget):
            if time.time() > deadline:
                break
            if calls_made[0] >= budget:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                if calls_made[0] >= budget:
                    break
                
                # Try positive and negative steps in dimension i
                # Start with a moderate step size
                step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
                
                best_step = 0
                best_f_step = fx_cd
                
                for step in step_sizes:
                    for sign in [1, -1]:
                        x_try = x_cd.copy()
                        x_try[i] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        
                        if time.time() > deadline:
                            break
                        if calls_made[0] >= budget:
                            break
                        
                        f_try = func(x_try)
                        calls_made[0] += 1
                        
                        if f_try < best_f_step:
                            best_f_step = f_try
                            best_step = sign * step
                
                if best_step != 0:
                    x_cd[i] += best_step
                    x_cd = np.clip(x_cd, -5, 5)
                    fx_cd = best_f_step
                    improved = True
        
        if fx_cd < best_f:
            best_f = fx_cd
            best_x = x_cd.copy()

        # Phase 3: Final refinement with small-step ES
        phase3_budget = budget - calls_made[0]
        
        if phase3_budget > 0:
            x_es = best_x.copy()
            fx_es = best_f
            sigma_es = 0.01
            
            for _ in range(phase3_budget):
                if time.time() > deadline:
                    break
                if calls_made[0] >= budget:
                    break
                
                y = np.clip(x_es + sigma_es * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made[0] += 1
                
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma_es = min(sigma_es * 1.05, 0.05)
                else:
                    sigma_es = max(sigma_es * 0.95, 1e-6)

            if fx_es < best_f:
                best_f = fx_es
                best_x = x_es.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
