"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Trust-region Broyden optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        
        # Helper to safely call func and update best
        def eval_point(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse Global Search (Random Sampling)
        # Use a fraction of the budget to find a good starting point for local refinement.
        # f17 is multimodal, so starting from a random point is crucial.
        coarse_budget = min(50, budget // 10)
        for _ in range(coarse_budget):
            if time.time() > deadline - 10.0:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)
            
        if calls >= budget:
            return best_x, best_f

        # Phase 2: Trust-Region Broyden (Stochastic Quasi-Newton)
        # We start from the best point found in Phase 1.
        x = best_x.copy()
        f = best_f
        
        # Initialize Broyden matrix (Hessian approximation) as Identity
        H = np.eye(dim)
        
        # Trust region radius
        delta = 2.0
        min_delta = 1e-6
        max_delta = 10.0
        
        # Parameters
        rho = 1e-8
        eta = 0.25
        gamma = 0.5
        gamma_expand = 2.0
        
        # BFGS/Broyden update parameters
        # We use a simplified Broyden update (BFGS-like) for stability
        # B_{k+1} = B_k + (s_k s_k^T) / (s_k^T s_k) - (B_k y_k y_k^T B_k) / (y_k^T B_k y_k)
        # But since we don't have gradients, we use a finite difference approximation for the "gradient"
        # or use the "Broyden's method" which updates the inverse Hessian H directly.
        # H_{k+1} = H_k + (s_k - H_k y_k)(s_k - H_k y_k)^T / (y_k^T (s_k - H_k y_k))
        # Here y_k is the gradient difference. Since we don't have gradients, we can approximate them
        # or use a different update rule.
        
        # Actually, for derivative-free, a common approach is to approximate the gradient
        # using finite differences within the trust region, or use a model-based approach.
        # Let's implement a simple finite-difference gradient estimation within the trust region
        # to drive a Newton-like step, but constrained by the trust region.
        
        # However, estimating gradients with finite differences costs O(dim) evaluations per step.
        # With budget 500 and dim 5, we can afford ~100 gradient estimates.
        # Let's use a more efficient approach: 
        # 1. Estimate gradient at current x using central differences (cost 2*dim)
        # 2. Compute Newton step: s = -H * grad
        # 3. If ||s|| > delta, scale s to delta.
        # 4. Evaluate x + s.
        # 5. If improvement is sufficient, update H using Broyden's formula with estimated gradients.
        
        # Let's refine this:
        # At each iteration:
        # 1. Estimate gradient g at x. (2*dim calls)
        # 2. Compute step s = -H @ g.
        # 3. Scale s to have norm <= delta.
        # 4. Evaluate f(x + s).
        # 5. If f(x+s) < f(x) - eta * ||g|| * ||s|| (sufficient decrease):
        #      Accept step.
        #      Estimate gradient g_new at x+s. (2*dim calls)
        #      Update H using Broyden: H_new = H + (s - H*(g_new-g))(s - H*(g_new-g))^T / ((g_new-g)^T (s - H*(g_new-g)))
        #      Expand delta.
        #    Else:
        #      Reject step.
        #      Shrink delta.
        
        # Total cost per successful step: 1 (new point) + 2*dim (grad new) + 2*dim (grad old, if not cached) 
        # We can cache the gradient from the previous step.
        
        g = None # Cached gradient
        x_old = x.copy()
        f_old = f
        
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            # 1. Estimate gradient at x
            h = max(1e-5, delta * 1e-3)
            g = np.zeros(dim)
            for i in range(dim):
                if calls >= budget: break
                xp = x.copy()
                xp[i] += h
                fp = eval_point(xp)
                
                if calls >= budget: break
                xm = x.copy()
                xm[i] -= h
                fm = eval_point(xm)
                
                g[i] = (fp - fm) / (2 * h)
            
            if calls >= budget:
                break
                
            # 2. Compute Newton step
            s = -H @ g
            s_norm = np.linalg.norm(s)
            if s_norm == 0:
                break
            
            # 3. Scale step to trust region
            if s_norm > delta:
                s = s * (delta / s_norm)
                s_norm = delta
                
            x_new = x + s
            x_new = np.clip(x_new, -5.0, 5.0)
            
            # 4. Evaluate new point
            f_new = eval_point(x_new)
            
            # 5. Check for sufficient decrease
            # Predicted decrease: 0.5 * s^T H^{-1} s ? No, Newton step s = -H g.
            # Predicted decrease approx: -0.5 * g^T s = 0.5 * ||g||^2 ||H||? 
            # Standard sufficient decrease: f_new <= f_old - eta * ||g|| * ||s||
            pred_decrease = -0.5 * np.dot(g, s) # This is for quadratic model with Hessian H^{-1}? 
            # Actually, if s = -H g, and model is f + g^T s + 0.5 s^T H^{-1} s, 
            # then predicted decrease is 0.5 ||g||^2 ||H||? 
            # Let's use the simpler Armijo-like condition:
            # f_new < f_old - eta * ||g|| * ||s||
            
            actual_decrease = f_old - f_new
            threshold = eta * np.linalg.norm(g) * s_norm
            
            if actual_decrease > threshold:
                # Accept step
                # Estimate gradient at new point for Broyden update
                g_new = np.zeros(dim)
                for i in range(dim):
                    if calls >= budget: break
                    xp = x_new.copy()
                    xp[i] += h
                    fp = eval_point(xp)
                    
                    if calls >= budget: break
                    xm = x_new.copy()
                    xm[i] -= h
                    fm = eval_point(xm)
                    
                    g_new[i] = (fp - fm) / (2 * h)
                
                if calls < budget:
                    # Broyden Update
                    # y = g_new - g
                    # r = s - H @ y
                    y = g_new - g
                    r = s - H @ y
                    
                    denom = np.dot(y, r)
                    if abs(denom) > 1e-12:
                        H = H + np.outer(r, r) / denom
                        # Symmetrize to maintain positive definiteness roughly
                        H = (H + H.T) / 2
                        
                        # Check for singularity
                        if np.linalg.cond(H) > 1e12:
                            H = np.eye(dim)
                            
                x = x_new
                f = f_new
                g = g_new
                
                # Expand trust region
                delta = min(max_delta, delta * gamma_expand)
                
            else:
                # Reject step
                # Shrink trust region
                delta = max(min_delta, delta * gamma)
                
                # If delta is too small, we are stuck. Try a random jump?
                if delta < 1e-4:
                    # Random restart from current best
                    x = best_x + rng.normal(0, 1, dim)
                    x = np.clip(x, -5.0, 5.0)
                    f = eval_point(x)
                    g = None
                    H = np.eye(dim)
                    delta = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
