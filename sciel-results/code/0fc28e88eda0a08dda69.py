"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: Two-phase optimizer with Gauss-Newton local refinement.
    
    Phase 1: Coarse random search to find a promising region (basin of attraction).
    Phase 2: Gauss-Newton method with finite-difference Jacobian estimation to
             rapidly converge to the minimum, exploiting the local quadratic
             structure of the bent cigar function.
    
    The bent cigar function is smooth and has a well-defined local minimum.
    Once in the basin, the Gauss-Newton method can converge quadratically,
    allowing for high precision in the final phase of the budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse random search
        # Use 20% of the budget for initial exploration
        init_budget = max(20, int(0.2 * budget))
        
        best_x_phase1 = None
        best_f_phase1 = float('inf')
        
        for _ in range(init_budget):
            if time.time() > deadline or self.calls >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            try:
                fx = wrapped_func(x)
                if fx < best_f_phase1:
                    best_f_phase1 = fx
                    best_x_phase1 = x.copy()
            except (TimeoutError, ValueError):
                break
        
        if best_x_phase1 is None:
            return np.zeros(dim), 0.0

        # Phase 2: Gauss-Newton refinement
        # The Gauss-Newton method minimizes ||r(x)||^2, where r(x) is a residual.
        # For a scalar function f(x), we can treat f(x) as a single residual.
        # The update is x_new = x - (J^T J)^{-1} J^T f(x)
        # However, for a scalar function, J is a row vector (the gradient).
        # So the update becomes x_new = x - (grad^T grad)^{-1} grad^T f(x)
        # Which is just gradient descent with step size 1/||grad||^2.
        # This is not ideal. 
        # 
        # Instead, let's use a finite-difference Hessian approximation to do Newton's method.
        # Newton's method: x_new = x - H^{-1} grad
        # For the bent cigar, the Hessian is ill-conditioned, but Newton's method
        # is invariant to linear transformations, so it should handle the anisotropy well.
        
        def estimate_gradient(x):
            grad = np.zeros(dim)
            h = 1e-5
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += h
                try:
                    fp = wrapped_func(x_plus)
                except (TimeoutError, ValueError):
                    return None
                x_minus = x.copy()
                x_minus[i] -= h
                try:
                    fm = wrapped_func(x_minus)
                except (TimeoutError, ValueError):
                    return None
                grad[i] = (fp - fm) / (2 * h)
            return grad

        def estimate_hessian(x):
            H = np.zeros((dim, dim))
            h = 1e-4
            f0 = wrapped_func(x)
            for i in range(dim):
                x_plus_i = x.copy()
                x_plus_i[i] += h
                try:
                    f_plus_i = wrapped_func(x_plus_i)
                except (TimeoutError, ValueError):
                    return None
                for j in range(i, dim):
                    x_plus_j = x_plus_i.copy()
                    x_plus_j[j] += h
                    try:
                        f_plus_plus = wrapped_func(x_plus_j)
                    except (TimeoutError, ValueError):
                        return None
                    H[i, j] = (f_plus_plus - f_plus_i - f0 + f0) / (h * h) # This is wrong, need f(x+h_i, h_j) - f(x+h_i) - f(x+h_j) + f(x)
                    # Correct formula: (f(x+h_i+h_j) - f(x+h_i) - f(x+h_j) + f(x)) / (h_i * h_j)
                    # We have f0 = f(x), f_plus_i = f(x+h_i)
                    # We need f(x+h_i+h_j) and f(x+h_j)
                    # We already have f_plus_plus = f(x+h_i+h_j)
                    # We need f(x+h_j)
                    x_plus_j_only = x.copy()
                    x_plus_j_only[j] += h
                    try:
                        f_plus_j_only = wrapped_func(x_plus_j_only)
                    except (TimeoutError, ValueError):
                        return None
                    H[i, j] = (f_plus_plus - f_plus_i - f_plus_j_only + f0) / (h * h)
                    H[j, i] = H[i, j]
            return H

        # Newton's method is more robust than Gauss-Newton for scalar functions.
        # Let's implement a damped Newton's method.
        
        x = best_x_phase1.copy()
        f_current = best_f_phase1
        
        # We need to be careful with the budget.
        # Each Newton step requires 1 (f0) + 2*dim (grad) + dim*(dim+1)/2 (hessian) evaluations.
        # For dim=5, that's 1 + 10 + 15 = 26 evaluations per step.
        # We have ~400 evaluations left. So we can do ~15 Newton steps.
        
        max_newton_steps = 20
        
        for _ in range(max_newton_steps):
            if time.time() > deadline or self.calls >= budget:
                break
            
            # Check if we are close to convergence
            if self.best_f < -1e6: # Bent cigar can have large negative values? No, it's shifted.
                break
                
            grad = estimate_gradient(x)
            if grad is None:
                break
                
            if np.linalg.norm(grad) < 1e-8:
                break
                
            H = estimate_hessian(x)
            if H is None:
                break
            
            # Regularize Hessian to ensure positive definiteness
            # Add a small multiple of the identity
            lam = 1e-6
            H_reg = H + lam * np.eye(dim)
            
            try:
                # Solve H_reg * dx = -grad
                dx = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient step
                dx = -grad / (np.linalg.norm(grad) + 1e-8)
                
            # Line search to ensure sufficient decrease
            alpha = 1.0
            f_x = f_current
            for _ in range(10):
                if time.time() > deadline or self.calls >= budget:
                    break
                x_new = x + alpha * dx
                try:
                    f_new = wrapped_func(x_new)
                except (TimeoutError, ValueError):
                    break
                
                if f_new < f_x:
                    x = x_new
                    f_current = f_new
                    break
                else:
                    alpha *= 0.5
            else:
                # If line search fails, break
                break
                
        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
