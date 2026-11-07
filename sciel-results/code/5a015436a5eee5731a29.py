"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290.0  # 10s safety margin
        
        # Phase 1: Global search to find a good starting point
        # Rosenbrock minimum is near (1,1,...,1) in original space, but shifted/rotated.
        # The minimizer is in [-4, 4]^5.
        # We'll do a small random search to get into the basin of attraction.
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Try 5 random points
        for i in range(5):
            if time.time() > deadline:
                break
            x = np.random.uniform(-4, 4, self.dim)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Local optimization with L-BFGS-B style logic
        # We implement a simplified L-BFGS loop using finite differences.
        
        # L-BFGS parameters
        m = 5  # Memory size
        eps = 1e-5 # Finite difference step
        max_iter = 100 # Max L-BFGS iterations, but limited by budget
        
        # L-BFGS state
        x = best_x
        fx = best_f
        
        # Initialize L-BFGS history
        s_history = [] # step vectors
        y_history = [] # gradient differences
        rho_history = [] # 1 / (y^T s)
        
        def compute_gradient(x, func, calls_ref):
            """Compute gradient using central differences."""
            g = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                # Clip to bounds to avoid out-of-bounds if func doesn't handle it well, 
                # though task says func clips.
                fp = func(x_plus)
                fm = func(x_minus)
                calls_ref[0] += 2
                
                g[i] = (fp - fm) / (2 * eps)
            return g

        g = compute_gradient(x, func, [calls_made])
        
        # L-BFGS main loop
        for _ in range(max_iter):
            if calls_made >= self.budget - 10: # Reserve some calls
                break
            if time.time() > deadline:
                break
            
            # Check for convergence
            if np.linalg.norm(g) < 1e-8:
                break
            
            # Compute search direction p using L-BFGS recursion
            # r = g - sum(rho_i * y_i * (s_i^T p))
            # We do two passes: forward and backward
            
            # Initialize r
            r = g.copy()
            
            # Forward pass
            alpha = np.zeros(len(s_history))
            for i in range(len(s_history) - 1, -1, -1):
                alpha[i] = rho_history[i] * np.dot(s_history[i], r)
                r -= alpha[i] * y_history[i]
            
            # Apply initial Hessian approximation H0 = I (or scaled identity)
            # For Rosenbrock, I is okay, or we can use 1/||g|| * I
            # Let's use I for simplicity, or a simple scaling
            p = r
            
            # Backward pass
            beta = np.zeros(len(s_history))
            for i in range(len(s_history)):
                beta[i] = rho_history[i] * np.dot(y_history[i], p)
                p += (alpha[i] - beta[i]) * s_history[i]
            
            # Line search (Backtracking)
            step_size = 1.0
            c1 = 1e-4
            max_ls_iter = 20
            ls_success = False
            
            for _ in range(max_ls_iter):
                x_new = x + step_size * p
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_made += 1
                
                # Armijo condition
                if f_new <= fx + c1 * step_size * np.dot(g, p):
                    ls_success = True
                    break
                step_size *= 0.5
                
                if time.time() > deadline:
                    break
            
            if not ls_success:
                # If line search fails, try a smaller step or break
                # Try a very small step
                x_new = x + 0.01 * p
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                if f_new < fx:
                    ls_success = True
                else:
                    break # Stagnation
            
            if not ls_success:
                break
            
            # Update L-BFGS history
            s = x_new - x
            y = compute_gradient(x_new, func, [calls_made]) - g
            
            sy = np.dot(s, y)
            if sy > 1e-10: # Curvature condition
                rho = 1.0 / sy
                s_history.append(s)
                y_history.append(y)
                rho_history.append(rho)
                
                # Keep only last m vectors
                if len(s_history) > m:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)
            
            x = x_new
            fx = f_new
            g = compute_gradient(x, func, [calls_made])
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
