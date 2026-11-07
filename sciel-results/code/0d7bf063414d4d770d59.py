"""Optimizer for BBOB f9 (Rosenbrock rotated) using Multi-Start BFGS with finite differences."""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-Start BFGS optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        dim = self.dim
        budget = self.budget
        calls_made = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            # Check budget
            if calls_made >= budget:
                return 1e10
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(dim)
            
            # Clip to box as per task description, though func does it too
            x = np.clip(x, -5.0, 5.0)
            
            f = float(func(x))
            calls_made += 1
            
            if np.isnan(f) or np.isinf(f):
                f = 1e10
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        # Multi-Start Strategy
        # We divide the budget among several restarts.
        # BFGS is a local method, so diverse starts are crucial.
        n_restarts = 5
        fev_per_restart = budget // n_restarts
        
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        # 1. Center
        # 2-5. Random points in a slightly larger box to ensure we cover the shift
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(n_restarts - 1):
            # Sample from [-4.5, 4.5] to stay within [-5,5] but cover the likely minimum location [-4,4]
            x0 = rng.uniform(-4.5, 4.5, size=dim)
            starts.append(x0)

        def estimate_gradient(x):
            nonlocal calls_made
            grad = np.zeros(dim)
            h = 1e-5
            f0 = safe_func(x)
            if calls_made >= budget:
                return None
            
            for j in range(dim):
                if calls_made >= budget - 1:
                    return None
                x_plus = x.copy()
                x_plus[j] += h
                x_minus = x.copy()
                x_minus[j] -= h
                fp = safe_func(x_plus)
                fm = safe_func(x_minus)
                grad[j] = (fp - fm) / (2 * h)
            return grad

        def line_search(x, grad):
            nonlocal calls_made, best_x, best_f
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            f_x = safe_func(x)
            
            if f_x >= budget: # Just a check, safe_func handles it
                 return None, None
                 
            for _ in range(20):
                if calls_made >= budget:
                    return None, None
                x_new = x - alpha * grad
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = safe_func(x_new)
                
                if f_new < f_x - c1 * alpha * np.dot(grad, grad):
                    return x_new, f_new
                alpha *= rho
            return None, None

        for i in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            x = starts[i].copy()
            # Ensure we have enough budget for at least one gradient eval (2*dim+1 calls)
            if budget - calls_made < 2 * dim + 1:
                break
                
            while calls_made < budget - (2 * dim + 1) and time.time() < deadline:
                grad = estimate_gradient(x)
                if grad is None:
                    break
                
                norm_grad = np.linalg.norm(grad)
                if norm_grad < 1e-8:
                    break
                    
                x_new, f_new = line_search(x, grad)
                if x_new is None:
                    break
                    
                x = x_new
                
                # Update global best
                if f_new < best_f:
                    best_f = f_new
                    best_x = x.copy()
                    
                # If we didn't improve significantly or got stuck, break to try next restart
                # Note: safe_func already updated best_x/best_f, so we can just check progress
                # But line_search returns the new point. If it failed, it returns None.
                
        return best_x, best_f
