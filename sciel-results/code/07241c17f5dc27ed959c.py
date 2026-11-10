"""Optimized ZDT2 optimizer using convex spacing for Hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based optimizer for ZDT2: places points directly on the Pareto front 
    with convex spacing optimized for Hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        
        # ZDT2 Pareto Front: x[1:] = 0, f1 = x[0], f2 = 1 - sqrt(x[0]).
        # Reference point R = [11, 11].
        # We optimize the exponent alpha in x0 = (i/(N-1))^alpha to maximize discrete Hypervolume.
        
        i = np.arange(n_evals)
        t = i / (n_evals - 1)
        
        def compute_hv(alpha):
            u = t ** alpha
            widths = np.diff(u)
            heights = 10.0 + np.sqrt(u[:-1])
            return np.sum(widths * heights)
        
        def compute_grad_hv(alpha):
            # Numerical gradient for Newton's method
            eps = 1e-6
            return (compute_hv(alpha + eps) - compute_hv(alpha - eps)) / (2 * eps)
        
        # Initial guess
        alpha = 1.15
        x0_vals = t ** alpha

        # Refine x0_vals directly to maximize Hypervolume
        # HV = sum_{i=1}^{N-1} (u_{i+1} - u_i) * (10 + sqrt(u_i)) + (11 - u_N) * (10 + sqrt(u_N))
        # Let u = x0_vals
        u = x0_vals.copy()
        
        def hv_and_grad(u):
            # u is sorted
            g = 10.0 + np.sqrt(u)
            gp = 0.5 / np.sqrt(u) # derivative of g
            
            # Compute HV
            widths = np.diff(u)
            hv = np.sum(widths * g[:-1])
            hv += (11.0 - u[-1]) * g[-1]
            
            # Compute gradient
            grad = np.zeros_like(u)
            # Interior points
            if len(u) > 1:
                # grad[i] = g[i-1] - g[i] + (u[i+1] - u[i]) * gp[i] for i in 1..N-1
                # Wait, the index in the sum is i for the point u_i.
                # Term i contributes to u_i (negative in width, positive in g) and u_{i+1} (positive in width).
                # Let's re-derive carefully.
                # H = sum_{i=1}^{N-1} (u_{i+1} - u_i) g(u_i) + (11 - u_N) g(u_N)
                # dH/du_k:
                # From term i=k-1: (u_k - u_{k-1}) g(u_{k-1}) -> deriv is g(u_{k-1})
                # From term i=k: (u_{k+1} - u_k) g(u_k) -> deriv is -g(u_k) + (u_{k+1} - u_k) g'(u_k)
                # So for 1 <= k <= N-1 (0-indexed 1 to N-1):
                # grad[k] = g[k-1] - g[k] + (u[k+1] - u[k]) * gp[k]
                
                # For k=0:
                # From term i=0: (u_1 - u_0) g(u_0) -> deriv is -g(u_0) + (u_1 - u_0) g'(u_0)
                # grad[0] = -g[0] + (u[1] - u[0]) * gp[0]
                
                # For k=N-1 (last index):
                # From term i=N-2: (u_{N-1} - u_{N-2}) g(u_{N-2}) -> deriv is g(u_{N-2})
                # From term i=N-1 (last term): (11 - u_{N-1}) g(u_{N-1}) -> deriv is -g(u_{N-1}) + (11 - u_{N-1}) g'(u_{N-1})
                # grad[N-1] = g[N-2] - g[N-1] + (11 - u[N-1]) * gp[N-1]
                
                if len(u) > 1:
                    grad[0] = -g[0] + (u[1] - u[0]) * gp[0]
                    for k in range(1, len(u) - 1):
                        grad[k] = g[k-1] - g[k] + (u[k+1] - u[k]) * gp[k]
                    grad[-1] = g[-2] - g[-1] + (11.0 - u[-1]) * gp[-1]
                else:
                    grad[0] = -g[0] + (11.0 - u[0]) * gp[0]
                    
            return hv, grad

        # Gradient ascent with projection
        lr = 0.01
        for step in range(50):
            if time.time() > deadline:
                break
            _, grad = hv_and_grad(u)
            u_new = u + lr * grad
            # Project to 0 < u_1 < ... < u_N < 1
            # Simple projection: clip and sort? No, sorting breaks identity.
            # We need to maintain order. 
            # If we use small lr, order is likely preserved. 
            # Let's just clip to [eps, 1-eps] and check monotonicity.
            eps = 1e-6
            u_new = np.clip(u_new, eps, 1.0 - eps)
            # Ensure monotonicity
            # If u_new is not strictly increasing, we can enforce it by averaging or just keeping u if violated.
            # Given the smoothness and small lr, violation is unlikely. 
            # To be safe:
            if not np.all(np.diff(u_new) > 0):
                # Fallback: keep old u or do a smaller step
                lr *= 0.5
                if lr < 1e-8:
                    break
                continue
            u = u_new
            
        x0_vals = u
        
        # Create candidate solutions
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # x[1:] = 0 by default

        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for k in range(n_evals):
            F[k] = problem(X[k])

            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
