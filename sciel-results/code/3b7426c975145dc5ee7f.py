import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _energy_and_grad(x):
    """
    x: shape (N, 3)
    Returns energy and gradient.
    Energy: sum_{i<j} log(1/||xi - xj||)
    Gradient: grad_k E = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    """
    # x is (N, 3)
    # Compute pairwise distances
    # dist[i, j] = ||x_i - x_j||
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :] # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2) # (N, N)
    
    # Avoid division by zero (points shouldn't be identical, but handle numerically)
    # Set diagonal to 1 to avoid log(0) and 0/0 in grad
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i < j of -log(dist_ij)
    # We can sum all i != j and divide by 2
    # log(1/d) = -log(d)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    energy = -np.sum(np.log(dist[mask])) / 2.0
    
    # Gradient
    # grad_k = sum_{j != k} (x_k - x_j) / dist_kj^2
    # diff is x_i - x_j. So diff[k, j] = x_k - x_j.
    # We want sum over j of diff[k, j] / dist_sq[k, j]
    # dist_sq is (N, N).
    # We need to divide each vector diff[k, j] by dist_sq[k, j].
    # diff / dist_sq[..., np.newaxis]
    inv_dist_sq = 1.0 / dist_sq
    # Set diagonal to 0 to ignore self-interaction
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[k] = sum_j diff[k, j] * inv_dist_sq[k, j]
    # diff is (N, N, 3). inv_dist_sq is (N, N).
    # Multiply diff by inv_dist_sq along the last axis? No, along the j axis.
    # grad = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    grad = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    
    return energy, grad

def _constraints(x):
    """
    x: shape (N, 3)
    Returns list of constraints for trust-constr.
    Each constraint: ||x_i||^2 - 1 = 0
    """
    # Flatten x to (3N,)
    x_flat = x.reshape(-1)
    cons = []
    for i in range(N):
        # Constraint: x[3i]^2 + x[3i+1]^2 + x[3i+2]^2 - 1 = 0
        # We can define a function for each
        def make_constraint(i):
            def cons_func(x_flat):
                return x_flat[3*i:3*i+3].dot(x_flat[3*i:3*i+3]) - 1.0
            def cons_grad(x_flat):
                g = np.zeros(3*N)
                g[3*i:3*i+3] = 2.0 * x_flat[3*i:3*i+3]
                return g
            return {'type': 'eq', 'fun': cons_func, 'jac': cons_grad}
        cons.append(make_constraint(i))
    return cons

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0_flat = x0.reshape(-1)
    
    def fun_flat(x_flat):
        x = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(x)
        return e, g.reshape(-1)
    
    constraints = _constraints(x0_flat)
    
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    try:
        res = minimize(fun_flat, x0_flat, method='trust-constr',
                       constraints=constraints,
                       options={'maxiter': 1000, 'gtol': 1e-8})
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed < timeout and res.success:
            # Try a second polish pass if we have time
            # Use tighter tolerance
            res2 = minimize(fun_flat, res.x, method='trust-constr',
                            constraints=constraints,
                            options={'maxiter': 500, 'gtol': 1e-12})
            if res2.fun < res.fun:
                res = res2
                elapsed = time.time() - start_time
        
        x_opt = res.x.reshape(N, 3)
        
        # Final normalization to ensure unit sphere
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        x_opt = x_opt / norms
        
        return x_opt
        
    except Exception:
        # Fallback to initial guess if optimization fails
        return x0
