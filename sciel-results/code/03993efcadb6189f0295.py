import numpy as np
import math
import time
from scipy.optimize import minimize, NonlinearConstraint

N = 100
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 10s safety margin

def log_energy(x):
    """
    Computes the logarithmic energy of points on the sphere.
    x: (N, 3) array of points.
    """
    X = x
    # Calculate squared distances: ||x_i - x_j||^2 = 2 - 2 x_i . x_j
    dist_sq = 2.0 - 2.0 * (X @ X.T)
    # Clip to avoid log(0) or negative values due to numerical precision
    np.clip(dist_sq, 1e-12, None, out=dist_sq)
    
    # Energy is sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    # Use upper triangle indices
    iu = np.triu_indices(N, k=1)
    total_energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    return total_energy

def log_energy_grad(x):
    """
    Computes the gradient of the logarithmic energy.
    x: (N, 3) array of points.
    Returns: (N, 3) array of gradients.
    """
    X = x
    dist_sq = 2.0 - 2.0 * (X @ X.T)
    np.clip(dist_sq, 1e-12, None, out=dist_sq)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # grad_i = x_i * sum_{j!=i} (1/d_ij^2) - sum_{j!=i} x_j / d_ij^2
    S = inv_dist_sq.sum(axis=1)
    V = inv_dist_sq @ X
    
    grad = X * S[:, None] - V
    return grad

def constraint_fun(x):
    """
    Equality constraints: ||x_i||^2 - 1 = 0 for each point i.
    x: (3N,) array.
    Returns: (N,) array.
    """
    X = x.reshape(N, 3)
    return np.sum(X * X, axis=1) - 1.0

def constraint_jac(x):
    """
    Jacobian of the constraint function.
    x: (3N,) array.
    Returns: (N, 3N) array.
    """
    X = x.reshape(N, 3)
    jac = np.zeros((N, 3 * N))
    idx = np.arange(N)
    jac[idx, 3 * idx] = 2 * X[:, 0]
    jac[idx, 3 * idx + 1] = 2 * X[:, 1]
    jac[idx, 3 * idx + 2] = 2 * X[:, 2]
    return jac

def construct():
    global START_TIME, DEADLINE
    current_time = time.time()
    if current_time > DEADLINE:
        # Fallback to Fibonacci if already past deadline (shouldn't happen)
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Generate starting points
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random Points
    rng = np.random.default_rng(42)
    x0_rand = rng.normal(size=(N, 3))
    x0_rand /= np.linalg.norm(x0_rand, axis=1, keepdims=True)
    
    # 3. Perturbed Fibonacci
    x0_pert = x0_fib + rng.normal(scale=0.01, size=(N, 3))
    x0_pert /= np.linalg.norm(x0_pert, axis=1, keepdims=True)
    
    starts = [x0_fib, x0_rand, x0_pert]
    
    best_x = x0_fib
    best_energy = log_energy(x0_fib)
    
    # Define constraints for trust-constr
    cons_tc = NonlinearConstraint(constraint_fun, 0, 0, jac=constraint_jac)
    
    # Phase 1: Trust-Constr optimization for each start
    for idx, x0 in enumerate(starts):
        if time.time() > DEADLINE - 30:
            break
            
        try:
            res = minimize(
                log_energy, 
                x0.flatten(), 
                jac=log_energy_grad,
                method='trust-constr',
                constraints=[cons_tc],
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            
            if res.success or res.status == 1:
                x_res = res.x.reshape(N, 3)
                # Normalize to ensure validity
                x_res /= np.linalg.norm(x_res, axis=1, keepdims=True)
                e_res = log_energy(x_res)
                if e_res < best_energy:
                    best_energy = e_res
                    best_x = x_res
        except Exception:
            pass

    # Phase 2: SLSQP Polish on the best result
    if time.time() < DEADLINE - 10:
        # SLSQP constraint format
        slsqp_cons = ({
            'type': 'eq',
            'fun': constraint_fun,
            'jac': constraint_jac
        },)
        
        try:
            res_slsqp = minimize(
                log_energy, 
                best_x.flatten(), 
                jac=log_energy_grad,
                method='SLSQP',
                constraints=slsqp_cons,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if res_slsqp.success:
                x_polished = res_slsqp.x.reshape(N, 3)
                x_polished /= np.linalg.norm(x_polished, axis=1, keepdims=True)
                e_polished = log_energy(x_polished)
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_x = x_polished
        except Exception:
            pass

    return best_x
