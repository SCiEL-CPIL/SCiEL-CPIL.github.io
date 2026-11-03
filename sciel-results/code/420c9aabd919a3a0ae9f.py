import numpy as np
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s

def _energy_and_grad(X):
    """
    Calculate logarithmic energy and gradient.
    X: (N, 3) array of points.
    Returns: energy, gradient (N, 3)
    """
    # X shape (N, 3)
    # Calculate pairwise distances
    # D[i, j] = |X[i] - X[j]|
    # Use broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero and log(0)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} -log(dist_ij)
    # We sum over all i!=j, so we divide by 2
    # dist = sqrt(dist_sq)
    # log_dist = log(sqrt(dist_sq)) = 0.5 * log(dist_sq)
    # E = -0.5 * sum_{i!=j} log(dist_sq[i,j])
    
    # Gradient:
    # grad_i = sum_{j!=i} (X_j - X_i) / dist_sq[i,j]
    # Note: (X_j - X_i) is -diff[i,j]
    # So grad_i = sum_{j!=i} -diff[i,j] / dist_sq[i,j]
    
    # Initialize grad
    grad = np.zeros_like(X)
    
    # Compute energy
    # Use only upper triangle or just sum all and divide by 2
    # log_dist_sq = np.log(dist_sq)
    # We need to handle dist_sq == 0 for i=j, but we mask them.
    # For i!=j, dist_sq > 0 (unless points coincide, which is a singularity).
    # To be safe, add small epsilon? No, if points coincide, energy is -inf.
    # But gradient is huge.
    # Let's assume points are distinct.
    
    # Efficient computation of gradient
    # inv_dist_sq = np.zeros_like(dist_sq)
    # np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
    
    # grad = np.sum(-diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Let's do it step by step to ensure correctness
    # diff is (N, N, 3). diff[i,j] = X[i] - X[j].
    # We want sum_j (X[j] - X[i]) / |X[i]-X[j]|^2
    # = sum_j -diff[i,j] / dist_sq[i,j]
    
    inv_dist_sq = np.zeros_like(dist_sq)
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
    
    # Weighted sum of diffs
    # grad[i] = sum_j (-diff[i,j] * inv_dist_sq[i,j])
    # diff[i,j] is vector. inv_dist_sq[i,j] is scalar.
    grad = np.sum(-diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    # E = -0.5 * sum_{i!=j} log(dist_sq[i,j])
    log_dist_sq = np.zeros_like(dist_sq)
    np.log(dist_sq, out=log_dist_sq, where=mask)
    energy = -0.5 * np.sum(log_dist_sq)
    
    return energy, grad

def _penalty_and_grad(X, mu=1000.0):
    """
    Penalty term to keep points on unit sphere.
    P = mu * sum_i (|x_i|^2 - 1)^2
    Grad_P_i = 4 * mu * |x_i|^2 * (x_i - ?) -> 4 * mu * (|x_i|^2 - 1) * x_i
    """
    norm_sq = np.sum(X * X, axis=1)
    penalty = mu * np.sum((norm_sq - 1.0) ** 2)
    grad_penalty = 4.0 * mu * (norm_sq - 1.0)[:, np.newaxis] * X
    return penalty, grad_penalty

def _objective(X, mu=1000.0):
    E, G_E = _energy_and_grad(X)
    P, G_P = _penalty_and_grad(X, mu)
    return E + P, G_E + G_P

def _generate_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_random():
    # Uniform random points on sphere
    v = np.random.randn(N, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def construct():
    start_time = time.time()
    
    # Initial candidates
    X0_fib = _generate_fibonacci()
    
    # Run optimization on Fibonacci spiral first
    # We use L-BFGS-B
    # Flatten X
    x0 = X0_fib.flatten()
    
    def fun_flat(x):
        X = x.reshape(N, 3)
        val, grad = _objective(X)
        return val, grad.flatten()
    
    best_X = X0_fib
    best_E = _energy_and_grad(best_X)[0]
    
    # Optimization settings
    # maxiter and ftol to prevent hanging
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 20
    }
    
    # Try Fibonacci start
    try:
        res = minimize(fun_flat, x0, method='L-BFGS-B', jac=True, options=options)
        X_opt = res.x.reshape(N, 3)
        # Normalize to be safe
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        X_opt = X_opt / norms
        E_opt = _energy_and_grad(X_opt)[0]
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT * 0.5:
        return best_X
    
    # Try random starts
    # Number of restarts depends on time
    # Each L-BFGS-B run might take ~1-5 seconds.
    # Let's allow up to 10 restarts if time permits.
    
    for i in range(10):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        X0_rand = _generate_random()
        x0_rand = X0_rand.flatten()
        
        try:
            res = minimize(fun_flat, x0_rand, method='L-BFGS-B', jac=True, options=options)
            X_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            X_opt = X_opt / norms
            E_opt = _energy_and_grad(X_opt)[0]
            if E_opt < best_E:
                best_E = E_opt
                best_X = X_opt
        except Exception:
            continue
            
    # Final check and normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    best_X = best_X / norms
    
    return best_X
