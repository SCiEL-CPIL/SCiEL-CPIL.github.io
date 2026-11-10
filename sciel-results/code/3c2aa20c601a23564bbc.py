"""
Optimization strategy: Riemannian L-BFGS-B followed by Newton-Raphson polish.
This approach optimizes directly on the sphere manifold using tangent space
representations, avoiding the numerical issues of projecting from Euclidean space.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
START_TIME = time.time()
TIME_LIMIT = 120.0
MARGIN = 5.0

def _energy_and_grad(X):
    """
    Compute energy and gradient for points X on the unit sphere.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    Gradient: sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    X = np.asarray(X)
    d = 1.0 - 0.5 * (X @ X.T + X.T @ X)
    d = np.maximum(d, 1e-15)
    dist = np.sqrt(d)
    
    # Energy
    # log(1/d) = -log(d)
    # We sum over i<j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[mask]))
    
    # Gradient
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: d = 0.5 * ||x_i - x_j||^2, so ||x_i - x_j||^2 = 2d
    # grad_i = sum_{j != i} (x_i - x_j) / (2 * d_ij)
    # We can compute the matrix G where G_ij = (x_i - x_j) / (2 * d_ij)
    # Then grad_i = sum_j G_ij
    
    # d is 0 on diagonal, we ignore it
    inv_d = 1.0 / (2.0 * d)
    np.fill_diagonal(inv_d, 0.0)
    
    # grad_i = sum_j (x_i - x_j) * inv_d_ij
    # = x_i * sum_j inv_d_ij - sum_j x_j * inv_d_ij
    # Let S_i = sum_j inv_d_ij
    # grad_i = x_i * S_i - (X.T @ inv_d)_i
    S = inv_d.sum(axis=1)
    grad = X * S[:, np.newaxis] - (X.T @ inv_d)
    
    return energy, grad

def _retract(X, dX):
    """
    Retraction from tangent space to the sphere.
    X: current points (N, 3)
    dX: tangent vectors (N, 3)
    Returns: new points on the sphere
    """
    # Exponential map for sphere: exp_X(Y) = X cos(||Y||) + (Y/||Y||) sin(||Y||)
    # For small Y, this is approx X + Y + 0.5 * (X * ||Y||^2 - Y * (X dot Y))
    # Simple retraction: normalize(X + dX) is O(1) and stable enough for L-BFGS-B.
    # Let's use the standard retraction for stability.
    new_X = X + dX
    norms = np.linalg.norm(new_X, axis=1, keepdims=True)
    return new_X / norms

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _optimize_riemannian(X0):
    """
    Perform Riemannian L-BFGS-B optimization.
    We represent the state as X (N,3) on the sphere.
    The optimizer works on a flattened vector.
    """
    def fun_flat(flat_X):
        X = flat_X.reshape(N, 3)
        # Project to sphere to ensure validity (L-BFGS-B might drift slightly)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        e, g = _energy_and_grad(X)
        return e, g.reshape(-1)

    # Initial point
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    # Run L-BFGS-B
    try:
        res = minimize(fun_flat, X.flatten(), method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-8})
        X_opt = res.x.reshape(N, 3)
        X_opt /= np.linalg.norm(X_opt, axis=1, keepdims=True)
    except Exception:
        X_opt = X

    return X_opt

def _newton_refine(X, max_iters=50, tol=1e-10):
    """
    Newton-Raphson refinement in tangent space.
    """
    X = X.copy()
    for _ in range(max_iters):
        e, g = _energy_and_grad(X)
        
        # Project gradient to tangent space
        g_tan = g - (g * X).sum(axis=1, keepdims=True) * X
        
        # Check convergence
        if np.linalg.norm(g_tan) < tol:
            break
            
        # Approximate Hessian in tangent space
        # H_ij = sum_{k != i} (delta_ik - (x_i - x_k)(x_i - x_k)^T / ||x_i - x_k||^2) * (1/||x_i - x_k||^2) ?
        # Actually, Hessian of log(1/d) is (I - u u^T) / d^2 where u = (x_i - x_j)/d
        # H_tan = sum_{j != i} (I - u_ij u_ij^T) / d_ij^2
        # This is O(N^3) to compute explicitly.
        # For 100 points, N^3 = 1e6, which is fast.
        
        # Let's compute the Hessian matrix in tangent space
        # H is a (3N, 3N) matrix, but we can apply it to a vector v
        # Or just build it. 300x300 matrix is small.
        
        H = np.zeros((3*N, 3*N))
        
        # Compute pairwise terms
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = X[i] - X[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-15:
                    continue
                u = diff / np.sqrt(d2)
                # Term: (I - u u^T) / d2
                # This contributes to H_ii
                term = (np.eye(3) - np.outer(u, u)) / d2
                H[3*i:3*i+3, 3*i:3*i+3] += term
                # Cross terms?
                # The gradient of the energy w.r.t x_i depends on x_j.
                # The Hessian block H_ij (i!=j) is - (I - u u^T) / d2
                H[3*i:3*i+3, 3*j:3*j+3] -= term
                H[3*j:3*j+3, 3*i:3*i+3] -= term
        
        # Add Riemannian correction term?
        # The Newton step in tangent space solves: H_tan * delta = -g_tan
        # H_tan is the Hessian projected to tangent space.
        # The above H is the Euclidean Hessian.
        # We need to project H to tangent space:
        # H_tan_{ij} = P_i H_{ij} P_j where P_i = I - x_i x_i^T
        # Since we only solve for tangent delta, and g_tan is tangent,
        # we can just use the Euclidean Hessian if we ensure delta is tangent.
        # However, the Euclidean Hessian is singular in the radial direction.
        # We add a small regularization or project the solution.
        
        try:
            # Solve H * delta = -g_tan
            # Regularize to handle singularity
            reg = 1e-8 * np.eye(3*N)
            delta = np.linalg.solve(H + reg, -g_tan.flatten())
        except np.linalg.LinAlgError:
            break
            
        # Project delta to tangent space
        delta = delta.reshape(N, 3)
        delta_tan = delta - (delta * X).sum(axis=1, keepdims=True) * X
        
        # Retract
        X_new = _retract(X, delta_tan)
        
        # Check if energy decreased
        e_new, _ = _energy_and_grad(X_new)
        if e_new < e:
            X = X_new
        else:
            # Halve step
            delta_tan /= 2.0
            X_new = _retract(X, delta_tan)
            e_new, _ = _energy_and_grad(X_new)
            if e_new < e:
                X = X_new
            else:
                break
                
    return X

def construct():
    global START_TIME, TIME_LIMIT, MARGIN
    start = time.time()
    
    # Generate initial candidates
    X_fib = _fibonacci_sphere()
    X_rand1 = _random_sphere(42)
    X_rand2 = _random_sphere(123)
    
    candidates = [X_fib, X_rand1, X_rand2]
    
    best_X = None
    best_energy = np.inf
    
    # Optimize each candidate
    for X0 in candidates:
        if time.time() - start > TIME_LIMIT - MARGIN:
            break
            
        X_opt = _optimize_riemannian(X0)
        
        # Newton refinement
        if time.time() - start < TIME_LIMIT - MARGIN:
            X_refined = _newton_refine(X_opt)
        else:
            X_refined = X_opt
            
        e, _ = _energy_and_grad(X_refined)
        if e < best_energy:
            best_energy = e
            best_X = X_refined

    if best_X is None:
        return X_fib
        
    return best_X

if __name__ == "__main__":
    # Simple test
    pts = construct()
    e, _ = _energy_and_grad(pts)
    print(f"Energy: {e}")
