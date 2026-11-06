import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances vectorized
    # d_ij = |x_i - x_j|
    # We want sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    # Use broadcasting for efficiency
    # pts shape (N, 3)
    # diff shape (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and diagonal
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    d = np.sqrt(d2)
    # Clip small distances to avoid inf
    d_safe = np.where(mask, d, 1.0)
    d_safe = np.maximum(d_safe, 1e-12)
    
    terms = np.where(mask, -np.log(d_safe), 0.0)
    
    # Sum only i < j
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    return math.fsum(terms[iu[0], iu[1]].tolist())

def _energy_and_grad(pts):
    """
    Compute energy and gradient of log energy on the sphere.
    E = sum_{i<j} -log(|x_i - x_j|)
    grad_i E = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    Note: This is the gradient in R^3. We project to tangent space in the optimizer.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero
    d2_safe = np.where(d2 < 1e-20, 1e-20, d2)
    
    # Factor for gradient: 1/d^2
    # grad_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    # Let W_ij = 1/d_ij^2 for i!=j, 0 otherwise
    W = np.zeros((n, n))
    np.fill_diagonal(W, 0)
    W = 1.0 / d2_safe
    np.fill_diagonal(W, 0)
    
    # grad = - W @ diff ? No.
    # grad_i = - sum_j W_ij (x_i - x_j)
    # Let's compute sum_j W_ij (x_i - x_j)
    # This is (row_sum(W) * pts) - (W @ pts)
    row_sums = W.sum(axis=1)
    W_pts = W @ pts
    raw_grad = row_sums[:, np.newaxis] * pts - W_pts
    grad = -raw_grad
    
    # Compute energy
    d = np.sqrt(d2_safe)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    terms = np.where(mask, -np.log(d), 0.0)
    iu = np.triu_indices(n, k=1)
    energy = math.fsum(terms[iu[0], iu[1]].tolist())
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(n, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _icosahedron_sphere(n):
    """
    Generate points based on icosahedron symmetry, then jitter and project.
    This provides a good symmetric starting point.
    """
    # Start with icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # We need N points. We can replicate and jitter.
    # A better approach for N=100 is to use a grid or just random, 
    # but icosahedron symmetry is good for local minima escape if we jitter.
    # Let's just use random for simplicity in multi-start, 
    # or use a spherical grid.
    # Actually, a "spherical cap" grid or Fibonacci is better.
    # Let's return a jittered icosahedron scaled to N by repeating? No.
    # Let's just return random points for this function in the context of multi-start,
    # or implement a proper spherical grid.
    # For now, let's use a different strategy: 
    # Use the fact that the optimal configuration for N=100 is likely to have some symmetry.
    # But random starts + Newton is robust.
    return _random_sphere(n, seed=42)

def _optimize_newton(pts_init, deadline):
    """
    Optimize using Newton's method on the sphere.
    We solve the linear system for the update in the tangent space.
    """
    pts = pts_init.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Newton step: 
    # We want to minimize f(x) subject to |x_i|=1.
    # Using Lagrange multipliers or projecting gradient.
    # A simpler, robust approach is to use the gradient projected to tangent space
    # and use an adaptive step size (line search) or Newton's method if we have Hessian.
    # Computing Hessian for N=100 is 300x300 matrix, which is fine.
    # H_ij = sum_{k!=i,j} (delta_ij/d_ik^2 - 3 (x_i-x_k)(x_j-x_k)^T / d_ik^4) ... complicated.
    
    # Alternative: Use scipy.optimize.minimize with L-BFGS-B or BFGS.
    # L-BFGS is memory efficient and fast.
    # Let's use L-BFGS-B with bounds? No, sphere constraint.
    # We can parameterize the sphere using stereographic projection or just use 
    # a custom constraint in scipy.
    
    # Better: Use the projected gradient method with a good line search.
    # Or, use the fact that for log energy, the Newton step is well-defined.
    
    # Let's implement a simple Newton-like method using the Hessian approximation 
    # or just use L-BFGS from scipy on the flattened vector with a projection callback.
    
    # SciPy's L-BFGS-B doesn't handle nonlinear constraints like |x|=1 directly well 
    # unless we use SLSQP or trust-constr.
    # Let's use a manual projected gradient descent with Armijo line search.
    # This is robust and fast enough with vectorization.
    
    lr = 0.01
    max_iter = 500
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        e, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        # g_tan_i = grad_i - (grad_i . x_i) x_i
        dots = np.sum(grad * pts, axis=1)
        grad_tan = grad - dots[:, np.newaxis] * pts
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < 1e-12:
            break
            
        # Line search
        alpha = lr
        for _ in range(20):
            pts_new = pts + alpha * grad_tan
            pts_new = _project_to_sphere(pts_new)
            e_new = _log_energy(pts_new)
            
            if e_new < e - 1e-4 * alpha * np.sum(grad_tan * grad_tan): # Armijo
                break
            alpha *= 0.5
            
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
            pts = pts_new
            # Increase learning rate if we made good progress
            lr = min(lr * 1.1, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
                
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Multi-start strategy
    # 1. Fibonacci
    # 2. Random with different seeds
    # 3. Icosahedron-based (if implemented)
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    # Add random starts
    for seed in range(10):
        starts.append(_random_sphere(N, seed=seed))
        
    # Add a start from a previous good solution if we had one, but we don't.
    # So we rely on multi-start.
    
    for i, init_pts in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        # Optimize
        pts_opt, e_opt = _optimize_newton(init_pts, deadline - 2.0)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
            
        # If we have time, try a second pass on the best solution found so far
        # with a smaller learning rate to polish
        if i > 0 and time.time() > deadline - 10.0:
            break
            
    # Final polish on the best solution found
    if best_pts is not None and time.time() < deadline - 5.0:
        # Run a few more iterations with very small lr
        pts_polished, e_polished = _optimize_newton(best_pts, deadline - 1.0)
        if e_polished < best_energy:
            best_pts = pts_polished
            best_energy = e_polished
            
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
        
    return best_pts
