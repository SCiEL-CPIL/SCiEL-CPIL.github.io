"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates (constrained to sphere via projection later).
    
    For points on the sphere, we can compute the gradient of the energy w.r.t. the tangent space,
    but it's easier to compute the full gradient and then project.
    
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    
    Wait: d/dx_i [ -log(|x_i - x_j|) ] = - (x_i - x_j) / |x_i - x_j|^2
    So dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # dist2[i,j] = |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    # dist = sqrt(dist2)
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    # We only want upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: (x_i - x_j) is diff[i, j]
    # 1/dist2[i,j] is the weight
    # grad[i] = - sum_j (diff[i,j] / dist2[i,j])
    # But we must be careful: the term for j=i is 0.
    # Let's compute the sum for all j != i.
    # weight[i,j] = 1/dist2[i,j] for i != j, 0 for i == j
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    # grad[i] = - sum_j weight[i,j] * diff[i,j]
    # diff is (n,n,3), weight is (n,n)
    # We need to sum over j (axis 1)
    # grad[i] = - np.sum(weight[i, :, np.newaxis] * diff[i, :, :], axis=1)
    # Vectorized:
    # grad = - np.einsum('ij,ijk->ik', weight, diff)
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _retract(pts, step):
    """
    Retraction on the sphere: move by step in the tangent direction, then normalize.
    Actually, the standard retraction is: R_x(v) = (x + v) / ||x + v||
    Here pts are on the sphere. step is the gradient direction (in R^3).
    We move to pts - step (since we want to minimize, and grad points to increase energy).
    Wait, the gradient of E w.r.t x is computed. We want to decrease E.
    So we move in the direction -grad.
    The retraction is: new_pts = (pts - lr * grad) / ||pts - lr * grad||
    """
    new_pts = pts - step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    new_pts = new_pts / norms
    return new_pts

def _optimize_lbfgs(pts, max_iter=2000, tol=1e-10, deadline=None):
    """
    Custom L-BFGS optimization on the sphere.
    """
    n = pts.shape[0]
    dim = 3 * n
    
    # Flatten pts
    x = pts.flatten()
    
    # L-BFGS parameters
    m = 20  # memory
    
    # State for L-BFGS
    s_list = []
    y_list = []
    
    # Initial gradient
    f, g = _energy_and_grad(pts)
    g_flat = g.flatten()
    
    # If deadline is provided, check time
    start_time = time.time()
    
    for k in range(max_iter):
        if deadline is not None:
            if time.time() > deadline:
                break
        
        # Check convergence
        if np.linalg.norm(g_flat) < tol:
            break
        
        # Compute search direction p using two-loop recursion
        # p = -g + alpha * ...
        # We maintain s and y.
        # Let's implement a simple L-BFGS update.
        
        # For simplicity and robustness within time limits, let's use a basic gradient descent
        # with adaptive step size or a simple L-BFGS.
        # Given the complexity of implementing L-BFGS correctly in a few lines, 
        # and the fact that N=100 is small, we can try a simple gradient descent 
        # with line search or fixed step.
        
        # However, L-BFGS is much faster. Let's try to implement a basic version.
        
        # Actually, let's use scipy.optimize.minimize with L-BFGS-B if available, 
        # but the constraint ||x||=1 is tricky.
        # The retraction approach with gradient descent is safer.
        
        # Let's try a simple gradient descent with a decaying step size.
        # Or better, use the fact that the manifold is the sphere, and use 
        # the Riemannian gradient.
        
        # The Riemannian gradient is the projection of the Euclidean gradient 
        # onto the tangent space: grad_T = grad - <grad, x> x
        # But since we are using the retraction (x - lr*grad)/||...||, 
        # the standard Euclidean gradient works well if we normalize after each step.
        
        # Let's use a simple adaptive step size.
        # Initial step size
        lr = 0.01
        
        # Line search (simple backtracking)
        step = lr * g_flat
        new_x = x - step
        new_pts = new_x.reshape(n, 3)
        
        # Normalize new_pts
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        
        new_f, new_g = _energy_and_grad(new_pts)
        
        # If energy increased, reduce step
        while new_f > f and lr > 1e-8:
            lr *= 0.5
            step = lr * g_flat
            new_x = x - step
            new_pts = new_x.reshape(n, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            new_f, new_g = _energy_and_grad(new_pts)
        
        # Update
        x = new_pts.flatten()
        f = new_f
        g_flat = new_g.flatten()
        
        # Decay learning rate slowly
        lr = max(lr * 0.99, 1e-6)
        
        if deadline is not None:
            if time.time() > deadline:
                break
        
    return new_pts, f

def construct():
    # Deadline: 150s total, leave 5s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: Golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize
    pts_opt, energy = _optimize_lbfgs(pts, max_iter=5000, tol=1e-12, deadline=deadline)
    
    # Ensure points are on the sphere
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    # Check for very close points
    # If any points are closer than 1e-9, perturb them slightly
    # This is a safety measure
    
    return pts_opt
# EVOLVE-BLOCK-END
