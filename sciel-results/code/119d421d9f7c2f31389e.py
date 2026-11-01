"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _optimize_sphere(pts, max_iter=5000, tol=1e-12, deadline=None, initial_lr=0.01):
    """
    Gradient descent on the sphere with backtracking line search.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    lr = initial_lr
    start_time = time.time()
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g) < tol:
            break
        
        # Riemannian gradient: project onto tangent space
        # grad_tangent = g - <g, x> x
        proj = np.einsum('ij,ij->i', g, x)
        grad_t = g - proj[:, np.newaxis] * x
        
        # Line search
        step = grad_t
        new_x = x - lr * step
        norms = np.linalg.norm(new_x, axis=1, keepdims=True)
        new_x = new_x / np.maximum(norms, 1e-15)
        
        new_f, new_g = _energy_and_grad(new_x)
        
        # Backtracking
        while new_f > f and lr > 1e-10:
            lr *= 0.5
            new_x = x - lr * step
            norms = np.linalg.norm(new_x, axis=1, keepdims=True)
            new_x = new_x / np.maximum(norms, 1e-15)
            new_f, new_g = _energy_and_grad(new_x)
        
        x = new_x
        f = new_f
        g = new_g
        
        # Slowly decay learning rate
        lr = max(lr * 0.995, 1e-8)
        
    return x, f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    # Generate multiple diverse initial configurations
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. Random points
    np.random.seed(42)
    for trial in range(5):
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for trial in range(3):
        noise = np.random.randn(N, 3) * 0.1
        pts_pert = pts_fib + noise
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        inits.append(pts_pert)
    
    # Optimize each initialization, keeping the best result
    time_per_init = 145.0 / len(inits)
    
    for init_pts in inits:
        if time.time() > deadline - 5.0:
            break
        
        init_deadline = min(deadline - 5.0, time.time() + time_per_init)
        pts_opt, energy = _optimize_sphere(init_pts, max_iter=3000, tol=1e-12, deadline=init_deadline)
        
        # Ensure points are on the sphere
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / np.maximum(norms, 1e-15)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    # Use the best result found
    if best_pts is None:
        best_pts = pts_fib
    
    return best_pts
# EVOLVE-BLOCK-END
