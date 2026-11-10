"""Rewritten program: minimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import math
import time

N = 100


def _energy_and_grad(pts):
    """Compute log-energy E = sum_{i<j} log(1/|x_i-x_j|) and gradient.
    
    Energy = -sum_{i<j} log(|x_i-x_j|)
    grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Pairwise distances
    # diff[i,j] = pts[i] - pts[j], shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    E = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j!=i} diff[i,j] / dist2[i,j]
    # dist2 has 1.0 on diagonal, so we need to zero out diagonal contribution
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return E, grad


def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _newton_step(pts, max_iter=50, tol=1e-12):
    """Newton's method on the sphere using Lagrange multipliers.
    
    Minimize E(x) subject to |x_i|^2 = 1.
    Use the standard approach: compute gradient, project to tangent space,
    solve for step using Hessian approximation (or use conjugate gradient on manifold).
    
    Here we use a simple projected gradient / Newton hybrid.
    """
    pts = _project_to_sphere(pts.copy())
    
    for it in range(max_iter):
        E, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space: grad_tangent_i = grad_i - (grad_i . x_i) x_i
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tan = grad - dot * pts
        
        # Check convergence
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break
        
        # Simple gradient descent with adaptive step size (backtracking)
        # Step size: start with something reasonable
        alpha = 0.01
        for _ in range(50):
            new_pts = _project_to_sphere(pts - alpha * grad_tan)
            new_E, _ = _energy_and_grad(new_pts)
            if new_E < E:
                break
            alpha *= 0.5
        
        pts = _project_to_sphere(pts - alpha * grad_tan)
    
    return pts, _energy_and_grad(pts)[0]


def _conjugate_gradient_manifold(pts, max_iter=200, tol=1e-12):
    """Conjugate gradient on the sphere (Riemannian CG).
    
    Uses the fact that we can move along tangent directions and project back.
    """
    pts = _project_to_sphere(pts.copy())
    n = pts.shape[0]
    
    E, grad = _energy_and_grad(pts)
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    g = grad - dot * pts  # tangent gradient
    
    # Initial direction
    d = -g
    # Make d tangent
    dot_d = np.sum(pts * d, axis=1, keepdims=True)
    d = d - dot_d * pts
    
    best_E = E
    best_pts = pts.copy()
    
    for it in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            break
        
        # Line search: find step size along direction d
        alpha = 0.01
        for _ in range(30):
            # Move along d and project
            new_pts = _project_to_sphere(pts + alpha * d)
            new_E, new_grad = _energy_and_grad(new_pts)
            if new_E < best_E:
                best_E = new_E
                best_pts = new_pts.copy()
            if new_E < E:
                break
            alpha *= 0.5
        
        if alpha < 1e-15:
            break
        
        pts = _project_to_sphere(pts + alpha * d)
        E, grad = _energy_and_grad(pts)
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        new_g = grad - dot * pts
        
        # Beta (Polak-Ribiere)
        dot_gg = np.sum(g * new_g)
        dot_gg_old = np.sum(g * g)
        if dot_gg_old > 1e-30:
            beta = max(0.0, dot_gg / dot_gg_old)
        else:
            beta = 0.0
        
        d = -new_g + beta * d
        # Ensure d is tangent
        dot_d = np.sum(pts * d, axis=1, keepdims=True)
        d = d - dot_d * pts
        
        g = new_g
    
    return best_pts, best_E


def _local_search(pts, max_iters=500, tol=1e-14):
    """Local search using Riemannian gradient descent with adaptive steps.
    
    Uses a combination of gradient descent and occasional Newton-like steps.
    """
    pts = _project_to_sphere(pts.copy())
    E, grad = _energy_and_grad(pts)
    
    best_E = E
    best_pts = pts.copy()
    
    for it in range(max_iters):
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        g = grad - dot * pts
        gnorm = np.linalg.norm(g)
        
        if gnorm < tol:
            break
        
        # Adaptive step size
        alpha = min(0.1, 0.01 * (1.0 + 50.0 / (it + 1.0)))
        
        for _ in range(40):
            new_pts = _project_to_sphere(pts - alpha * g)
            new_E, new_grad = _energy_and_grad(new_pts)
            if new_E < best_E:
                best_E = new_E
                best_pts = new_pts.copy()
            if new_E < E:
                break
            alpha *= 0.5
        
        if alpha < 1e-18:
            break
        
        pts = _project_to_sphere(pts - alpha * g)
        E, grad = _energy_and_grad(pts)
    
    return best_pts, best_E


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    # Start from Fibonacci sphere
    pts = _fibonacci_sphere()
    E, _ = _energy_and_grad(pts)
    
    best_pts = pts.copy()
    best_E = E
    
    # Phase 1: Coarse gradient descent
    pts, E = _local_search(pts, max_iters=200, tol=1e-10)
    if E < best_E:
        best_E = E
        best_pts = pts.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return best_pts
    
    # Phase 2: Riemannian conjugate gradient for finer optimization
    pts, E = _conjugate_gradient_manifold(best_pts, max_iter=300, tol=1e-12)
    if E < best_E:
        best_E = E
        best_pts = pts.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return best_pts
    
    # Phase 3: Final high-precision polish
    pts, E = _local_search(best_pts, max_iters=500, tol=1e-14)
    if E < best_E:
        best_E = E
        best_pts = pts.copy()
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                diff = best_pts[i] - best_pts[j]
                if np.linalg.norm(diff) < 1e-15:
                    diff = np.random.randn(3)
                diff /= np.linalg.norm(diff)
                best_pts[i] += 1e-8 * diff
                best_pts[j] -= 1e-8 * diff
                best_pts[i] /= np.linalg.norm(best_pts[i])
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
