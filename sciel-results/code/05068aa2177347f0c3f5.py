"""Improved: vectorized Riemannian CG with Armijo line search + Newton polish."""
import time
import math
import numpy as np

N = 150
DEADLINE = 115.0  # safety margin before 120s limit


def _pairwise_dists(pts):
    """Compute pairwise distances matrix."""
    sq = np.sum(pts * pts, axis=1)
    sq_d = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(sq_d, 0.0, out=sq_d)
    return np.sqrt(sq_d)


def _log_energy(pts):
    """Compute logarithmic energy."""
    d = _pairwise_dists(pts)
    np.fill_diagonal(d, 1.0)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(d[iu]))


def _energy_and_grad(pts):
    """Compute log energy and Riemannian gradient (tangent space)."""
    d = _pairwise_dists(pts)
    np.fill_diagonal(d, 1.0)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(d[iu]))
    
    # Gradient: for each i, sum over j!=i of (p_j - p_i) / |p_i - p_j|^2
    # Vectorized: grad[i] = sum_j (p_j - p_i) / d[i,j]^2
    inv_d2 = np.zeros_like(d)
    np.fill_diagonal(d, 1.0)
    inv_d2 = 1.0 / (d * d)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j inv_d2[i,j] * (p_j - p_i)
    #         = sum_j inv_d2[i,j] * p_j - (sum_j inv_d2[i,j]) * p_i
    row_sums = inv_d2.sum(axis=1)
    grad = (inv_d2 @ pts) - row_sums[:, None] * pts
    
    # Project to tangent space
    g_dot_p = (grad * pts).sum(axis=1)
    grad = grad - g_dot_p[:, None] * pts
    
    return energy, grad


def _normalize(pts):
    """Normalize to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exp_map(pts, delta):
    """Exponential map on sphere: move points along tangent vectors."""
    # For small steps, use: new_p = cos(|delta|)*p + sin(|delta|)/|delta| * delta
    # For simplicity and stability, use: new_p = normalize(p + delta)
    return _normalize(pts + delta)


def _riemannian_cg(pts, max_iter=2000, tol=1e-12):
    """Riemannian conjugate gradient with Armijo line search."""
    energy, grad = _energy_and_grad(pts)
    d = -grad  # search direction
    prev_grad = grad.copy()
    
    for k in range(max_iter):
        # Armijo line search
        alpha = 0.1
        c1 = 1e-4
        slope = (grad * d).sum()
        
        if slope >= 0:
            # Direction is not descent, reset
            d = -grad
            prev_grad = grad.copy()
            slope = (grad * d).sum()
        
        for ls in range(50):
            new_pts = _exp_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy <= energy + c1 * alpha * slope:
                break
            alpha *= 0.5
        
        else:
            # Line search failed, try smaller step
            alpha = 1e-6
            new_pts = _exp_map(pts, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Check convergence
        if np.abs(energy - new_energy) < tol and np.linalg.norm(new_grad) < 1e-10:
            return new_pts, new_energy
        
        # Update
        pts = new_pts
        energy = new_energy
        
        # Conjugate gradient update (Hestenes-Stiefel)
        grad = new_grad
        if k > 0:
            beta = np.sum((grad - prev_grad) * grad) / np.sum(prev_grad * prev_grad)
            beta = max(0.0, beta)
            d = -grad + beta * d
        prev_grad = grad
    
    return pts, energy


def _newton_polish(pts, max_iter=200):
    """Newton-type polish using approximate Hessian via finite differences."""
    energy, grad = _energy_and_grad(pts)
    
    for k in range(max_iter):
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Approximate Hessian action using finite differences
        # Use a simple approach: take a step in -grad direction and measure curvature
        step = 1e-5
        new_pts = _exp_map(pts, -step * grad)
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Approximate Hessian eigenvalue in gradient direction
        delta_grad = new_grad - grad
        hessian_action = delta_grad / step
        curvature = np.sum(hessian_action * grad)
        
        if curvature > 1e-8:
            # Newton step
            newton_step = -grad / (curvature / np.sum(grad * grad) + 1e-10)
            new_pts2 = _exp_map(pts, newton_step)
            new_energy2, _ = _energy_and_grad(new_pts2)
            
            if new_energy2 < energy:
                pts = new_pts2
                energy = new_energy2
                _, grad = _energy_and_grad(pts)
            else:
                # Fallback to gradient step
                pts = new_pts
                energy = new_energy
                _, grad = _energy_and_grad(pts)
        else:
            pts = new_pts
            energy = new_energy
            _, grad = _energy_and_grad(pts)
    
    return pts, energy


def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n):
    """Generate points using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        w = halton(i, 5)
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u
        cos_phi = 2.0 * v - 1.0
        sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
        points.append([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
    return np.array(points)


def construct():
    start_time = time.time()
    
    # Generate initial points using multiple methods
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
    ]
    
    # Add some random inits
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 3)
        inits.append(_normalize(pts))
    
    best_points = None
    best_energy = float('inf')
    
    for init_pts in inits:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Optimize with Riemannian CG
        opt_pts, opt_energy = _riemannian_cg(init_pts, max_iter=1500, tol=1e-13)
        
        if opt_energy < best_energy:
            best_points = opt_pts.copy()
            best_energy = opt_energy
    
    # Newton polish on the best result
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        polished_pts, polished_energy = _newton_polish(best_points, max_iter=100)
        if polished_energy < best_energy:
            best_points = polished_pts
            best_energy = polished_energy
    
    # Final CG pass with tighter tolerance
    if time.time() - start_time < DEADLINE - 3:
        final_pts, final_energy = _riemannian_cg(best_points, max_iter=500, tol=1e-15)
        if final_energy < best_energy:
            best_points = final_pts
            best_energy = final_energy
    
    # Ensure valid output
    best_points = _normalize(best_points)
    
    return best_points
