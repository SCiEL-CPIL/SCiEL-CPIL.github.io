"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import fsum, log

N = 100
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0  # seconds to keep before the hard limit


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # Use vectorized approach but be careful with memory for N=100 (manageable)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal and lower triangle to avoid self and double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    # Filter out any zero distances (shouldn't happen with valid points)
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    terms = -np.log(d)  # log(1/d) = -log(d)
    return fsum(terms.tolist())


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    d_safe = np.maximum(d, 1e-15)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    energy = -np.sum(np.log(d_safe))
    
    # Gradient: d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Note: -log(1/d) = log(d), so d/dx log(d) = (x_i - x_j)/d^2 for the term log(d_ij)
    # But we have -log(d_ij), so derivative is -(x_i - x_j)/d^2 = (x_j - x_i)/d^2
    
    # Vectorized gradient computation
    # dists^2 matrix
    d2 = dists * dists
    d2_safe = np.maximum(d2, 1e-30)
    
    # We want: grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # = sum_{j != i} x_j / d_ij^2 - x_i * sum_{j != i} 1/d_ij^2
    
    # Create weight matrix W where W[i,j] = 1/d_ij^2 for i != j, 0 otherwise
    W = np.zeros((n, n))
    W[mask] = 1.0 / d2_safe[mask]
    
    # grad_i = (sum_j W[i,j] * x_j) - x_i * (sum_j W[i,j])
    # sum_j W[i,j] * x_j = W @ pts
    # sum_j W[i,j] = W.sum(axis=1)
    
    Wx = W @ pts
    Wsum = W.sum(axis=1, keepdims=True)
    
    grad = Wx - pts * Wsum
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent(pts, max_iters=5000, lr=0.01, tol=1e-10, time_limit=None, start_time=None):
    """Constrained gradient descent on the sphere using projection."""
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        energy, grad = _log_energy_and_grad(pts)
        
        # Take a step in the negative gradient direction
        # For sphere constraint, we can use a simple projected gradient method
        # or move in the tangent space
        # Simpler: move in Euclidean space and project back
        step = lr * grad
        new_pts = pts - step
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        # Check convergence
        if it > 0 and abs(energy - best_energy) < tol:
            break
        
        pts = new_pts
    
    return best_pts, best_energy


def _backtracking_line_search(pts, grad, energy, max_alpha=1.0, min_alpha=1e-10, c=1e-4, max_iter=50, time_limit=None, start_time=None):
    """Backtracking line search for gradient descent on sphere."""
    if start_time is None:
        start_time = time.time()
    
    alpha = max_alpha
    for _ in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        new_pts = _project_to_sphere(pts - alpha * grad)
        new_energy = _log_energy(new_pts)
        
        if new_energy <= energy - c * alpha * np.sum(grad * grad):
            return alpha, new_pts, new_energy
        alpha *= 0.5
    
    return alpha, _project_to_sphere(pts - alpha * grad), _log_energy(_project_to_sphere(pts - alpha * grad))


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Phase 1: Gradient descent with backtracking line search
    lr = 0.05
    max_iters = 2000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        energy, grad = _log_energy_and_grad(pts)
        
        # Check for convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Backtracking line search
        alpha, new_pts, new_energy = _backtracking_line_search(
            pts, grad, energy, max_alpha=lr, time_limit=deadline, start_time=start_time
        )
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        # Adaptive learning rate
        if new_energy < energy:
            lr = min(lr * 1.05, 0.5)
        else:
            lr = max(lr * 0.5, 0.001)
        
        pts = new_pts
    
    # Phase 2: Fine-tuning with smaller steps
    if time.time() < deadline:
        pts = best_pts.copy()
        lr = 0.01
        max_iters_fine = 1000
        
        for it in range(max_iters_fine):
            if time.time() > deadline:
                break
            
            energy, grad = _log_energy_and_grad(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            alpha, new_pts, new_energy = _backtracking_line_search(
                pts, grad, energy, max_alpha=lr, time_limit=deadline, start_time=start_time
            )
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
            
            if new_energy < energy:
                lr = min(lr * 1.02, 0.05)
            else:
                lr = max(lr * 0.5, 0.0005)
            
            pts = new_pts
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts = _project_to_sphere(best_pts)
                break
    
    return best_pts
