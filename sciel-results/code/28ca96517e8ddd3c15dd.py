"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -1e18  # invalid: points too close
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _gradient_descent(pts, max_iter=5000, lr_init=0.01, lr_decay=0.999, tol=1e-12, deadline=None):
    """Optimize points on sphere to minimize log energy using gradient descent."""
    pts = pts.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    lr = lr_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances and gradient
        # Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
        # d_ij = |x_i - x_j|
        # grad_i E = sum_{j!=i} (x_i - x_j) / (d_ij^2 * ... )
        # Actually: d/dx_i [log(1/|x_i - x_j|)] = d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
        # So grad_i E = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
        
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-30:
                    continue
                d = np.sqrt(d2)
                grad[i] += -diff / (d2 * d)  # Wait, let me recompute
        
        # Let me redo the gradient properly
        # E = sum_{i<j} -log(|x_i - x_j|)
        # dE/dx_i = -sum_{j!=i} d/dx_i log|x_i - x_j| = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
        # But we need to be careful: for pair (i,j), contribution to x_i is -(x_i-x_j)/|x_i-x_j|^2
        # and to x_j is +(x_i-x_j)/|x_i-x_j|^2 = -(x_j-x_i)/|x_j-x_i|^2
        
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d2 = np.dot(diff, diff)
                if d2 < 1e-30:
                    continue
                d = np.sqrt(d2)
                # Gradient of -log(d) w.r.t. x_i: -(x_i - x_j)/d^2
                # Gradient of -log(d) w.r.t. x_j: +(x_i - x_j)/d^2
                g = diff / d2
                grad[i] += -g
                grad[j] += g
        
        # Project gradient onto tangent space (remove radial component)
        for i in range(n):
            radial = np.dot(grad[i], pts[i])
            grad[i] -= radial * pts[i]
        
        # Normalize gradient for adaptive step
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Take step
        step = lr * grad / gnorm
        pts_new = pts + step
        pts_new = _project_to_sphere(pts_new)
        
        # Check energy
        e_new = _log_energy(pts_new)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-10:
                break
        
        # Decay learning rate
        lr *= lr_decay
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Run gradient descent
    pts_opt, e_opt = _gradient_descent(pts, max_iter=10000, lr_init=0.05, lr_decay=0.9995, deadline=deadline)
    if e_opt < best_energy:
        best_pts = pts_opt
        best_energy = e_opt
    
    # Try a few more iterations if time permits
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Try with different initial learning rates
        for lr_init in [0.02, 0.1, 0.2]:
            if time.time() > deadline - 2.0:
                break
            pts_opt2, e_opt2 = _gradient_descent(best_pts, max_iter=5000, lr_init=lr_init, lr_decay=0.999, deadline=deadline)
            if e_opt2 < best_energy:
                best_pts = pts_opt2
                best_energy = e_opt2
    
    return best_pts
# EVOLVE-BLOCK-END
