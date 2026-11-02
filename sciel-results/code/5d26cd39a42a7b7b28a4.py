"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(pts):
    """Compute log(1/d) energy sum over pairs, with fsum."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # Use broadcasting: (n, 1, 3) - (1, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    # We only want i < j, so take upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Compute log(1/d) = -log(d)
    terms = -np.log(d)
    return math.fsum(terms.tolist())

def _log_energy_grad(pts):
    """Compute gradient of log energy. d/dx_i sum_{j!=i} log(1/|x_i-x_j|)
    = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    Wait: d/dx_i log(1/|x_i-x_j|) = d/dx_i (-log|x_i-x_j|) = -(x_i-x_j)/|x_i-x_j|^2
    So grad_i = -sum_{j!=i} (x_i-x_j)/|x_i-x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist2 = np.sum(diff * diff, axis=2)  # (n,n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist2, 1.0)
    # grad_i = -sum_{j!=i} (x_i-x_j) / |x_i-x_j|^2
    # = -sum_{j!=i} diff[i,j] / dist2[i,j]
    inv_dist2 = 1.0 / dist2  # (n,n)
    # Set diagonal to 0
    np.fill_diagonal(inv_dist2, 0.0)
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _constrained_energy_and_grad(pts):
    """Compute energy and gradient projected to tangent space of sphere constraint.
    For a point x on the sphere, the gradient of the constrained problem is
    (I - x x^T) grad_unconstrained.
    """
    e = _log_energy(pts)
    g = _log_energy_grad(pts)
    # Project gradient to tangent space
    # g_tangent = g - (g . x) x
    dot = np.sum(g * pts, axis=1, keepdims=True)
    g_tangent = g - dot * pts
    return e, g_tangent

def _lbfgs_sphere(pts, max_iter=5000, tol=1e-12):
    """L-BFGS optimization on the sphere using retraction.
    We use the exponential map / retraction: x_new = normalize(x + alpha * v)
    where v is in the tangent space.
    """
    n = pts.shape[0]
    x = _project_to_sphere(pts.copy())
    m = 20  # L-BFGS memory
    s_list = []  # list of (s, y)
    rho_list = []
    W = np.zeros((3, 3))  # for L-BFGS two-loop recursion, but we'll do it in terms of vectors
    
    # Actually, let's implement a simple L-BFGS with the two-loop recursion for the direction
    # But since we're on a manifold, we need to be careful.
    # Let's use a simpler approach: gradient descent with adaptive step size and momentum,
    # or use scipy.optimize.minimize with a constraint.
    
    # Let's use scipy if available, otherwise implement a simple method.
    # Actually, let's implement a proper L-BFGS on the sphere.
    # The retraction is R_x(v) = normalize(x + v) for v in tangent space.
    # The gradient in the ambient space projected to tangent space is what we have.
    
    # Let's do a simple steepest descent with backtracking line search for robustness,
    # then polish with a more sophisticated method if time allows.
    
    best_x = x.copy()
    best_e = _log_energy(x)
    
    # Gradient descent with adaptive step
    lr = 0.01
    for it in range(max_iter):
        e, g_t = _constrained_energy_and_grad(x)
        if it % 100 == 0:
            e_check = _log_energy(x)
            if e_check < best_e:
                best_e = e_check
                best_x = x.copy()
        
        # Check convergence
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Backtracking line search
        alpha = lr
        for _ in range(30):
            x_new = _project_to_sphere(x + alpha * g_t)
            # Wait, gradient descent should go in direction -g_t
            x_new = _project_to_sphere(x - alpha * g_t)
            e_new = _log_energy(x_new)
            if e_new < e:
                x = x_new
                break
            alpha *= 0.5
        else:
            # If no improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-15:
                break
        # Occasionally increase learning rate if stuck
        if it % 50 == 0 and lr < 0.01:
            lr = min(lr * 1.5, 0.01)
    
    e_final = _log_energy(x)
    if e_final < best_e:
        best_x = x.copy()
        best_e = e_final
    
    return best_x, best_e

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    pts = _project_to_sphere(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # First pass: gradient descent
    pts, e = _lbfgs_sphere(pts, max_iter=3000, tol=1e-10)
    if e < best_energy:
        best_energy = e
        best_pts = pts.copy()
    
    if time.time() > deadline - 20:
        return best_pts
    
    # Second pass: restart with perturbation
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        # Perturb best solution
        noise = np.random.randn(N, 3) * 0.1
        pts_pert = _project_to_sphere(best_pts + noise)
        pts, e = _lbfgs_sphere(pts_pert, max_iter=2000, tol=1e-11)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Final polish
    if time.time() < deadline - 2:
        best_pts, best_energy = _lbfgs_sphere(best_pts, max_iter=5000, tol=1e-13)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
