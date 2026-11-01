"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    # Compute all pairwise distances
    # Using broadcasting for vectorized computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal (self-distances)
    np.fill_diagonal(dists, 1.0)
    # Compute log(1/d) = -log(d) for upper triangle only
    iu = np.triu_indices(n, k=1)
    vals = -np.log(dists[iu])
    return math.fsum(vals.tolist())


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere.
    Returns (energy, grad) where grad is shape (N, 3).
    The gradient is projected to the tangent space of the sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-30)
    dists = np.sqrt(dists_sq)
    # log(1/d) = -log(d), derivative w.r.t. x_i: 
    # d/dx_i [-log(|x_i - x_j|)] = (x_j - x_i) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Note: this is the gradient of the energy w.r.t. position in R^3
    # The factor: d/dx_i log(1/|x_i-x_j|) = -d/dx_i log|x_i-x_j| = -(x_i-x_j)/|x_i-x_j|^2 = (x_j-x_i)/|x_i-x_j|^2
    
    # Build gradient matrix
    # G[i,j] = (x_j - x_i) / |x_i - x_j|^2 for i != j, 0 otherwise
    # We can compute this vectorized:
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) * inv_dists_sq[i,j]
    # = sum_j x_j * inv_dists_sq[i,j] - x_i * sum_j inv_dists_sq[i,j]
    sum_inv = np.sum(inv_dists_sq, axis=1, keepdims=True)  # (N, 1)
    grad = pts.T @ (inv_dists_sq * 1.0) - pts * sum_inv.T
    # Wait, let me redo this more carefully.
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = sum_{j != i} x_j / |x_i - x_j|^2 - x_i * sum_{j != i} 1/|x_i - x_j|^2
    
    # sum_j x_j * inv_dists_sq[i,j] = (inv_dists_sq @ pts).T? No.
    # Let M = inv_dists_sq (N x N), M[i,j] = 1/|x_i-x_j|^2
    # sum_j M[i,j] * x_j = (M @ pts).T? No, M @ pts gives (N x 3) where row i is sum_j M[i,j]*pts[j]
    term1 = inv_dists_sq @ pts  # (N, 3), row i = sum_j M[i,j] * x_j
    term2 = pts * sum_inv.T  # (N, 3), row i = x_i * sum_j M[i,j]
    grad = term1 - term2
    
    # Project gradient to tangent space: grad_tan = grad - (grad . x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - dot * pts
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    vals = -np.log(dists[iu])
    energy = math.fsum(vals.tolist())
    
    return energy, grad_tan


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _golden_spiral(n):
    """Generate initial points using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lbfgs_optimize(pts0, max_iter=5000, time_limit=None, start_time=None):
    """Simple L-BFGS optimization on the sphere using retraction."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    pts = _project_to_sphere(pts0.copy())
    n = len(pts)
    
    # L-BFGS memory
    m = 10
    W = []  # list of (s, y) pairs
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Line search parameters
    c1 = 1e-4
    c2 = 0.9
    rho_min = 1e-10
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _log_energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        if len(W) == 0:
            s_dir = -grad
        else:
            alpha = np.zeros(m)
            q = grad.copy()
            # Backward loop
            for k in range(len(W) - 1, -1, -1):
                s_k, y_k, gamma_k = W[k]
                alpha[k] = gamma_k * np.dot(s_k, q)
                q = q - alpha[k] * y_k
            
            if len(W) > 0:
                s_last, y_last, _ = W[-1]
                y_norm_sq = np.dot(y_last, y_last)
                if y_norm_sq > 1e-15:
                    gamma0 = np.dot(s_last, y_last) / y_norm_sq
                else:
                    gamma0 = 1.0
            else:
                gamma0 = 1.0
            
            r = q / gamma0
            
            # Forward loop
            for k in range(len(W)):
                s_k, y_k, gamma_k = W[k]
                beta_k = np.dot(y_k, r)
                r = r + (alpha[k] - beta_k) * s_k
            
            s_dir = -r
        
        # Normalize direction to have unit norm for step size
        dir_norm = np.linalg.norm(s_dir)
        if dir_norm < 1e-15:
            break
        s_dir = s_dir / dir_norm
        
        # Line search with retraction
        step = 1.0
        found = False
        for ls in range(30):
            # Retraction: exponential map approximated by normalization
            new_pts = _project_to_sphere(pts + step * s_dir)
            new_energy = _log_energy(new_pts)
            
            # Armijo condition
            if new_energy <= energy + c1 * step * np.dot(grad, s_dir):
                found = True
                break
            step *= 0.5
        
        if not found:
            # Try smaller step
            new_pts = _project_to_sphere(pts + 1e-6 * s_dir)
            new_energy = _log_energy(new_pts)
            if new_energy >= energy:
                break
        
        # Update L-BFGS memory
        s_step = new_pts - pts
        y_step = _log_energy_grad(new_pts)[1] - grad  # We need new grad
        # Recompute new grad properly
        _, new_grad = _log_energy_grad(new_pts)
        y_step = new_grad - grad
        
        sy = np.dot(s_step, y_step)
        if sy > 1e-15:
            W.append((s_step, y_step, 1.0 / sy))
            if len(W) > m:
                W.pop(0)
        
        pts = new_pts
    
    return best_pts, best_energy


def _conjugate_gradient(pts0, max_iter=10000, time_limit=None, start_time=None):
    """Conjugate gradient method on the sphere."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    pts = _project_to_sphere(pts0.copy())
    energy, grad = _log_energy_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        
        # Line search
        step = 1.0
        found = False
        for ls in range(30):
            new_pts = _project_to_sphere(pts + step * d)
            new_energy = _log_energy(new_pts)
            if new_energy <= energy + 1e-4 * step * np.dot(grad, d):
                found = True
                break
            step *= 0.5
        
        if not found:
            new_pts = _project_to_sphere(pts + 1e-6 * d)
            new_energy = _log_energy(new_pts)
            if new_energy >= energy:
                break
        
        new_energy2, new_grad = _log_energy_grad(new_pts)
        
        if new_energy2 < best_energy:
            best_energy = new_energy2
            best_pts = new_pts.copy()
        
        # Fletcher-Reeves update
        beta = np.dot(new_grad, new_grad) / max(np.dot(grad, grad), 1e-15)
        d = -new_grad + beta * d
        
        pts = new_pts
        energy = new_energy2
        grad = new_grad
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s safety margin
    
    # Start with golden spiral
    pts0 = _golden_spiral(N)
    
    # Phase 1: Conjugate gradient for rapid initial convergence
    pts1, e1 = _conjugate_gradient(pts0, max_iter=5000, time_limit=time_limit - 60, start_time=start_time)
    
    if time.time() - start_time > time_limit - 40:
        return _project_to_sphere(pts1)
    
    # Phase 2: L-BFGS for finer convergence
    pts2, e2 = _lbfgs_optimize(pts1, max_iter=5000, time_limit=time_limit - 10, start_time=start_time)
    
    if time.time() - start_time > time_limit - 5:
        return _project_to_sphere(pts2)
    
    # Phase 3: Final polish with more CG iterations
    pts3, e3 = _conjugate_gradient(pts2, max_iter=10000, time_limit=time_limit - 2, start_time=start_time)
    
    if e3 < e2:
        pts_final = pts3
    else:
        pts_final = pts2
    
    return _project_to_sphere(pts_final)
# EVOLVE-BLOCK-END
