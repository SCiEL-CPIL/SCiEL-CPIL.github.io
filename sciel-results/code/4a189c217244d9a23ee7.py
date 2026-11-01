"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    (This is the gradient of -sum log(||x_i - x_j||), which equals the energy gradient)
    """
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    # Compute all pairwise distances
    # For efficiency with N=150, we can use vectorized operations
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # We only want i < j, so take upper triangle
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    energy = -math.fsum(math.log(d) for d in dists)
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: d/dx_i [-log(||x_i - x_j||)] = (x_i - x_j) / ||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _halton_init(n, dims=3):
    """Generate n points using the Halton sequence in 3D, then project to sphere."""
    def halton_sequence(n, base):
        seq = []
        f = 1.0 / base
        result = 0.0
        for i in range(1, n + 1):
            result = 0.0
            f = 1.0 / base
            num = i
            while num > 0:
                num, rem = divmod(num, base)
                result += rem * f
                f /= base
            seq.append(result)
        return np.array(seq)
    
    # Use first 3 primes for 3D
    bases = [2, 3, 5]
    h1 = halton_sequence(n, bases[0])
    h2 = halton_sequence(n, bases[1])
    h3 = halton_sequence(n, bases[2])
    
    # Map from [0,1)^3 to sphere using a mapping that gives good dispersion
    # Use the standard approach: convert to spherical coordinates
    u = h1  # [0, 1) for azimuth
    v = h2  # [0, 1) for polar angle
    w = h3  # unused or for refinement
    
    theta = 2.0 * np.pi * u
    phi = np.arccos(1.0 - 2.0 * v)  # uniform distribution on sphere
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    pts = np.column_stack([x, y, z])
    return pts


def _riemannian_cg(pts, max_iter=200, tol=1e-10):
    """Riemannian conjugate gradient on the sphere using exponential map.
    
    The Riemannian gradient is the projection of the Euclidean gradient
    onto the tangent space: grad_R = grad_E - <grad_E, x> * x
    """
    n = len(pts)
    x = pts.copy()
    
    energy, grad_e = _energy_and_grad(x)
    
    # Project gradient to tangent space
    inner = np.sum(grad_e * x, axis=1)
    grad_r = grad_e - inner[:, np.newaxis] * x
    
    p = -grad_r.copy()  # initial direction
    grad_r_prev = grad_r.copy()
    
    for it in range(max_iter):
        # Check convergence
        grad_norm = np.linalg.norm(grad_r)
        if grad_norm < tol:
            break
        
        # Line search: find t that minimizes energy along geodesic
        # Geodesic: exp_x(t * p) = x * cos(t*||p||) + (p/||p||) * sin(t*||p||)
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        p_norm[p_norm < 1e-12] = 1e-12
        p_unit = p / p_norm
        
        # Try several step sizes
        best_t = None
        best_energy = energy
        for t in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            x_trial = x * np.cos(t * p_norm) + p_unit * np.sin(t * p_norm)
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < best_energy:
                best_energy = e_trial
                best_t = t
        
        if best_t is None:
            # No improvement, try smaller steps
            for t in [0.001, 0.005, 0.02, 0.08]:
                x_trial = x * np.cos(t * p_norm) + p_unit * np.sin(t * p_norm)
                x_trial = _project_to_sphere(x_trial)
                e_trial, _ = _energy_and_grad(x_trial)
                if e_trial < best_energy:
                    best_energy = e_trial
                    best_t = t
        
        if best_t is None:
            break
        
        # Update position
        x = x * np.cos(best_t * p_norm) + p_unit * np.sin(best_t * p_norm)
        x = _project_to_sphere(x)
        energy = best_energy
        
        # Compute new Riemannian gradient
        _, grad_e_new = _energy_and_grad(x)
        inner_new = np.sum(grad_e_new * x, axis=1)
        grad_r_new = grad_e_new - inner_new[:, np.newaxis] * x
        
        # Conjugate gradient update (Polak-Ribiere)
        beta = np.sum(grad_r_new * (grad_r_new - grad_r_prev))
        beta = max(0.0, beta)
        p = -grad_r_new + beta * p
        
        # Project p to tangent space
        inner_p = np.sum(p * x, axis=1)
        p = p - inner_p[:, np.newaxis] * x
        
        grad_r_prev = grad_r_new
    
    return x, energy


def _projected_gradient(pts, max_iter=300, lr_init=0.01):
    """Projected gradient descent in Cartesian space with projection to sphere."""
    x = pts.copy()
    lr = lr_init
    
    for it in range(max_iter):
        energy, grad_e = _energy_and_grad(x)
        
        # Project gradient to tangent space
        inner = np.sum(grad_e * x, axis=1)
        grad_t = grad_e - inner[:, np.newaxis] * x
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-14:
            break
        
        # Try a few step sizes
        improved = False
        for scale in [1.0, 0.5, 0.25, 0.1]:
            t = lr * scale
            x_trial = x - t * grad_t
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < energy:
                x = x_trial
                energy = e_trial
                improved = True
                break
        
        if not improved:
            lr *= 0.5
            if lr < 1e-10:
                break
        else:
            lr = min(lr * 1.1, 0.1)
    
    return x, energy


def construct():
    # Initialize with Halton sequence for good global dispersion
    pts = _halton_init(N)
    pts = _project_to_sphere(pts)
    
    # Phase 1: Projected gradient descent in Cartesian space
    pts, e1 = _projected_gradient(pts, max_iter=200, lr_init=0.01)
    
    # Phase 2: Riemannian conjugate gradient on the sphere
    pts, e2 = _riemannian_cg(pts, max_iter=150, tol=1e-10)
    
    # Phase 3: Another round of projected gradient for fine-tuning
    pts, e3 = _projected_gradient(pts, max_iter=100, lr_init=0.005)
    
    # Phase 4: Final Riemannian CG polish
    pts, e4 = _riemannian_cg(pts, max_iter=100, tol=1e-11)
    
    # Ensure all points are on the unit sphere and valid
    pts = _project_to_sphere(pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                pts[j] += 1e-8 * np.random.randn(3)
                pts[j] = _project_to_sphere(pts[j])
    
    # Verify finite values
    if not np.all(np.isfinite(pts)):
        pts = _halton_init(N)
        pts = _project_to_sphere(pts)
    
    return pts
# EVOLVE-BLOCK-END
