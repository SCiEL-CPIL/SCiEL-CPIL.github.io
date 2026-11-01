"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    
    Energy: sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    
    # Compute all pairwise differences and squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy using math.fsum for accuracy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    energy = -math.fsum(math.log(d) for d in dists)
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _halton_init(n, offset=0):
    """Generate n points using the Halton sequence in 3D with an offset."""
    def halton_sequence(n, base, start=1):
        seq = []
        for i in range(start, start + n):
            result = 0.0
            f = 1.0 / base
            num = i
            while num > 0:
                num, rem = divmod(num, base)
                result += rem * f
                f /= base
            seq.append(result)
        return np.array(seq)
    
    bases = [2, 3, 5]
    h1 = halton_sequence(n, bases[0], start=1 + offset)
    h2 = halton_sequence(n, bases[1], start=1 + offset)
    h3 = halton_sequence(n, bases[2], start=1 + offset)
    
    u = h1
    v = h2
    
    theta = 2.0 * np.pi * u
    phi = np.arccos(1.0 - 2.0 * v)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    pts = np.column_stack([x, y, z])
    return pts


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points using the Fibonacci sphere distribution."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n)
    shifted = (indices + offset) % n
    
    z = 1.0 - 2.0 * (shifted + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    theta = golden_angle * shifted
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    pts = np.column_stack([x, y, z])
    return pts


def _tangent_gradient_step(pts, max_iter=300, lr_init=0.01, tol=1e-12):
    """Efficient tangent-space gradient descent with adaptive step size."""
    x = pts.copy()
    lr = lr_init
    n = len(x)
    
    energy, grad_e = _energy_and_grad(x)
    inner = np.sum(grad_e * x, axis=1)
    grad_t = grad_e - inner[:, np.newaxis] * x
    
    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < tol:
            break
        
        # Adaptive step size with backtracking
        best_t = None
        best_energy = energy
        
        for scale in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
            t = lr * scale
            x_trial = x - t * grad_t
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < best_energy:
                best_energy = e_trial
                best_t = t
        
        if best_t is None:
            lr *= 0.5
            if lr < 1e-14:
                break
        else:
            x = x - best_t * grad_t
            x = _project_to_sphere(x)
            energy = best_energy
            lr = min(lr * 1.02, 0.1)
            
            # Update gradient
            _, grad_e = _energy_and_grad(x)
            inner = np.sum(grad_e * x, axis=1)
            grad_t = grad_e - inner[:, np.newaxis] * x
    
    return x, energy


def _riemannian_cg(pts, max_iter=200, tol=1e-11):
    """Riemannian conjugate gradient on the sphere using exponential map."""
    x = pts.copy()
    n = len(x)
    
    energy, grad_e = _energy_and_grad(x)
    inner = np.sum(grad_e * x, axis=1)
    grad_r = grad_e - inner[:, np.newaxis] * x
    
    p = -grad_r.copy()
    grad_r_prev = grad_r.copy()
    
    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad_r)
        if grad_norm < tol:
            break
        
        # Exponential map line search
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        p_norm[p_norm < 1e-12] = 1e-12
        p_unit = p / p_norm
        
        best_t = None
        best_energy = energy
        for t in [0.5, 0.25, 0.1, 0.05, 0.02, 1.0, 2.0, 0.01]:
            x_trial = x * np.cos(t * p_norm) + p_unit * np.sin(t * p_norm)
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            if e_trial < best_energy:
                best_energy = e_trial
                best_t = t
        
        if best_t is None:
            break
        
        x = x * np.cos(best_t * p_norm) + p_unit * np.sin(best_t * p_norm)
        x = _project_to_sphere(x)
        energy = best_energy
        
        _, grad_e_new = _energy_and_grad(x)
        inner_new = np.sum(grad_e_new * x, axis=1)
        grad_r_new = grad_e_new - inner_new[:, np.newaxis] * x
        
        # Polak-Ribiere beta
        beta_num = np.sum(grad_r_new * (grad_r_new - grad_r_prev))
        beta_den = np.sum(grad_r_prev * grad_r_prev)
        beta = max(0.0, beta_num / beta_den) if beta_den > 1e-16 else 0.0
        p = -grad_r_new + beta * p
        
        # Project to tangent space
        inner_p = np.sum(p * x, axis=1)
        p = p - inner_p[:, np.newaxis] * x
        
        grad_r_prev = grad_r_new
    
    return x, energy


def construct():
    """Multi-start optimization with diverse initializations and two-phase optimization."""
    best_pts = None
    best_energy = float('inf')
    
    # Multiple diverse initializations
    initializations = []
    
    # Halton sequences with different offsets
    for offset in range(5):
        pts = _halton_init(N, offset=offset)
        pts = _project_to_sphere(pts)
        initializations.append(pts)
    
    # Fibonacci sphere with different offsets
    for offset in [0.0, 0.5, 1.0, 2.0]:
        pts = _fibonacci_sphere(N, offset=offset)
        pts = _project_to_sphere(pts)
        initializations.append(pts)
    
    # Optimize from each starting point
    for init_pts in initializations:
        try:
            # Phase 1: Tangent gradient descent for rough convergence
            pts, e = _tangent_gradient_step(init_pts, max_iter=200, lr_init=0.01)
            
            # Phase 2: Riemannian CG for curvature-aware refinement
            pts, e = _riemannian_cg(pts, max_iter=150, tol=1e-11)
            
            # Phase 3: Final tangent gradient polish
            pts, e = _tangent_gradient_step(pts, max_iter=100, lr_init=0.005)
            
            # Ensure validity
            pts = _project_to_sphere(pts)
            
            if not np.all(np.isfinite(pts)):
                continue
            
            # Check minimum distance
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d > 1e-9 and e < best_energy:
                best_energy = e
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback: use Halton offset 0
        pts = _halton_init(N, offset=0)
        pts = _project_to_sphere(pts)
        try:
            pts, _ = _tangent_gradient_step(pts, max_iter=100, lr_init=0.01)
            pts, _ = _riemannian_cg(pts, max_iter=80, tol=1e-10)
        except:
            pass
        best_pts = pts
    
    return best_pts
# EVOLVE-BLOCK-END
