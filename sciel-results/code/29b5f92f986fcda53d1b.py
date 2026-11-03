"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.optimize as so
import scipy.stats as st

_START = time.time()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and gradient on the sphere (tangent space).
    
    Returns:
        energy: float
        grad: np.ndarray of shape (N, 3), projected to tangent space
    """
    n = pts.shape[0]
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)  # (n, n)
    
    # Energy: sum_{i<j} log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    # Fast energy: use numpy log in batches, then fsum for precision
    log_terms = np.log(1.0 / d)
    # Split into chunks for fsum precision
    chunk_size = 500
    energy_parts = []
    for start in range(0, len(log_terms), chunk_size):
        chunk = log_terms[start:start+chunk_size]
        energy_parts.append(math.fsum([float(x) for x in chunk]))
    energy = math.fsum(energy_parts)
    
    # Gradient of -log|xi-xj| wrt xi is -(xi-xj)/|xi-xj|^2
    # Sum over j != i
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-30
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space: grad_tan = grad - (grad·x)x
    dot = np.sum(grad * pts, axis=1, keepdims=True)  # (n, 1)
    grad_tan = grad - dot * pts  # (n, 3)
    
    return energy, grad_tan


def _penalized_energy_and_grad(pts_flat, penalty_weight=1000.0):
    """Compute energy + penalty for sphere constraint, and gradient.
    
    Args:
        pts_flat: np.ndarray of shape (N*3,)
        penalty_weight: float
    
    Returns:
        val: float
        grad_flat: np.ndarray of shape (N*3,)
    """
    pts = pts_flat.reshape(-1, 3)
    
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    log_terms = np.log(1.0 / d)
    chunk_size = 500
    energy_parts = []
    for start in range(0, len(log_terms), chunk_size):
        chunk = log_terms[start:start+chunk_size]
        energy_parts.append(math.fsum([float(x) for x in chunk]))
    energy = math.fsum(energy_parts)
    
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-30
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    raw_grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Penalty term: sum_i (|xi|^2 - 1)^2
    norms_sq = np.sum(pts * pts, axis=1)
    penalty = penalty_weight * np.sum((norms_sq - 1.0) ** 2)
    
    # Gradient of penalty: 4 * penalty_weight * (|xi|^2 - 1) * xi
    pen_grad = 4.0 * penalty_weight * (norms_sq - 1.0)[:, np.newaxis] * pts
    
    total_grad = raw_grad + pen_grad
    total_val = energy + penalty
    
    return total_val, total_grad.flatten()


def _riemannian_gradient_descent(pts, max_iter=5000, tol=1e-10, lr_init=0.01):
    """Riemannian gradient descent with backtracking line search on the sphere.
    
    Args:
        pts: initial points on unit sphere, shape (N, 3)
        max_iter: maximum iterations
        tol: convergence tolerance on gradient norm
        lr_init: initial learning rate
    
    Returns:
        pts: optimized points on unit sphere
        energy: final energy
    """
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    
    best_energy, _ = _log_energy_and_grad(pts)
    best_pts = pts.copy()
    
    lr = lr_init
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        energy, grad_tan = _log_energy_and_grad(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = lr
        best_alpha = None
        for _ in range(30):
            # Riemannian exponential map: exp_x(v) = x*cos(|v|) + (v/|v|)*sin(|v|)
            v = -alpha * grad_tan
            v_norm = np.linalg.norm(v, axis=1, keepdims=True)
            v_norm = np.maximum(v_norm, 1e-15)
            # For small v, use first-order approximation
            mask_large = v_norm > 1e-4
            new_pts = np.zeros_like(pts)
            # Small v: x + v (first order)
            new_pts[~mask_large[:, 0]] = pts[~mask_large[:, 0]] + v[~mask_large[:, 0]]
            # Large v: use exponential map
            if np.any(mask_large):
                v_dir = v[mask_large] / v_norm[mask_large]
                cos_v = np.cos(v_norm[mask_large])
                sin_v = np.sin(v_norm[mask_large])
                new_pts[mask_large] = pts[mask_large] * cos_v + v_dir * sin_v
            
            new_pts = _project_to_sphere(new_pts)
            
            new_energy, _ = _log_energy_and_grad(new_pts)
            
            if new_energy < energy - 1e-4 * alpha * grad_norm:
                best_alpha = alpha
                pts = new_pts
                energy = new_energy
                break
            alpha *= 0.5
        
        if best_alpha is None:
            # No improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-12:
                break
        else:
            lr = min(lr * 1.1, 0.1)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
    
    return best_pts, best_energy


def _fibonacci_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    """Random points on the unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _sobol_points(seed=None):
    """Sobol sequence points on the unit sphere."""
    try:
        sampler = st.qmc.Sobol(d=3, seed=seed)
        samples = sampler.random(N)
        # Map [0,1]^3 to sphere
        # Use inverse CDF for uniform distribution on sphere
        # u1, u2 in [0,1]
        # theta = 2*pi*u1
        # phi = acos(2*u2 - 1)
        u1 = samples[:, 0]
        u2 = samples[:, 1]
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        pts = np.zeros((N, 3))
        pts[:, 0] = np.sin(phi) * np.cos(theta)
        pts[:, 1] = np.sin(phi) * np.sin(theta)
        pts[:, 2] = np.cos(phi)
        return pts
    except Exception:
        # Fallback to random if Sobol fails
        return _random_points(seed)


def _optimize_l_bfgsb(init_pts, penalty_weight=1000.0, maxiter=2000):
    """Optimize using L-BFGS-B with penalty for sphere constraint."""
    x0 = init_pts.flatten()
    
    # Check deadline
    if _remaining() < 10.0:
        return init_pts, _log_energy_and_grad(init_pts)[0]
    
    try:
        res = so.minimize(
            _penalized_energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
    except Exception:
        return init_pts, _log_energy_and_grad(init_pts)[0]
    
    pts_opt = res.x.reshape(-1, 3)
    pts_opt = _project_to_sphere(pts_opt)
    
    # Calculate true energy
    energy, _ = _log_energy_and_grad(pts_opt)
    
    return pts_opt, energy


def _halton_sequence(dim, count, skip=1):
    """Generate Halton sequence points in [0,1)^dim."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(skip, skip + count):
        point = [halton(i, base) for base in range(2, dim + 2)]
        points.append(point)
    return np.array(points)


def _sobol_to_sphere(samples):
    """Convert Sobol samples to sphere points."""
    u1 = samples[:, 0]
    u2 = samples[:, 1]
    theta = 2.0 * np.pi * u1
    phi = np.arccos(2.0 * u2 - 1.0)
    pts = np.zeros((len(samples), 3))
    pts[:, 0] = np.sin(phi) * np.cos(theta)
    pts[:, 1] = np.sin(phi) * np.sin(theta)
    pts[:, 2] = np.cos(phi)
    return pts


def _halton_to_sphere(halton_samples):
    """Convert Halton samples to sphere points."""
    u1 = halton_samples[:, 0]
    u2 = halton_samples[:, 1]
    theta = 2.0 * np.pi * u1
    phi = np.arccos(2.0 * u2 - 1.0)
    pts = np.zeros((len(halton_samples), 3))
    pts[:, 0] = np.sin(phi) * np.cos(theta)
    pts[:, 1] = np.sin(phi) * np.sin(theta)
    pts[:, 2] = np.cos(phi)
    return pts


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    # Multiple diverse initializations
    initializations = [
        _fibonacci_spiral(),
        _random_points(seed=42),
        _random_points(seed=123),
        _random_points(seed=999),
        _sobol_points(seed=1),
        _sobol_points(seed=2),
        _random_points(seed=777),
        _random_points(seed=314),
        _random_points(seed=271),
        _random_points(seed=628),
    ]
    
    # Add Halton-based initializations
    try:
        halton_samples = _halton_sequence(3, N, skip=1)
        initializations.append(_halton_to_sphere(halton_samples))
    except Exception:
        pass
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: L-BFGS-B optimization for each initialization
    for init_idx, init_pts in enumerate(initializations):
        if _remaining() < 20.0:
            break
        
        pts, energy = _optimize_l_bfgsb(init_pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Riemannian gradient descent polish on best result
    if best_pts is not None and _remaining() > 30.0:
        pts, energy = _riemannian_gradient_descent(best_pts, max_iter=3000, tol=1e-11, lr_init=0.005)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and _remaining() > 15.0:
        pts, energy = _optimize_l_bfgsb(best_pts, penalty_weight=10000.0, maxiter=500)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
        
        # Another Riemannian GD pass with very small learning rate
        if _remaining() > 10.0:
            pts, energy = _riemannian_gradient_descent(best_pts, max_iter=2000, tol=1e-12, lr_init=0.001)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_spiral()
    
    # Normalize to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral()
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Points too close, use Fibonacci as fallback
        best_pts = _fibonacci_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
