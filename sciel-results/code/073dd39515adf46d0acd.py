"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # Use broadcasting: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clip to avoid log of very small numbers (shouldn't happen but safety)
    d = np.maximum(d, 1e-15)
    return math.fsum([math.log(1.0 / di) for di in d])


def _log_energy_fast(pts):
    """Faster computation using vectorized log and fsum."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _energy_and_grad(pts):
    """Compute energy and gradient of logarithmic energy on sphere.
    
    Energy E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    Gradient on sphere: for each point i, grad_i = sum_{j!=i} (xi - xj) / |xi-xj|^2
    (This is the gradient of -log|xi-xj| with respect to xi, projected to tangent space)
    
    Actually, let's be careful. E = sum_{i<j} log(1/d_ij) where d_ij = |xi-xj|.
    dE/dxi = sum_{j!=i} d/dxi [log(1/|xi-xj|)] = sum_{j!=i} d/dxi [-log|xi-xj|]
    d/dxi [-log|xi-xj|] = - (xi-xj)/|xi-xj|^2
    So grad_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    
    But we want to minimize E, so we do gradient descent: x_new = x - lr * grad_projected
    The projected gradient is (grad - (grad·x)x) since x is on unit sphere.
    """
    n = pts.shape[0]
    # Compute pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)  # (n,n)
    
    # Gradient: grad_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # diff[i,j] = xi - xj, dists_sq[i,j] = |xi-xj|^2
    # grad[i] = -sum_j diff[i,j] / dists_sq[i,j]
    # Avoid division by zero on diagonal (set to 0)
    inv_dists_sq = np.zeros_like(dists_sq)
    np.fill_diagonal(inv_dists_sq, 0.0)
    mask = dists_sq > 1e-30
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    # grad[i] = -sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)  # (n,3)
    
    # Project to tangent space: grad_tan = grad - (grad·x)x
    dot = np.sum(grad * pts, axis=1, keepdims=True)  # (n,1)
    grad_tan = grad - dot * pts  # (n,3)
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = math.fsum([math.log(1.0 / di) for di in d])
    
    return energy, grad_tan


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_gradient_descent(pts0, max_iters=50000, lr0=0.01, lr_decay=0.9999,
                                  kick_interval=500, kick_strength=0.1,
                                  tol=1e-10, deadline=None):
    """Riemannian gradient descent with adaptive step size and random kicks."""
    pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy_fast(pts)
    best_pts = pts.copy()
    
    lr = lr0
    n_stagnant = 0
    
    for it in range(max_iters):
        # Deadline check
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad_tan = _energy_and_grad(pts)
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            n_stagnant = 0
        else:
            n_stagnant += 1
        
        # Check convergence
        if abs(energy - best_energy) < tol and n_stagnant > 100:
            # Try a kick
            if n_stagnant % kick_interval == 0:
                # Random kick in tangent direction
                kick = np.random.randn(N, 3)
                kick -= np.sum(kick * pts, axis=1, keepdims=True) * pts
                kick_norms = np.linalg.norm(kick, axis=1, keepdims=True)
                kick_norms = np.maximum(kick_norms, 1e-15)
                kick = kick / kick_norms * kick_strength
                pts = _project_to_sphere(pts + kick)
                n_stagnant = 0
                lr = lr0 * 0.5
                continue
        
        # Gradient step
        pts_new = pts - lr * grad_tan
        pts_new = _project_to_sphere(pts_new)
        
        # Check if energy improved
        energy_new = _log_energy_fast(pts_new)
        if energy_new < energy:
            pts = pts_new
            # Increase learning rate slightly if improving
            lr = min(lr * 1.01, 0.1)
        else:
            # Decrease learning rate
            lr *= 0.9
            if lr < 1e-8:
                # Stuck, try a kick
                if n_stagnant > 100:
                    kick = np.random.randn(N, 3)
                    kick -= np.sum(kick * pts, axis=1, keepdims=True) * pts
                    kick_norms = np.linalg.norm(kick, axis=1, keepdims=True)
                    kick_norms = np.maximum(kick_norms, 1e-15)
                    kick = kick / kick_norms * kick_strength
                    pts = _project_to_sphere(pts + kick)
                    n_stagnant = 0
                    lr = lr0 * 0.5
        
        # Decay learning rate slowly
        lr *= lr_decay
    
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


def _halton_sequence(base, n):
    """Generate Halton sequence points."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        for j in range(3):
            pts[i, j] = halton(i + 1, base[j])
    # Map to sphere using inverse CDF or just normalize
    # Actually, let's use a different approach: use Halton for angles
    # theta = 2*pi*halton(i,2), phi = acos(2*halton(i,3)-1)
    pts = np.zeros((n, 3))
    for i in range(n):
        u1 = halton(i + 1, 2)
        u2 = halton(i + 1, 3)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        pts[i, 0] = np.sin(phi) * np.cos(theta)
        pts[i, 1] = np.sin(phi) * np.sin(theta)
        pts[i, 2] = np.cos(phi)
    return pts


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Multiple diverse initializations
    initializations = [
        _fibonacci_spiral(),
        _random_points(seed=42),
        _random_points(seed=123),
        _random_points(seed=999),
        _halton_sequence([2, 3, 5], N),
        _random_points(seed=777),
        _random_points(seed=314),
    ]
    
    best_pts = None
    best_energy = float('inf')
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30.0:
            # Not enough time for another full optimization
            break
        
        # Run Riemannian gradient descent
        pts, energy = _riemannian_gradient_descent(
            init_pts,
            max_iters=30000,
            lr0=0.005,
            lr_decay=0.99995,
            kick_interval=200,
            kick_strength=0.05,
            tol=1e-12,
            deadline=deadline - 5.0
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish: run a few more iterations with smaller step size
    if best_pts is not None and time.time() < deadline - 5.0:
        pts, energy = _riemannian_gradient_descent(
            best_pts,
            max_iters=10000,
            lr0=0.001,
            lr_decay=0.99999,
            kick_interval=500,
            kick_strength=0.01,
            tol=1e-14,
            deadline=deadline - 2.0
        )
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
