"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)"""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    return -np.sum(np.log(d))

def _log_energy_and_grad(pts):
    """Compute logarithmic energy and gradient (projected to tangent space)."""
    n = len(pts)
    # Compute all pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    dists = np.sqrt(dists_sq + 1e-30)  # (n, n)
    
    # Energy: -sum_{i<j} log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dx_i = -sum_{j≠i} (x_i - x_j) / d_ij^2
    # But we need to be careful: dE/dx_i = sum_{j≠i} (x_i - x_j) / d_ij^2
    # Wait, let me re-derive:
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} (1/d_ij) * d(d_ij)/dx_i + sum_{j<i} (1/d_ij) * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # So dE/dx_i = -sum_{j>i} (x_i - x_j)/d_ij^2 + sum_{j<i} (x_i - x_j)/d_ij^2
    # = -sum_{j≠i} (x_i - x_j)/d_ij^2  ... wait no
    # Actually: for j > i, term is -log(d_ij), derivative wrt x_i is -(x_i-x_j)/d_ij^2
    # For j < i, term is -log(d_ji) = -log(d_ij), derivative wrt x_i is -(x_i-x_j)/d_ij^2
    # So dE/dx_i = -sum_{j≠i} (x_i - x_j) / d_ij^2
    # But wait, that's not right either. Let me be more careful.
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} (1/d_ij) * (x_i - x_j)/d_ij - sum_{j<i} (1/d_ij) * (x_j - x_i)/d_ij
    # = -sum_{j>i} (x_i - x_j)/d_ij^2 + sum_{j<i} (x_i - x_j)/d_ij^2
    # Hmm, for j < i: d(d_ji)/dx_i = (x_i - x_j)/d_ji, so d(-log(d_ji))/dx_i = -(x_i-x_j)/d_ji^2
    # So dE/dx_i = -sum_{j≠i} (x_i - x_j)/d_ij^2
    # This is the full gradient in R^3.
    
    # Compute gradient: grad[i] = -sum_{j≠i} (x_i - x_j) / d_ij^2
    # = -sum_{j≠i} diff[i,j] / dists_sq[i,j]
    
    # Set diagonal to 0 to avoid division by zero
    dists_sq_safe = dists_sq.copy()
    np.fill_diagonal(dists_sq_safe, 1.0)
    
    # grad[i] = -sum_j diff[i,j] / dists_sq[i,j]
    inv_dists_sq = 1.0 / dists_sq_safe  # (n, n)
    # Zero out diagonal
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Project gradient onto tangent space
    # tangent_grad[i] = grad[i] - (grad[i] · x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    tangent_grad = grad - dot * pts
    
    return energy, tangent_grad

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _gradient_descent_sphere(pts, max_iters=5000, tol=1e-12, lr_init=0.01, deadline=None):
    """Gradient descent on the sphere with adaptive learning rate."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    # Initial energy
    energy, tgrad = _log_energy_and_grad(pts)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = lr_init
    min_lr = 1e-8
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        if np.max(np.abs(tgrad)) < tol:
            break
        
        # Adaptive step size: try to decrease energy
        # Use a simple backtracking line search
        step = lr
        accepted = False
        for _ in range(20):
            new_pts = pts + step * tgrad
            new_pts = _normalize(new_pts)
            
            new_energy, new_tgrad = _log_energy_and_grad(new_pts)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                tgrad = new_tgrad
                accepted = True
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = pts.copy()
                break
            else:
                step *= 0.5
        
        if not accepted:
            # Try a smaller step or break
            if step < min_lr:
                break
            # Try with even smaller step
            new_pts = pts + step * tgrad
            new_pts = _normalize(new_pts)
            new_energy, new_tgrad = _log_energy_and_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                tgrad = new_tgrad
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = pts.copy()
            else:
                break
        
        # Decay learning rate slightly
        lr *= 0.995
        if lr < min_lr:
            lr = min_lr
    
    return best_pts, best_energy

def _newton_polish(pts, max_iters=100, tol=1e-14, deadline=None):
    """Newton's method polish for high-precision convergence."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    energy, tgrad = _log_energy_and_grad(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.max(np.abs(tgrad)) < tol:
            break
        
        # For Newton's method on the sphere, we need the Hessian projected to tangent space
        # This is expensive, so let's use a quasi-Newton approach (L-BFGS style)
        # Or just do a few more gradient steps with very small step
        
        # Simple: use gradient with very small step
        step = 1e-6 / (1 + it * 0.1)
        new_pts = pts + step * tgrad
        new_pts = _normalize(new_pts)
        
        new_energy, new_tgrad = _log_energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            tgrad = new_tgrad
        else:
            break
    
    return pts, energy

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    v = rng.randn(n, 3)
    norms = np.sqrt(np.sum(v * v, axis=1, keepdims=True))
    return v / norms

def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_energy = np.inf
    
    # Try multiple restarts with different initializations
    # 1. Fibonacci spiral
    # 2. Random initializations with different seeds
    # 3. Perturbed Fibonacci spiral
    
    init_configs = []
    
    # Fibonacci spiral
    init_configs.append(_fibonacci_spiral(N))
    
    # Random initializations
    for seed in range(10):
        init_configs.append(_random_sphere(N, seed=seed))
    
    # Perturbed Fibonacci spiral
    rng = np.random.RandomState(42)
    fib = _fibonacci_spiral(N)
    for scale in [0.01, 0.05, 0.1]:
        perturbed = fib + scale * rng.randn(N, 3)
        init_configs.append(_normalize(perturbed))
    
    for idx, init in enumerate(init_configs):
        if time.time() > deadline:
            break
        
        # Check if we have enough time for this restart
        # Estimate: each restart takes roughly 5-20 seconds
        if time.time() + 5.0 > deadline:
            break
        
        # Gradient descent
        pts, energy = _gradient_descent_sphere(
            init, max_iters=3000, tol=1e-13, lr_init=0.01, deadline=deadline
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # If we have more time, do a polish
        if time.time() < deadline - 2.0:
            pts_polished, energy_polished = _newton_polish(
                pts, max_iters=50, tol=1e-15, deadline=deadline
            )
            if energy_polished < best_energy:
                best_energy = energy_polished
                best_pts = pts_polished.copy()
    
    # Final normalization
    if best_pts is not None:
        best_pts = _normalize(best_pts)
    else:
        # Fallback: return Fibonacci spiral
        best_pts = _normalize(_fibonacci_spiral(N))
    
    return best_pts
# EVOLVE-BLOCK-END
