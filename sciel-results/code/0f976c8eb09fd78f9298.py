"""Rewritten program for minimal logarithmic energy of 100 points on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time
from scipy.optimize import minimize


def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3)."""
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    d = np.sqrt(d2)
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - M @ points
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(N, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _spherical_harmonic_basis(l_max):
    """Generate spherical harmonic basis vectors up to degree l_max."""
    from scipy.special import lpmv
    import cmath
    
    basis = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # We'll generate basis functions as vectors on the sphere
            # Actually, for optimization we need to represent points as coefficients
            # Let's use a different approach: parameterize points directly with better initialization
            pass
    
    # For now, let's use a more practical approach:
    # Better initialization + L-BFGS optimization
    return None


def _lbfgs_sphere(points, maxiter=5000, ftol=1e-12, gtol=1e-10):
    """Optimize points on sphere using L-BFGS-B with spherical constraint.
    
    We parameterize each point using (theta, phi) or use a projection method.
    Here we use the approach of optimizing in R^3 with projection to sphere.
    """
    points = _normalize(points)
    
    def objective_and_grad(flat):
        pts = flat.reshape(N, 3)
        # Energy
        d2 = 2.0 - 2.0 * (pts @ pts.T)
        np.clip(d2, 1e-30, 4.0, out=d2)
        d = np.sqrt(d2)
        iu = np.triu_indices(N, k=1)
        dists = d[iu]
        dists = np.maximum(dists, 1e-15)
        energy = fsum((-np.log(dists)).tolist())
        
        # Gradient
        M = 1.0 / d2
        np.fill_diagonal(M, 0.0)
        row_sums = M.sum(axis=1, keepdims=True)
        grad = row_sums * pts - M @ pts
        return energy, grad.flatten()
    
    # Use L-BFGS-B on the flat parameterization
    # We'll use a custom approach: optimize in tangent space
    result = minimize(
        lambda x: objective_and_grad(x)[0],
        points.flatten(),
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    opt_points = result.x.reshape(N, 3)
    opt_points = _normalize(opt_points)
    opt_energy = _log_energy(opt_points)
    
    return opt_points, opt_energy


def _gradient_descent_sphere(points, lr, n_iters, momentum=0.9):
    """Gradient descent on the sphere with momentum."""
    points = points.copy()
    points = _normalize(points)
    v = np.zeros_like(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iters):
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        
        # Project gradient to tangent space
        dots = (points * grad).sum(axis=1, keepdims=True)
        grad_tangent = grad - dots * points
        
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-12:
            break
        
        # Momentum update
        v = momentum * v - lr * grad_tangent / (grad_norm + 1e-15)
        points = points + v
        points = _normalize(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        # Decay learning rate
        lr = max(lr * 0.999, 1e-8)
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    time_budget = 110.0  # Leave 10s buffer
    
    # Phase 1: Good initial configuration
    points = _fibonacci_spiral(N)
    points = _normalize(points)
    
    # Phase 2: Coarse gradient descent
    points, e1 = _gradient_descent_sphere(points, lr=0.005, n_iters=300, momentum=0.9)
    
    # Phase 3: L-BFGS optimization
    points, e2 = _lbfgs_sphere(points, maxiter=3000, ftol=1e-11, gtol=1e-9)
    
    # Phase 4: Multi-restart strategy
    rng = np.random.default_rng(42)
    best_points = points.copy()
    best_energy = e2
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget * 0.7:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Perturb from best with varying noise
            noise_scale = 0.02 + 0.01 * restart
            noise = rng.standard_normal((N, 3)) * noise_scale
            pts = _normalize(best_points + noise)
        
        # Gradient descent from perturbed start
        pts, e = _gradient_descent_sphere(pts, lr=0.003, n_iters=200, momentum=0.9)
        
        # L-BFGS polish
        pts, e = _lbfgs_sphere(pts, maxiter=2000, ftol=1e-11, gtol=1e-9)
        
        if e < best_energy:
            best_energy = e
            best_points = pts.copy()
    
    points = best_points
    
    # Final polish if time allows
    if time.time() - start_time < time_budget * 0.85:
        points, e_final = _lbfgs_sphere(points, maxiter=3000, ftol=1e-12, gtol=1e-10)
        if e_final < best_energy:
            points = _normalize(points)
    
    # Ensure valid output
    points = _normalize(points)
    
    # Verify no points are too close
    d = _pairwise_distances(points)
    if d.min() < 1e-9:
        points = _fibonacci_spiral(N)
        points = _normalize(points)
    
    return points
# EVOLVE-BLOCK-END
