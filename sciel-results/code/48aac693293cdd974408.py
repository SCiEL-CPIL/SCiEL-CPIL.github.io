"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_fast(pts):
    """Fast energy using dot-product distance computation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _energy_grad(x):
    """Compute energy and gradient for scipy optimizer.
    x is flattened (N*3,) vector of points on sphere.
    """
    pts = x.reshape(N, 3)
    pts = _normalize(pts)
    
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dists_sq, 1.0)
    dists_sq = np.maximum(dists_sq, 1e-30)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    # Gradient: grad_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-30
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)  # (n,3)
    
    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - dot * pts
    
    return energy, grad_tan.reshape(-1)


def _precise_energy(pts):
    """High-precision energy computation."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum([math.log(1.0 / di) for di in d])


def _fibonacci_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed):
    """Random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _normalize(pts)


def _sobol_sphere():
    """Sobol sequence points on the sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=42)
    samples = sampler.random(N)
    # Map to sphere: use first two dims for angles
    u1 = samples[:, 0]
    u2 = samples[:, 1]
    theta = 2.0 * np.pi * u1
    phi = np.arccos(2.0 * u2 - 1.0)
    pts = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return _normalize(pts)


def _optimize(pts0, deadline):
    """Optimize using L-BFGS-B with time limit."""
    x0 = pts0.reshape(-1)
    
    def fun_and_grad(x):
        if _remaining() < 5.0:
            raise Exception("timeout")
        e, g = _energy_grad(x)
        return e, g
    
    try:
        result = minimize(fun_and_grad, x0, method='L-BFGS-B',
                         options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 5000})
        pts = _normalize(result.x.reshape(N, 3))
        energy = _precise_energy(pts)
        return pts, energy
    except Exception:
        # If timeout or error, try gradient descent fallback
        return _gradient_descent_fallback(pts0, deadline)


def _gradient_descent_fallback(pts0, deadline):
    """Simple gradient descent as fallback."""
    pts = _normalize(pts0.copy())
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    lr = 0.005
    
    for it in range(20000):
        if _remaining() < 3.0:
            break
        
        energy, grad_flat = _energy_grad(pts.reshape(-1))
        grad_tan = grad_flat.reshape(N, 3)
        
        pts_new = _normalize(pts - lr * grad_tan)
        energy_new = _energy_fast(pts_new)
        
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.9
        
        if lr < 1e-10:
            break
    
    return best_pts, best_energy


def _anneal(pts0, deadline):
    """Simulated annealing to escape local minima."""
    pts = _normalize(pts0.copy())
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    n_accepted = 0
    n_total = 0
    
    for it in range(50000):
        if _remaining() < 5.0:
            break
        
        # Random perturbation
        rng = np.random.default_rng(it)
        kick = rng.standard_normal((N, 3))
        kick -= np.sum(kick * pts, axis=1, keepdims=True) * pts
        kick_norms = np.linalg.norm(kick, axis=1, keepdims=True)
        kick_norms = np.maximum(kick_norms, 1e-15)
        kick = kick / kick_norms * 0.05
        
        pts_new = _normalize(pts + kick)
        energy_new = _energy_fast(pts_new)
        energy_old = _energy_fast(pts)
        
        delta = energy_new - energy_old
        n_total += 1
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            pts = pts_new
            n_accepted += 1
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts_new.copy()
        
        T = max(T * cooling, T_min)
        
        # If temperature is too low, do a local optimization
        if T < 0.01 and it % 500 == 0:
            pts_opt, e_opt = _optimize(pts, _DEADLINE - 10.0)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
            pts = best_pts.copy()
    
    return best_pts, best_energy


def _valid(pts):
    """Check if points are valid."""
    if pts.shape != (N, 3):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        return False
    return True


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    # Multiple diverse initializations
    initializations = [
        _fibonacci_spiral(),
        _sobol_sphere(),
        _random_sphere(42),
        _random_sphere(123),
        _random_sphere(999),
        _random_sphere(777),
        _random_sphere(314),
        _random_sphere(271),
    ]
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize each initialization with L-BFGS-B
    for init_pts in initializations:
        if _remaining() < 15.0:
            break
        
        pts, energy = _optimize(init_pts, _DEADLINE)
        
        if _valid(pts) and energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Simulated annealing from best point
    if best_pts is not None and _remaining() > 20.0:
        pts, energy = _anneal(best_pts, _DEADLINE)
        if _valid(pts) and energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Final polish with high-precision optimization
    if best_pts is not None and _remaining() > 10.0:
        pts, energy = _optimize(best_pts, _DEADLINE)
        if _valid(pts) and energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Ensure valid output
    if best_pts is None or not _valid(best_pts):
        best_pts = _fibonacci_spiral()
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
