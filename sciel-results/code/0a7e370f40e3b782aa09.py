"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad_flat(x):
    """
    Compute energy and gradient for flat vector x of shape (3N,).
    Energy is sum_{i<j} log(1/|xi - xj|).
    """
    pts = x.reshape(N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Avoid log(0)
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    energy = -math.fsum(np.log(dist_safe[iu]).tolist())
    
    # Gradient
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad_pts = -pts * row_sum[:, np.newaxis] + W @ pts
    
    # Penalty for sphere constraint
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    penalty = 1e8 * (norms - 1.0)**2
    penalty_val = np.sum(penalty)
    
    # Gradient of penalty
    grad_penalty = 2e8 * (norms - 1.0) * pts / norms
    
    total_grad = (grad_pts + grad_penalty).flatten()
    
    return energy + penalty_val, total_grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _optimize_from_start(x0, max_iter=10000):
    """
    Run L-BFGS-B optimization from a given starting point.
    """
    res = minimize(_energy_and_grad_flat, x0, jac=True, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    best_pts = _project_to_sphere(res.x.reshape(N, 3))
    # Re-calculate energy on the projected points for accurate comparison
    # Note: The optimizer minimizes energy + penalty, but we want pure energy on sphere
    # Since penalty is 0 on sphere, the value is close.
    # However, we should evaluate the true energy.
    energy, _ = _energy_and_grad_flat(best_pts.flatten())
    # Subtract penalty which is ~0
    # Actually, let's just use the returned energy from the objective function at the end
    # But _energy_and_grad_flat returns energy + penalty.
    # Let's compute true energy separately to be safe.
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    dist_safe = np.maximum(dist, 1e-15)
    iu = np.triu_indices(N, k=1)
    true_energy = -math.fsum(np.log(dist_safe[iu]).tolist())
    
    return best_pts, true_energy

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sobol_points(n=N):
    """Generate Sobol sequence points on the unit sphere."""
    try:
        sampler = qmc.Sobol(d=3, scramble=True)
        samples = sampler.random(n)
        # Map to sphere
        # Simple mapping: normalize
        pts = samples
        pts = _project_to_sphere(pts)
        return pts
    except:
        # Fallback to random if qmc not available or fails
        pts = np.random.normal(size=(n, 3))
        return _project_to_sphere(pts)

def construct():
    np.random.seed(42)
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Sobol points
    inits.append(_sobol_points(N))
    
    # 3. Random points
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    # 4. Perturbed Fibonacci
    noise = np.random.normal(0, 0.1, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # 5. Another random
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Run optimization from each start
    for x0 in inits:
        try:
            res_pts, res_energy = _optimize_from_start(x0.flatten(), max_iter=5000)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
            
    # Refine the best solution with a tighter tolerance
    try:
        res_pts, res_energy = _optimize_from_start(best_pts.flatten(), max_iter=10000)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = res_pts.copy()
    except Exception:
        pass
        
    # Final projection
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    # Check min distance
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
# EVOLVE-BLOCK-END
