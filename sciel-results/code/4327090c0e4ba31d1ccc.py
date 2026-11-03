"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fast_energy(pts):
    """Vectorized energy computation for optimization (less precise but faster)."""
    n = pts.shape[0]
    # Compute pairwise squared distances
    sq_dists = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
    sq_dists = np.maximum(sq_dists, 0)
    # Only upper triangle, exclude diagonal
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(sq_dists[iu])
    # Clip to avoid log(0)
    dists = np.maximum(dists, 1e-12)
    return np.sum(np.log(1.0 / dists))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    return _normalize(pts)

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Uses golden-angle spiral as initial guess, then optimizes with L-BFGS-B.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good starting point.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on unit sphere
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Optimization using L-BFGS-B on the sphere (parameterized by angles or direct coordinates)
    # We'll optimize in 3D and project to sphere at each step
    
    def objective(x):
        pts_candidate = _project_to_sphere(x.reshape(N, 3))
        return _fast_energy(pts_candidate)
    
    # Gradient computation (numerical or analytical)
    # For efficiency, we'll use numerical gradient via finite differences initially
    # but L-BFGS-B can use numerical gradients
    
    try:
        # Optimize with L-BFGS-B
        result = minimize(
            objective,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result.success or result.fun < best_energy:
            candidate_pts = _normalize(result.x.reshape(N, 3))
            candidate_energy = _log_energy(candidate_pts)
            if candidate_energy < best_energy and np.all(np.isfinite(candidate_pts)):
                best_pts = candidate_pts
                best_energy = candidate_energy
        
        # Check if we have time for a second refinement pass
        if time.time() < deadline - 30:
            # Use the optimized points as new starting point for a finer optimization
            result2 = minimize(
                objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
            )
            
            if result2.success or result2.fun < best_energy:
                candidate_pts2 = _normalize(result2.x.reshape(N, 3))
                candidate_energy2 = _log_energy(candidate_pts2)
                if candidate_energy2 < best_energy and np.all(np.isfinite(candidate_pts2)):
                    best_pts = candidate_pts2
                    best_energy = candidate_energy2
                    
    except Exception as e:
        # If optimization fails, return the initial golden spiral
        pass
    
    # Final validation
    best_pts = _normalize(best_pts)
    
    # Check for validity
    if not np.all(np.isfinite(best_pts)):
        # Fallback to golden spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
