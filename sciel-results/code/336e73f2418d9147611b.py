"""
Optimized 50-point spherical covering using differentiable Voronoi-based optimization.

Approach:
- Represent configuration as 50 points on the unit sphere.
- For each point, compute its Voronoi cell inradius (max distance to cell vertices).
- Cell vertices are circumcenters of Delaunay triangles (triples of neighbors).
- Minimize sum of squared inradii using gradient-based optimization.
- Use smooth approximations to handle non-differentiability at cell boundaries.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _circumradius_of_triple(p1, p2, p3):
    """
    Compute the circumradius of a spherical triangle with vertices p1, p2, p3.
    Returns the angular radius of the smallest cap containing all three points.
    
    The circumcenter c satisfies c·p1 = c·p2 = c·p3 = cos(R).
    c = normalize(cross(p1-p2, p1-p3))
    R = arccos(c·p1)
    """
    v1 = p1 - p2
    v2 = p1 - p3
    c = np.cross(v1, v2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        # Degenerate: return max pairwise distance / 2
        d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        d13 = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        d23 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        return max(d12, d13, d23) / 2.0
    c = c / norm_c
    cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
    return np.arccos(cos_r)


def _voronoi_inradii(points, k_neighbors=12):
    """
    Compute the inradius (covering radius contribution) for each point.
    
    For each point i, consider its k nearest neighbors. For each triple of neighbors
    (including i), compute the circumradius. The inradius of point i's Voronoi cell
    is the maximum of these circumradii over all relevant triples.
    
    We use a soft-max approximation for differentiability.
    """
    n = len(points)
    # Compute pairwise angular distances
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    inradii = np.zeros(n)
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        idx = np.argsort(dists[i])[:k_neighbors]
        
        # Collect circumradii for all triples involving i and pairs of neighbors
        circumradii = []
        neighbors = list(idx)
        
        # Triples: (i, j, k) for j, k in neighbors
        for a_idx in range(len(neighbors)):
            for b_idx in range(a_idx + 1, len(neighbors)):
                j = neighbors[a_idx]
                k = neighbors[b_idx]
                cr = _circumradius_of_triple(points[i], points[j], points[k])
                circumradii.append(cr)
        
        if len(circumradii) == 0:
            inradii[i] = 0.0
        else:
            circumradii = np.array(circumradii)
            # Use soft-max for differentiable max
            temp = 0.05
            inradii[i] = temp * np.log(np.sum(np.exp(circumradii / temp)))
    
    return inradii


def _objective(x):
    """
    Objective: sum of squared Voronoi inradii.
    This is differentiable and directly targets the covering radius.
    """
    points = _normalize(x.reshape(N, 3))
    inradii = _voronoi_inradii(points)
    return np.sum(inradii ** 2)


def _covering_radius(points):
    """Compute the actual covering radius for evaluation."""
    points = _normalize(points)
    inradii = _voronoi_inradii(points)
    return np.max(inradii) * 180.0 / np.pi


def _grad_finite_diff(x, eps=1e-5):
    """Finite difference gradient of the objective."""
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for j in range(len(x)):
        x_plus = x.copy()
        x_plus[j] += eps
        f_plus = _objective(x_plus)
        grad[j] = (f_plus - f0) / eps
    return grad


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization.
    """
    start_time = time.time()
    max_time = 110  # Leave margin for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = points.flatten().copy()
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization with L-BFGS-B
    def obj_wrapper(x):
        return _objective(x)
    
    try:
        result = minimize(
            obj_wrapper,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-10}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() - start_time > max_time - 30:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 2: Local search with multiple restarts
    rng = np.random.RandomState(42)
    for restart in range(8):
        # Create perturbation
        scale = 0.15 * (1.0 - restart / 10.0)
        x_perturbed = best_x + rng.randn(N * 3) * scale
        try:
            result = minimize(
                obj_wrapper,
                x_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
        
        if time.time() - start_time > max_time - 15:
            break
    
    # Phase 3: Final fine-tuning
    if time.time() - start_time < max_time - 5:
        try:
            result = minimize(
                obj_wrapper,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 15000, 'ftol': 1e-12}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    points = _normalize(best_x.reshape(N, 3))
    
    # Validate: ensure points are on unit sphere and not degenerate
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return points

# EVOLVE-BLOCK-END
