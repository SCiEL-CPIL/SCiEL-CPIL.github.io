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


def _angular_bisection_vertex(p_i, p_j, p_k):
    """
    Compute the Voronoi cell vertex (circumcenter) of the spherical triangle
    formed by points p_i, p_j, p_k using angular bisection planes.
    
    The circumcenter c satisfies:
      c · p_i = c · p_j = c · p_k = cos(R)
    
    This means c lies on the bisection planes:
      c · (p_i - p_j) = 0
      c · (p_i - p_k) = 0
    
    So c is parallel to (p_i - p_j) × (p_i - p_k).
    
    Returns the circumradius R = arccos(c · p_i) and the circumcenter c.
    """
    v1 = p_i - p_j
    v2 = p_i - p_k
    c = np.cross(v1, v2)
    norm_c = np.linalg.norm(c)
    
    if norm_c < 1e-14:
        # Degenerate case: points are nearly collinear on sphere
        # Return a large penalty radius
        return np.pi / 2.0, np.array([1.0, 0.0, 0.0])
    
    c = c / norm_c
    
    # Ensure c is on the correct side (same hemisphere as the triangle)
    # The circumcenter should be in the direction that minimizes the distance
    dot_i = np.dot(c, p_i)
    if dot_i < 0:
        c = -c
        dot_i = -dot_i
    
    # Clip for numerical stability
    dot_i = np.clip(dot_i, -1.0, 1.0)
    R = np.arccos(dot_i)
    
    return R, c


def _compute_voronoi_inradii(points):
    """
    Compute the Voronoi inradius for each point using angular bisection planes.
    
    For each point i, we find its nearest neighbors and compute the circumradius
    of each spherical triangle formed by i and pairs of its neighbors.
    The inradius of point i's Voronoi cell is the maximum of these circumradii.
    
    We use a soft-max approximation for differentiability.
    """
    n = len(points)
    
    # Compute pairwise angular distances
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    inradii = np.zeros(n)
    
    for i in range(n):
        # Get nearest neighbors (excluding self)
        # Use a reasonable number of neighbors to capture the Voronoi cell
        idx = np.argsort(dists[i])
        
        # Find the "Voronoi neighbors" - points that could form cell vertices
        # Typically, the nearest ~10-15 points are sufficient
        k_neighbors = min(15, n - 1)
        neighbors = idx[1:1+k_neighbors]
        
        # Compute circumradii for all triples (i, j, k) where j, k are neighbors
        circumradii = []
        
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                j = neighbors[a]
                k = neighbors[b]
                R, _ = _angular_bisection_vertex(points[i], points[j], points[k])
                circumradii.append(R)
        
        if len(circumradii) == 0:
            inradii[i] = 0.0
        else:
            circumradii = np.array(circumradii)
            # Use soft-max for differentiable max
            temp = 0.03
            inradii[i] = temp * np.log(np.sum(np.exp(circumradii / temp)))
    
    return inradii


def _objective(x):
    """
    Objective: sum of squared Voronoi inradii.
    This is differentiable and directly targets the covering radius.
    """
    points = _normalize(x.reshape(N, 3))
    inradii = _compute_voronoi_inradii(points)
    return np.sum(inradii ** 2)


def _covering_radius(points):
    """Compute the actual covering radius for evaluation."""
    points = _normalize(points)
    inradii = _compute_voronoi_inradii(points)
    return np.max(inradii) * 180.0 / np.pi


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization with angular bisection planes.
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
    
    def obj_wrapper(x):
        return _objective(x)
    
    # Phase 1: Coarse optimization with L-BFGS-B
    try:
        result = minimize(
            obj_wrapper,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() - start_time > max_time - 40:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 2: Local search with multiple restarts from different initializations
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    initial_configs = [best_x.copy()]
    
    # Add perturbed versions of Fibonacci
    for scale in [0.1, 0.2, 0.3]:
        x_perturbed = x0 + rng.randn(N * 3) * scale
        initial_configs.append(x_perturbed)
    
    # Add random configurations
    for _ in range(3):
        x_random = rng.randn(N * 3)
        x_random = _normalize(x_random.reshape(N, 3)).flatten()
        initial_configs.append(x_random)
    
    for config in initial_configs:
        if time.time() - start_time > max_time - 20:
            break
        
        try:
            result = minimize(
                obj_wrapper,
                config,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 4000, 'ftol': 1e-10}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    # Phase 3: Final fine-tuning
    if time.time() - start_time < max_time - 5:
        try:
            result = minimize(
                obj_wrapper,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-12}
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
