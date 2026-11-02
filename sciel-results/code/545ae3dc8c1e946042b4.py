"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull


def _exact_covering_radius(points):
    """Compute the exact covering radius using convex hull facets.
    
    The covering radius is the maximum circumradius of the spherical Voronoi cells,
    which equals the maximum over all facets of the convex hull of the distance
    from the origin to the facet plane (for facets not containing the origin).
    """
    try:
        hull = ConvexHull(points)
        # For each facet, compute the distance from origin to the plane
        # Facet normal is given by hull.equations[:, :3]
        # Distance from origin to plane: |c| / ||n|| where plane is n·x + c = 0
        equations = hull.equations
        normals = equations[:, :3]
        constants = equations[:, 3]
        
        # Distance from origin to each facet plane
        dists = np.abs(constants) / np.linalg.norm(normals, axis=1)
        
        # The covering radius is arccos(max distance from origin to facet)
        # because the farthest point on the sphere from all points is at distance
        # arccos(d) where d is the max facet distance
        max_dist = np.max(dists)
        max_dist = np.clip(max_dist, -1.0, 1.0)
        return np.arccos(max_dist)
    except Exception:
        # Fallback to sampling if convex hull fails
        return _sampled_covering_radius(points)


def _sampled_covering_radius(points, n_samples=50000):
    """Fallback: estimate covering radius by sampling."""
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    phi = np.arccos(np.random.uniform(-1, 1, n_samples))
    samples = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    dots = np.dot(samples, points.T)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    angles = np.arccos(min_dots)
    return np.max(angles)


def _fast_objective(params):
    """Fast objective: maximize minimum pairwise distance (proxy for covering radius)."""
    points = params.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Compute all pairwise dot products
    dots = np.dot(points, points.T)
    np.fill_diagonal(dots, 1.0)
    
    # For each point, find its nearest neighbor
    min_dots = np.min(dots, axis=1)  # minimum dot with any other point
    max_min_dot = np.max(min_dots)
    
    # We want to minimize the covering radius, which is related to maximizing
    # the minimum distance. Use negative to minimize.
    return -np.arccos(np.clip(max_min_dot, -1.0, 1.0))


def _refine_local(points):
    """Local refinement using gradient descent on the exact covering radius."""
    best_points = points.copy()
    best_radius = _exact_covering_radius(best_points)
    
    lr = 0.01
    for step in range(500):
        # Compute gradient by finite differences
        grad = np.zeros_like(points)
        eps = 1e-5
        
        for i in range(N):
            for j in range(3):
                p_plus = points.copy()
                p_plus[i, j] += eps
                # Project back to sphere
                p_plus[i] /= np.linalg.norm(p_plus[i])
                
                p_minus = points.copy()
                p_minus[i, j] -= eps
                p_minus[i] /= np.linalg.norm(p_minus[i])
                
                r_plus = _exact_covering_radius(p_plus)
                r_minus = _exact_covering_radius(p_minus)
                
                grad[i, j] = (r_plus - r_minus) / (2 * eps)
        
        # Update
        points_new = points - lr * grad
        # Project to sphere
        norms = np.linalg.norm(points_new, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        points_new = points_new / norms
        
        new_radius = _exact_covering_radius(points_new)
        if new_radius < best_radius:
            best_points = points_new.copy()
            best_radius = new_radius
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.7
        
        if lr < 1e-6:
            break
    
    return best_points, best_radius


def _initial_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    np.random.seed(42)
    
    # Stage 1: Use differential evolution with fast objective
    # Bounds for each coordinate: [-1, 1]
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    # Run differential evolution with fast objective
    de_result = differential_evolution(
        _fast_objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=20,
        polish=False
    )
    
    points_de = de_result.x.reshape(N, 3)
    norms = np.linalg.norm(points_de, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points_de = points_de / norms
    
    # Stage 2: Also start from Fibonacci and refine
    points_fib = _initial_fibonacci()
    
    # Stage 3: Local refinement from both starting points
    points1, radius1 = _refine_local(points_de)
    points2, radius2 = _refine_local(points_fib)
    
    # Choose the better one
    if radius1 < radius2:
        best_points = points1
    else:
        best_points = points2
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
