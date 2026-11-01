import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
M = 5000  # Number of test points for covering radius estimation
DEADLINE = 140.0  # Leave 10s margin for evaluation overhead


def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on a unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])


def _covering_radius(points, test_points):
    """Compute the covering radius in radians for a given configuration."""
    # points: (N, 3), test_points: (M, 3)
    # For each test point, find the minimum angular distance to any configuration point
    dots = test_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    angles = np.arccos(min_dots)
    return np.max(angles)


def _hull_facet_radii(points):
    """
    Compute the circumradius (in radians) of each facet of the convex hull.
    The largest facet circumradius gives an upper bound on the covering radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    
    radii = []
    for simplex in hull.simplices:
        pts = points[simplex]  # (3, 3) for triangular facets
        # For a triangle on the unit sphere, the circumradius is the max angle from centroid to vertices
        # Actually, for a spherical triangle, the covering radius is the max of the three vertex distances
        # But the true circumcenter might be inside the facet.
        # For a unit sphere, the facet is a spherical triangle.
        # The circumradius of the spherical triangle is the angular distance from the spherical circumcenter to each vertex.
        # A simpler approach: the maximum angular distance from the centroid of the facet (normalized) to each vertex.
        # This is a lower bound on the true covering radius contribution from this facet, but often close.
        # Better: for a flat triangle in 3D, the circumradius R_c satisfies:
        #   R_c = a*b*c / (4*Area)
        # Then the angular distance from the origin to the circumcenter is arccos(1 / sqrt(R_c^2 + 1))? No.
        # Actually, for points on the unit sphere, the circumcenter of the triangle (in 3D) lies at distance R_c from each vertex.
        # The angular distance from the origin to a point at distance R_c from the vertex (which is at distance 1) is:
        #   If the circumcenter is C, |C| = d, then |C - p| = R_c, |p| = 1.
        #   |C - p|^2 = |C|^2 + |p|^2 - 2|C||p|cos(theta) = d^2 + 1 - 2d cos(theta) = R_c^2
        #   cos(theta) = (d^2 + 1 - R_c^2) / (2d)
        #   But we don't know d directly.
        
        # Simpler: just use the maximum angular distance from the normalized centroid to each vertex.
        centroid = np.mean(pts, axis=0)
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-12:
            radii.append(0.0)
            continue
        centroid_unit = centroid / norm_c
        dots = pts @ centroid_unit
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        radii.append(np.max(angles))
    
    return np.array(radii)


def _optimize_repulsion(points, test_points, n_iters=200, learning_rate=0.01):
    """
    Iteratively refine points using repulsion forces.
    Each point is pushed away from other points, and also pulled toward reducing the covering radius.
    """
    points = points.copy()
    
    for it in range(n_iters):
        if time.time() > DEADLINE:
            break
        
        # Normalize points
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        points = points / norms
        
        # Compute gradient-like forces
        # For each pair (i, j), if they are close, push them apart
        dots = points @ points.T  # (N, N)
        dots = np.clip(dots, -1.0, 1.0)
        
        # Repulsion force: for each point i, sum over j != i of (points[i] - points[j]) / (1 - dots[i,j] + eps)
        # This approximates the gradient of the sum of log(1 - dots) or similar
        # Actually, let's use a simple approach: move each point in the direction of the gradient of the covering radius
        # But that's complex. Instead, use a heuristic: push points apart proportionally to their closeness.
        
        # Create a matrix of repulsion strengths
        repulsion = np.zeros_like(dots)
        # Only consider pairs that are relatively close (dot > 0.5, i.e., angle < 60 degrees)
        close_mask = dots > 0.5
        close_mask = close_mask & ~np.eye(N, dtype=bool)
        repulsion[close_mask] = 1.0 / (1.0 - dots[close_mask] + 1e-6)
        
        # Force on point i is the sum of (points[i] - points[j]) * repulsion[i,j] for all j
        # This can be computed as: (points @ repulsion) - points * np.sum(repulsion, axis=1, keepdims=True)
        # Wait, let's think: force_i = sum_j repulsion[i,j] * (points[i] - points[j])
        #                  = points[i] * sum_j repulsion[i,j] - sum_j repulsion[i,j] * points[j]
        #                  = points[i] * row_sum_i - (points.T @ repulsion[:, i])... no
        # sum_j repulsion[i,j] * points[j] = (points.T @ repulsion[i,:])... 
        # Actually, if we define R = repulsion, then force_i = points[i] * R[i].sum() - (R[i,:] @ points)
        # But R is symmetric, so R[i,:] @ points = (R @ points.T).T[i]
        
        row_sums = repulsion.sum(axis=1, keepdims=True)
        forces = points * row_sums - (repulsion @ points.T).T
        
        # Update points
        points = points + learning_rate * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        points = points / norms
    
    return points


def construct():
    start_time = time.time()
    
    # Precompute test points
    test_points = _fibonacci_sphere(M, offset=0.5)
    
    # Initial configuration: Fibonacci spiral
    best_points = _fibonacci_sphere(N, offset=0.5)
    best_radius = _covering_radius(best_points, test_points)
    
    # Multiple random starts
    n_starts = 5
    for s in range(n_starts):
        if time.time() > DEADLINE - 20:
            break
        
        if s == 0:
            points = best_points.copy()
        else:
            # Random perturbation of the Fibonacci spiral
            noise = np.random.normal(0, 0.5, size=(N, 3))
            points = best_points + noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            points = points / norms
        
        # Optimize using repulsion
        points = _optimize_repulsion(points, test_points, n_iters=100, learning_rate=0.02)
        
        # Compute covering radius
        radius = _covering_radius(points, test_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Final refinement: use a smaller learning rate for more iterations
    if time.time() < DEADLINE - 10:
        best_points = _optimize_repulsion(best_points, test_points, n_iters=200, learning_rate=0.005)
        final_radius = _covering_radius(best_points, test_points)
        if final_radius < best_radius:
            best_radius = final_radius
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms
    
    return best_points
