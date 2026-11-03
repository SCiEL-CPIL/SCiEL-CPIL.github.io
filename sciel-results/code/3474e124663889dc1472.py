"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius
    using a repulsive force simulation (Thomson problem analogue) followed by
    local optimization of the maximum distance to nearest neighbor.
    """
    rng = np.random.default_rng(42)
    
    # Start with a Fibonacci spiral for a good initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random perturbation to break symmetry
    points += rng.normal(scale=0.01, size=points.shape)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    def energy(points):
        """
        Compute the negative of the minimum pairwise distance as a proxy for
        maximizing the minimum distance (which correlates with minimizing covering radius).
        We actually want to maximize the minimum distance, so we minimize -min_dist.
        But for better coverage, we use a sum of inverse distances as a repulsive potential.
        """
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)  # avoid division by zero
        
        # Use a soft repulsive potential: sum of 1/d^k for k=2 (Coulomb-like)
        # This encourages uniform distribution
        repulsive = np.sum(1.0 / (dists ** 2))
        
        return repulsive
    
    # Run gradient-based optimization using L-BFGS-B with numerical gradients
    # We optimize in Cartesian coordinates and re-normalize after each step
    # This is tricky, so we use a different approach: optimize on the sphere using
    # tangent space projections
    
    # Alternative: use the fact that we can parameterize each point by two angles
    # But that's complex. Instead, let's use a simple iterative repulsion method.
    
    # Iterative repulsion: each point is pushed away from others by inverse-square force
    # projected onto the tangent plane of the sphere
    
    def project_to_tangent(v, p):
        """Project vector v onto the tangent plane at point p on unit sphere."""
        # Tangent plane is orthogonal to p
        return v - np.dot(v, p) * p
    
    # Run iterative relaxation
    for iteration in range(5000):
        # Compute forces on each point
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Coulomb repulsion: F = k / dist^2 * unit_vector
                force = diff / (dist ** 3)
                forces[i] += force
        
        # Update points: move in direction of force, projected to tangent plane
        step_size = 0.001
        for i in range(N):
            # Project force to tangent plane at points[i]
            tangent_force = project_to_tangent(forces[i], points[i])
            points[i] += step_size * tangent_force
            # Re-normalize to unit sphere
            points[i] /= np.linalg.norm(points[i])
        
        # Decay step size
        step_size *= 0.999
    
    # Final local optimization: minimize the covering radius directly
    # The covering radius is the maximum over all points y on the sphere of
    # the distance to the nearest configuration point.
    # This is equivalent to the maximum circumradius of the Delaunay triangulation
    # on the sphere.
    
    # For a practical approach, we can use the fact that the covering radius is
    # determined by the farthest point in each Voronoi cell, which is typically
    # at the center of a facet of the convex hull.
    
    # Let's use scipy's Delaunay to find the convex hull and compute circumradii
    from scipy.spatial import Delaunay
    
    # Compute the convex hull
    try:
        hull = Delaunay(points)
    except Exception:
        # If Delaunay fails, return the current configuration
        return points
    
    # For each tetrahedron in the hull (3D), the circumradius gives a lower bound
    # on the covering radius for that region.
    # Actually, for a sphere, the relevant facets are the spherical triangles.
    # The covering radius is the maximum of the circumradii of the spherical triangles
    # in the Voronoi diagram, which correspond to the facets of the convex hull.
    
    # Compute circumradius for each facet (triangle) of the convex hull
    # The circumradius of a spherical triangle is the angular radius of the
    # circumscribed circle, which equals the maximum angular distance from
    # the triangle's vertices to the center of the circumscribed circle.
    
    # For a planar triangle with vertices A, B, C, the circumradius is:
    # R = |A-B|*|B-C|*|C-A| / (4 * area)
    # But we need the spherical circumradius.
    
    # A simpler approach: for each facet (triangle) of the convex hull,
    # compute the circumcenter and the maximum angular distance to the vertices.
    
    # Let's compute the covering radius by sampling the sphere densely
    # and finding the maximum distance to the nearest point.
    
    # Create a dense grid on the sphere
    n_theta = 200
    n_phi = 400
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    sample_points = np.zeros((n_theta * n_phi, 3))
    sample_points[:, 0] = np.sin(THETA) * np.cos(PHI).flatten()
    sample_points[:, 1] = np.sin(THETA) * np.sin(PHI).flatten()
    sample_points[:, 2] = np.cos(THETA).flatten()
    
    # For each sample point, find the nearest configuration point
    # and compute the angular distance
    max_dist = 0.0
    for sp in sample_points:
        dists = np.arccos(np.clip(np.dot(points, sp), -1.0, 1.0))
        min_dist = np.min(dists)
        if min_dist > max_dist:
            max_dist = min_dist
    
    # Convert to degrees
    covering_radius_deg = np.degrees(max_dist)
    
    # Try to improve by moving points away from the worst regions
    # Find the sample point with maximum distance
    worst_sp = None
    worst_dist = 0.0
    for sp in sample_points:
        dists = np.arccos(np.clip(np.dot(points, sp), -1.0, 1.0))
        min_dist = np.min(dists)
        if min_dist > worst_dist:
            worst_dist = min_dist
            worst_sp = sp
    
    if worst_sp is not None:
        # Find the three nearest points to the worst point
        dists_to_worst = np.arccos(np.clip(np.dot(points, worst_sp), -1.0, 1.0))
        nearest_indices = np.argsort(dists_to_worst)[:3]
        
        # Move these points slightly away from the worst point
        for idx in nearest_indices:
            # Direction from worst_sp to points[idx]
            direction = points[idx] - worst_sp
            direction /= np.linalg.norm(direction)
            # Project to tangent plane at points[idx]
            tangent_dir = project_to_tangent(direction, points[idx])
            tangent_dir /= np.linalg.norm(tangent_dir)
            points[idx] += 0.005 * tangent_dir
            points[idx] /= np.linalg.norm(points[idx])
    
    return points
# EVOLVE-BLOCK-END
