"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _covering_radius(points):
    """Compute the covering radius (in radians) of points on the unit sphere."""
    # Points are already on the unit sphere
    # Use convex hull to find facets
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use nearest neighbor max distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample many points on the sphere to estimate covering radius
        # But this is expensive; better to just return a large value or use a different method
        # For now, let's use a grid-based approach
        # Actually, let's just compute max over a dense set of test points
        # This is slow but works
        n_theta = 200
        n_phi = 400
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(thetas, phis, indexing='ij')
        test_pts = np.column_stack([
            np.sin(T) * np.cos(P).ravel(),
            np.sin(T) * np.sin(P).ravel(),
            np.cos(T).ravel()
        ])
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(test_pts)
        # Angular distance from chord distance: for unit sphere, chord = 2*sin(theta/2)
        # So theta = 2*arcsin(chord/2)
        max_chord = np.max(dists)
        max_angle = 2 * np.arcsin(min(max_chord / 2, 1.0))
        return max_angle
    
    # For each facet, compute the circumcenter and its angular distance to the facet vertices
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the facet vertices
        facet_pts = points[simplex]
        # Compute the circumcenter of the spherical triangle
        # For a triangle on the unit sphere, the circumcenter can be found by 
        # finding the point equidistant from all three vertices
        # Use the formula: circumcenter = (a/|a| + b/|b| + c/|c|) normalized? No, that's not right.
        
        # Better approach: The circumcenter of a spherical triangle is the point on the sphere
        # that is equidistant from all three vertices. It lies at the intersection of the 
        # perpendicular bisectors of the sides.
        
        # For a triangle with vertices A, B, C on the unit sphere, the circumcenter O satisfies
        # |O-A| = |O-B| = |O-C|, which means O·A = O·B = O·C (since |O|=|A|=|B|=|C|=1)
        # So O·(A-B) = 0 and O·(A-C) = 0
        # Thus O is parallel to (A-B) × (A-C)
        
        a, b, c = facet_pts
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-15:
            # Degenerate facet, skip
            continue
        o = cross / norm_cross
        
        # Check if o is on the correct side (should point outward from origin through the facet)
        # The facet normal should point outward. Check that o·a > 0 (since a is on the sphere)
        if np.dot(o, a) < 0:
            o = -o
        
        # The circumradius is the angular distance from o to any of the vertices
        # cos(angle) = o · a
        cos_angle = np.clip(np.dot(o, a), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # But we need to check that the circumcenter is actually the farthest point in the facet
        # from the configuration. The covering radius for this facet is the maximum angular
        # distance from any point in the facet to the nearest configuration point.
        # For a spherical triangle, the farthest point from the vertices is either the circumcenter
        # (if it's inside the triangle) or one of the vertices' opposite side midpoints.
        
        # A simpler and more robust approach: for each facet, the covering radius contribution
        # is the radius of the smallest spherical cap centered at one of the three vertices 
        # that covers the entire facet. Actually, no - it's the maximum over all points in the 
        # facet of the distance to the nearest vertex.
        
        # The maximum distance from a point in a spherical triangle to its vertices occurs 
        # at the circumcenter if the circumcenter is inside the triangle, otherwise at one 
        # of the vertices (but vertices have distance 0). Actually, it occurs at the point 
        # farthest from all three vertices, which for a triangle is either the circumcenter 
        # or a point on one of the edges.
        
        # For simplicity and correctness, let's just use the circumcenter distance as an 
        # upper bound, and also check edge midpoints.
        
        # Actually, the standard approach: the covering radius of the Delaunay triangulation 
        # on the sphere is the maximum circumradius over all facets. This is correct because 
        # the farthest point from the nearest site in any cell is the circumcenter of the 
        # corresponding Delaunay triangle (if it exists and is inside the cell).
        
        max_radius = max(max_radius, angle)
    
    return max_radius


def _refine(points, n_iterations=5000, lr_schedule='cosine'):
    """Refine points using gradient descent on the covering radius (via repulsive forces)."""
    points = points.copy()
    n = len(points)
    
    # We'll use a force-based approach: each point repels others with a force that 
    # increases the minimum pairwise distance, which helps improve the covering radius.
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Use simulated annealing-like approach with repulsive forces
    # Force on point i: sum over j != i of (1 - p_i · p_j) * p_i direction
    # Actually, let's use a potential energy approach:
    # E = -sum_{i<j} log(p_i · p_j)  (Riesz s=0, but this blows up)
    # Better: E = -sum_{i<j} (p_i · p_j)^k for some k, or use a repulsive kernel
    
    # Let's use a simple approach: maximize the minimum pairwise distance
    # by using a repulsive force proportional to 1/(1 - p_i·p_j)^2
    
    for it in range(n_iterations):
        # Compute pairwise dot products
        dots = points @ points.T
        np.fill_diagonal(dots, 0.0)
        
        # Clip to avoid numerical issues
        dots = np.clip(dots, -0.9999, 0.9999)
        
        # Repulsive force: for each pair, force is proportional to (1 - dot) in the direction
        # that increases the angular separation
        # Gradient of (p_i · p_j) w.r.t. p_i is p_j
        # We want to maximize the minimum distance, so we use a softmax-like weighting
        
        # Simple approach: each point moves away from its nearest neighbors
        # Force on i = sum_j w_ij * (p_i - (p_i·p_j)*p_j) where w_ij is large for close pairs
        
        # Use a soft repulsion: w_ij = 1 / (1 - dots[i,j])^2
        weights = 1.0 / (1.0 - dots) ** 2
        np.fill_diagonal(weights, 0.0)
        
        # Force on point i is sum_j w_ij * (p_i - (p_i·p_j) * p_j)
        # This is the component of p_i perpendicular to p_j, weighted by closeness
        # Actually, the gradient of the angular distance d_ij = arccos(p_i·p_j) w.r.t. p_i
        # is -p_j / sqrt(1 - (p_i·p_j)^2)
        # So to increase d_ij, we should move p_i in the direction of -p_j / sqrt(1-(p_i·p_j)^2)
        
        # Let's use: force_i = sum_{j!=i} w_ij * (p_i - (p_i·p_j)*p_j) / |p_i - (p_i·p_j)*p_j|
        # This normalizes the repulsive direction
        
        # Compute p_i - (p_i·p_j)*p_j for all pairs
        # This is points[:, :, None] - dots[:, :, None] * points[None, :, :]
        # Shape: (n, n, 3)
        
        # This might be memory intensive for n=130, but 130*130*3 = 50700 floats, fine.
        
        proj = dots[:, :, None] * points[None, :, :]  # (n, n, 3)
        perp = points[:, None, :] - proj  # (n, n, 3)
        
        # Normalize
        perp_norms = np.linalg.norm(perp, axis=2, keepdims=True)
        perp_norms = np.maximum(perp_norms, 1e-15)
        perp_unit = perp / perp_norms
        
        # Weighted sum
        force = np.sum(weights[:, :, None] * perp_unit, axis=1)  # (n, 3)
        
        # Normalize force
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-15)
        force_unit = force / force_norms
        
        # Learning rate with cooling
        if lr_schedule == 'cosine':
            lr = 0.01 * (1 + np.cos(np.pi * it / n_iterations)) / 2
        else:
            lr = 0.01 * (1 - it / n_iterations)
        
        # Update points
        points = points + lr * force_unit
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Check coverage every 100 iterations
        if it % 100 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using repulsive force simulation
    points, radius = _refine(points, n_iterations=3000)
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
