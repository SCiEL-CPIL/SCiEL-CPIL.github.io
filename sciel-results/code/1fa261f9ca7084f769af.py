import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _normalize(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _covering_radius_deg(points):
    """
    Compute the covering radius in degrees.
    This is the maximum angular distance from any point on the sphere to the nearest point in the set.
    This is equivalent to the maximum circumradius of the facets of the convex hull.
    For a set of points on the unit sphere, the covering radius is the maximum over all empty caps.
    We approximate this by checking the Voronoi cells or using the property that the farthest point
    from the set lies on the circumcircle of some triangle (or arc of a circle for 2 points).
    A robust way is to compute the convex hull and check the facets.
    However, a simpler and often sufficient approximation for optimization is to use a dense grid,
    but that's slow and inaccurate.
    
    A better approach: The covering radius is the maximum of the circumradii of the Delaunay triangles
    on the sphere. We can compute the spherical Delaunay triangulation.
    scipy.spatial.Delaunay works on 3D points. The facets of the convex hull of the points
    correspond to the Delaunay triangles.
    For each facet (triangle), we compute the circumradius.
    The circumradius of a triangle on the unit sphere is related to the circumcircle of the triangle.
    The center of the circumcircle is the point on the sphere farthest from the three vertices
    (within the spherical triangle).
    Actually, the covering radius is the maximum distance from any point on the sphere to its nearest neighbor.
    This maximum is attained at a vertex of the Voronoi diagram (which is the dual of the Delaunay triangulation).
    The vertices of the Voronoi diagram are the circumcenters of the Delaunay triangles.
    So we need to compute the circumcenter of each Delaunay triangle and find its distance to the nearest point.
    """
    from scipy.spatial import ConvexHull
    
    # Check for coincident points
    if len(np.unique(points, axis=0)) < N:
        return 180.0
        
    # Check if all points are in a closed hemisphere
    # If they are, the covering radius is at least 90 degrees
    # A simple check: if the convex hull has a facet with area > 2*pi (impossible)
    # Better: check if there is a plane through the origin that separates all points
    # This is equivalent to checking if the origin is inside the convex hull
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
        
    # If the origin is not inside the convex hull, the points are in a closed hemisphere
    # We can check this by seeing if the minimum distance from the origin to the convex hull is 0
    # Or simpler: if the sum of the outward normals of the facets (weighted by area) is 0, the origin is inside.
    # Actually, a simpler check: if the points are in a hemisphere, there exists a vector v such that v . p >= 0 for all p.
    # This is a linear programming problem, but we can approximate.
    # For now, let's assume they are not in a hemisphere if the convex hull is valid.
    
    # Get the facets (triangles)
    simplices = hull.simplices
    
    # For each triangle, compute the circumcenter
    # The circumcenter is the point equidistant from the three vertices.
    # On the unit sphere, the circumcenter is the normalized cross product of the edges?
    # No, the circumcenter is the point c such that |c - a| = |c - b| = |c - c|.
    # This is equivalent to c being in the plane defined by a, b, c? No.
    # The circumcenter lies on the line passing through the centroid and perpendicular to the plane of the triangle? No.
    # The circumcenter is the intersection of the perpendicular bisectors.
    # For points on the unit sphere, the circumcenter is the normalized vector:
    # c = (a x b) x (b x c) ... no.
    # The circumcenter is the normalized vector:
    # c = a * (sin(2*B)) + b * (sin(2*C)) + c * (sin(2*A)) ... this is for the centroid.
    
    # A more direct way: solve the linear system for the circumcenter.
    # Let the circumcenter be u.
    # u . a = u . b = u . c (since |u - a|^2 = 1 - 2 u . a + 1 = 2 - 2 u . a)
    # So u . (a - b) = 0 and u . (a - c) = 0.
    # Also |u| = 1.
    # So u is the unit vector perpendicular to the plane spanned by (a - b) and (a - c).
    # This is the normal to the plane of the triangle.
    # There are two such vectors: n and -n.
    # The circumcenter is the one that lies in the spherical triangle.
    # We can check which one has a larger dot product with the centroid.
    
    # Compute the normal
    # n = (b - a) x (c - a)
    # This normal points outward from the origin if the triangle is oriented correctly.
    # We need to be careful with orientation.
    
    # Let's compute the circumcenter for each triangle.
    max_radius = 0.0
    for tri in simplices:
        a = points[tri[0]]
        b = points[tri[1]]
        c = points[tri[2]]
        
        # Normal to the plane
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Check orientation
        # The centroid is (a + b + c) / 3
        centroid = (a + b + c) / 3.0
        # The circumcenter should be on the same side of the plane as the centroid?
        # Actually, the circumcenter is on the line through the origin in the direction of n or -n.
        # We want the one that is "inside" the spherical triangle.
        # This is the one that has a positive dot product with the centroid?
        # Let's check: if the triangle is small, the centroid is close to the circumcenter.
        # So we want n such that n . centroid > 0.
        if np.dot(n, centroid) < 0:
            n = -n
            
        # The circumcenter is n.
        # The distance from n to the nearest point is the radius of the cap.
        # But we need the distance to the nearest point in the ENTIRE set, not just the triangle.
        # However, for the covering radius, we only need to check the circumcenters of the Delaunay triangles.
        # The distance from the circumcenter to the nearest point is the distance to one of the vertices of the triangle.
        # Because the Voronoi vertex is the circumcenter, and its nearest neighbors are the vertices of the Delaunay triangle.
        
        # Compute the distance to the nearest vertex
        d2 = np.minimum.reduce([
            np.sum((n - a)**2),
            np.sum((n - b)**2),
            np.sum((n - c)**2)
        ])
        # d2 = 2 - 2 * cos(theta)
        # cos(theta) = 1 - d2/2
        cos_theta = 1.0 - d2 / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        radius_deg = np.degrees(theta)
        
        if radius_deg > max_radius:
            max_radius = radius_deg
            
    return max_radius

def _combined_loss(params, points_init):
    """
    Loss function for optimization.
    We want to minimize the covering radius.
    We also add a repulsion term to avoid coincident points.
    """
    x = _normalize(params.reshape(N, 3))
    
    # Repulsion energy: sum 1/d_ij
    d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
    np.fill_diagonal(d2, 1e-10)
    d = np.sqrt(d2)
    repulsion = np.sum(1.0 / d) - N
    
    # Covering radius
    # We use a smooth approximation: log-sum-exp of the distances
    # But for the loss, we can just use the covering radius itself, which is non-smooth.
    # To make it smooth, we can use a soft-max.
    # However, the covering radius is already a max, so it's non-smooth.
    # We can use a penalty method or just use the covering radius as the loss.
    # Since we are using L-BFGS, which can handle non-smooth functions to some extent,
    # we can just use the covering radius.
    # But it's better to use a smooth surrogate.
    
    # Let's use the soft-max of the distances to the nearest point.
    # For each point on a grid, compute the distance to the nearest point.
    # Then take the soft-max.
    # This is expensive.
    
    # Alternative: use the covering radius directly.
    cover = _covering_radius_deg(x)
    
    return cover + 0.1 * repulsion

def _fast_local_search(x):
    """
    Local search to refine the configuration.
    """
    for _ in range(50):
        # For each point, try to move it to a better position
        # We can use a gradient step on the covering radius.
        # But the covering radius is non-smooth.
        # Let's use a simple repulsion-based local search.
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        
        # Gradient of repulsion
        inv_d3 = 1.0 / (d2 * d)
        np.fill_diagonal(inv_d3, 0.0)
        grad = - (x * np.sum(inv_d3, axis=1)[:, None]) + (inv_d3 @ x)
        
        # Project to tangent space
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - dot * x
        
        # Update
        x = x - 0.01 * grad_tangent
        x = _normalize(x)
        
    return x

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    x = _fibonacci_sphere(N)
    
    # 2. Optimization
    # Use L-BFGS-B to minimize the combined loss
    # We need to handle the constraint that points are on the unit sphere.
    # We can parameterize the points using spherical coordinates, but that has singularities.
    # Alternatively, we can use the fact that the points are on the unit sphere and use a Riemannian optimizer.
    # Since we don't have a Riemannian optimizer, we can use a penalty method or just project after each step.
    
    # Let's use a simple gradient descent with projection.
    # We will use the repulsion energy as the main objective, and add a covering radius term.
    
    # First, run a few iterations of gradient descent on the repulsion energy.
    for it in range(100):
        if time.time() > deadline:
            break
            
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        
        # Gradient of repulsion
        inv_d3 = 1.0 / (d2 * d)
        np.fill_diagonal(inv_d3, 0.0)
        grad = - (x * np.sum(inv_d3, axis=1)[:, None]) + (inv_d3 @ x)
        
        # Project to tangent space
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - dot * x
        
        # Update
        x = x - 0.01 * grad_tangent
        x = _normalize(x)
        
    # 3. Local search
    x = _fast_local_search(x)
    
    # 4. Try to improve by moving points that are too close
    # We can use a simple heuristic: if two points are too close, move one of them.
    for _ in range(20):
        if time.time() > deadline:
            break
            
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        
        # Find the closest pair
        min_d = np.min(d)
        if min_d < 0.1:  # If the closest pair is too close
            # Find the indices
            min_idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = min_idx
            if i != j:
                # Move point i in a random direction
                # We can move it away from point j
                direction = x[i] - x[j]
                direction = _normalize(direction)
                x[i] = _normalize(x[i] + 0.1 * direction)
                
    return x

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    x = _fibonacci_sphere(N)
    
    # 2. Optimization
    # Use L-BFGS-B to minimize the combined loss
    # We need to handle the constraint that points are on the unit sphere.
    # We can parameterize the points using spherical coordinates, but that has singularities.
    # Alternatively, we can use the fact that the points are on the unit sphere and use a Riemannian optimizer.
    # Since we don't have a Riemannian optimizer, we can use a penalty method or just project after each step.
    
    # Let's use a simple gradient descent with projection.
    # We will use the repulsion energy as the main objective, and add a covering radius term.
    
    # First, run a few iterations of gradient descent on the repulsion energy.
    for it in range(100):
        if time.time() > deadline:
            break
            
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        
        # Gradient of repulsion
        inv_d3 = 1.0 / (d2 * d)
        np.fill_diagonal(inv_d3, 0.0)
        grad = - (x * np.sum(inv_d3, axis=1)[:, None]) + (inv_d3 @ x)
        
        # Project to tangent space
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - dot * x
        
        # Update
        x = x - 0.01 * grad_tangent
        x = _normalize(x)
        
    # 3. Local search
    x = _fast_local_search(x)
    
    # 4. Try to improve by moving points that are too close
    # We can use a simple heuristic: if two points are too close, move one of them.
    for _ in range(20):
        if time.time() > deadline:
            break
            
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        
        # Find the closest pair
        min_d = np.min(d)
        if min_d < 0.1:  # If the closest pair is too close
            # Find the indices
            min_idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = min_idx
            if i != j:
                # Move point i in a random direction
                # We can move it away from point j
                direction = x[i] - x[j]
                direction = _normalize(direction)
                x[i] = _normalize(x[i] + 0.1 * direction)
                
    return x
# EVOLVE-BLOCK-END
