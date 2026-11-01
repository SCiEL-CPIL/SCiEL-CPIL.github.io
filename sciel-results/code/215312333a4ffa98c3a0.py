"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(pts):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This is achieved at circumcenters of Delaunay triangles
    (facets of the convex hull) that have empty caps.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0  # fallback
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Each simplex is a triangle (3 points) on the convex hull
        P = pts[simplex]  # 3x3 array
        
        # Compute the normal of the plane containing the triangle
        v1 = P[1] - P[0]
        v2 = P[2] - P[0]
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        
        # For a triangle on the unit sphere, the circumcenter is the point on the sphere
        # that is equidistant from all 3 vertices. It lies in the direction of the normal
        # (or its negative).
        
        # The angular distance from n to each vertex should be the same if n is the circumcenter.
        # Let's check both n and -n.
        
        for sign in [1, -1]:
            c = sign * n
            
            # Compute angular distances from c to all points
            dots = np.clip(pts @ c, -1, 1)
            angles = np.arccos(dots)
            
            # The circumradius is the maximum angle from c to the 3 vertices
            circumradius = np.max(angles[simplex])
            
            # Check if c is a valid circumcenter: the cap centered at c with radius circumradius
            # should contain no other points (i.e., all other points should be at distance >= circumradius)
            # Actually, we need to check that no other point is CLOSER to c than the circumradius.
            # If another point is within circumradius of c, then c is not the farthest point in that Voronoi cell.
            
            # The minimum distance from c to any point should be >= circumradius for c to be a valid circumcenter
            # Actually, the 3 vertices are at distance circumradius, and all others should be >= circumradius.
            min_angle = np.min(angles)
            
            # If min_angle < circumradius - epsilon, then some point is inside the cap, so c is not valid
            # But wait, the 3 vertices ARE at distance circumradius, so min_angle should be <= circumradius.
            # We need to check that no OTHER point is strictly inside.
            
            # A better check: the number of points at distance < circumradius - epsilon should be 0
            # (excluding the 3 vertices which are at exactly circumradius)
            # Actually, due to floating point, let's check: are there points at distance < circumradius - 1e-6?
            # If yes, then c is not a valid circumcenter for the covering radius.
            
            # Simpler: just take the maximum circumradius over all facets. The covering radius
            # is the maximum over all valid circumcenters. But in practice, we can just take
            # the maximum circumradius of all facets, as invalid circumcenters (with non-empty caps)
            # will have smaller circumradii than the true covering radius.
            
            max_radius = max(max_radius, circumradius)
    
    return np.degrees(max_radius)


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _soft_covering_loss(pts, temperature=0.01):
    """Compute a differentiable soft approximation of the covering radius.
    
    Uses a soft-min over angular distances to create a smooth surrogate.
    """
    # Compute all pairwise angular distances
    dots = np.clip(pts @ pts.T, -1, 1)
    angles = np.arccos(dots)  # N x N matrix of angular distances
    
    # For each point on the sphere, the distance to the nearest config point
    # We approximate this by considering the config points themselves and their
    # circumcenters. But for a differentiable loss, we use a soft-min.
    
    # A simpler differentiable approach: minimize the sum of squared distances
    # from each point to its k-nearest neighbors, which correlates with covering radius.
    
    # Actually, let's use a different approach: the covering radius is bounded by
    # the maximum distance from any point to its nearest neighbor. We can approximate
    # this by considering a fine grid of test points.
    
    # For efficiency, use a coarse grid
    n_grid = 20
    # Generate grid points on the sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2*np.pi, n_grid)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Convert to Cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_pts /= np.linalg.norm(grid_pts, axis=1, keepdims=True)
    
    # Compute distances from each grid point to each config point
    dots = np.clip(grid_pts @ pts.T, -1, 1)  # (n_grid^2, N)
    angles = np.arccos(dots)  # angular distances
    
    # Soft-min over config points for each grid point
    # Use log-sum-exp trick for numerical stability
    min_angles = np.min(angles, axis=1)
    
    # Return the maximum of these minimum distances (the covering radius approximation)
    return np.max(min_angles)


def _gradient_optimize(pts, max_time=60.0, lr=0.01, temperature=0.01):
    """Optimize point positions using gradient descent on a soft covering loss."""
    start_time = time.time()
    pts = pts.copy()
    
    # We'll use a simple approach: for each point, compute the gradient of the
    # covering radius with respect to its position, and update.
    
    # A practical approach: use the fact that the covering radius is determined
    # by the Delaunay triangulation. For each point, move it towards the centroid
    # of its Voronoi cell (which reduces the maximum distance in that cell).
    
    # Actually, let's use a simpler heuristic: for each point, compute the average
    # of the circumcenters of the triangles it belongs to, and move towards that.
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
    
    # For each point, find the triangles it belongs to
    point_triangles = [[] for _ in range(N)]
    for simplex in hull.simplices:
        for idx in simplex:
            point_triangles[idx].append(simplex)
    
    # Compute the target position for each point as the average of circumcenters
    # of its triangles
    targets = np.zeros_like(pts)
    counts = np.zeros(N)
    
    for simplex in hull.simplices:
        P = pts[simplex]
        v1 = P[1] - P[0]
        v2 = P[2] - P[0]
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        
        # The circumcenter is in the direction of n or -n. Choose the one
        # that is "inside" the triangle (closer to the centroid)
        centroid = np.mean(P, axis=0)
        if np.dot(n, centroid) < 0:
            n = -n
        
        # Add this circumcenter to the targets of each point in the triangle
        for idx in simplex:
            targets[idx] += n
            counts[idx] += 1
    
    # Normalize targets
    for i in range(N):
        if counts[i] > 0:
            targets[i] /= counts[i]
            norm_t = np.linalg.norm(targets[i])
            if norm_t > 1e-12:
                targets[i] /= norm_t
    
    # Update points using gradient descent
    for iteration in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Recompute targets
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        point_triangles = [[] for _ in range(N)]
        for simplex in hull.simplices:
            for idx in simplex:
                point_triangles[idx].append(simplex)
        
        new_targets = np.zeros_like(pts)
        new_counts = np.zeros(N)
        
        for simplex in hull.simplices:
            P = pts[simplex]
            v1 = P[1] - P[0]
            v2 = P[2] - P[0]
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            
            centroid = np.mean(P, axis=0)
            if np.dot(n, centroid) < 0:
                n = -n
            
            for idx in simplex:
                new_targets[idx] += n
                new_counts[idx] += 1
        
        for i in range(N):
            if new_counts[i] > 0:
                new_targets[i] /= new_counts[i]
                norm_t = np.linalg.norm(new_targets[i])
                if norm_t > 1e-12:
                    new_targets[i] /= norm_t
        
        # Update: move towards target
        for i in range(N):
            if new_counts[i] > 0:
                # Move towards the target
                pts[i] = pts[i] + lr * (new_targets[i] - pts[i])
                # Normalize
                pts[i] /= np.linalg.norm(pts[i])
    
    return pts


def _local_refine(pts, max_time=30.0):
    """Local refinement: try small moves and keep improvements."""
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    for iteration in range(1000):
        if time.time() - start_time > max_time:
            break
        
        # Pick a random point
        idx = np.random.randint(N)
        
        # Try several random perturbations
        best_local = best_pts.copy()
        best_local_radius = best_radius
        
        for attempt in range(20):
            delta = np.random.randn(3) * 0.005
            new_pts = best_local.copy()
            new_pts[idx] += delta
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_local_radius - 1e-8:
                best_local = new_pts
                best_local_radius = new_radius
        
        if best_local_radius < best_radius - 1e-8:
            best_pts = best_local
            best_radius = best_local_radius
    
    return best_pts, best_radius


def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Gradient-based optimization
    remaining = deadline - time.time() - 20.0
    if remaining > 10.0:
        grad_pts = _gradient_optimize(best_pts, max_time=min(remaining, 60.0))
        grad_radius = _covering_radius(grad_pts)
        if grad_radius < best_radius:
            best_pts = grad_pts
            best_radius = grad_radius
    
    # Phase 2: Local refinement
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        refined_pts, refined_radius = _local_refine(best_pts, max_time=remaining)
        if refined_radius < best_radius:
            best_pts = refined_pts
            best_radius = refined_radius
    
    # Phase 3: Additional local refinement passes
    remaining = deadline - time.time() - 2.0
    if remaining > 3.0:
        refined_pts, refined_radius = _local_refine(best_pts, max_time=remaining)
        if refined_radius < best_radius:
            best_pts = refined_pts
            best_radius = refined_radius
    
    # Ensure validity: no coincident points, all on unit sphere
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                best_pts[j] += np.random.randn(3) * 0.01
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
