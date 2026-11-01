"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _fast_covering_radius(pts, test_points):
    """Compute a fast approximation of the covering radius using pre-generated test points."""
    dots = np.clip(pts @ test_points.T, -1, 1)
    angles = np.arccos(dots)
    min_angles = np.min(angles, axis=1)
    return np.max(min_angles)

def _covering_radius(pts):
    """Compute the exact covering radius in degrees using ConvexHull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0  # fallback
    
    max_radius = 0.0
    for simplex in hull.simplices:
        P = pts[simplex]
        v1 = P[1] - P[0]
        v2 = P[2] - P[0]
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        
        for sign in [1, -1]:
            c = sign * n
            dots = np.clip(pts @ c, -1, 1)
            angles = np.arccos(dots)
            circumradius = np.max(angles[simplex])
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


def _local_refine_fast(pts, test_points, max_time=30.0):
    """Local refinement using fast covering radius approximation."""
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _fast_covering_radius(best_pts, test_points)
    
    # Adaptive step size
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        idx = np.random.randint(N)
        best_local = best_pts.copy()
        best_local_radius = best_radius
        
        for attempt in range(10):
            delta = np.random.randn(3) * step_size
            new_pts = best_local.copy()
            new_pts[idx] += delta
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_radius = _fast_covering_radius(new_pts, test_points)
            if new_radius < best_local_radius - 1e-6:
                best_local = new_pts
                best_local_radius = new_radius
        
        if best_local_radius < best_radius - 1e-6:
            best_pts = best_local
            best_radius = best_local_radius
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-5:
                    break
    
    return best_pts, best_radius


def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate test points for fast evaluation
    # Use a dense grid to approximate the sphere
    n_grid = 40
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2*np.pi, n_grid)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    test_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    test_points /= np.linalg.norm(test_points, axis=1, keepdims=True)
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_radius = _fast_covering_radius(best_pts, test_points)
    
    # Phase 1: Fast local refinement
    remaining = deadline - time.time() - 10.0
    if remaining > 10.0:
        refined_pts, refined_radius = _local_refine_fast(best_pts, test_points, max_time=remaining)
        if refined_radius < best_radius:
            best_pts = refined_pts
            best_radius = refined_radius
    
    # Phase 2: Exact refinement with fewer steps
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        # Use exact radius for final polish
        exact_radius = _covering_radius(best_pts)
        # Do a few steps of exact local search if time permits
        # We'll just use the fast one again but with smaller steps if we have time
        # Actually, let's just do a final check and return
        pass
    
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
