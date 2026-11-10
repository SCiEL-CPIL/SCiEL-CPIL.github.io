"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START

def _circumradius_degrees(points):
    """
    Compute the covering radius in degrees for a set of N points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This equals the maximum circumradius of the facets
    of the convex hull (in angular terms).
    
    For a spherical cap defined by a facet with vertices v1, ..., vk, the circumcenter
    is the point on the sphere equidistant from all vertices. The angular radius is
    the angle from that center to any vertex.
    
    We compute this by: for each triangular facet (the convex hull of points on the sphere
    gives triangular facets), find the circumcenter on the sphere and the angular radius.
    """
    points = np.asarray(points)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    
    # For each facet (triangle), compute the spherical circumradius
    # A facet has 3 vertices (indices into points)
    max_circ_r = 0.0
    
    for simplex in hull.simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        
        # The circumcenter of the triangle on the sphere:
        # It's the intersection of the sphere with the plane that is the perpendicular bisector
        # in 3D of the three points. Actually, the circumcenter of a spherical triangle
        # is the point on the sphere equidistant from all three vertices.
        
        # The plane through the three points has normal n = (v2-v1) x (v3-v1)
        # The circumcenter c on the sphere satisfies |c - v1| = |c - v2| = |c - v3|
        # and |c| = 1.
        
        # Alternative: the circumcenter is along the direction of the cross product of
        # (v2-v1) and (v3-v1), but that's the normal to the plane, not the circumcenter.
        
        # Better approach: solve for c such that c·v1 = c·v2 = c·v3 = cos(theta)
        # and |c| = 1.
        
        # The circumcenter direction is the normalized solution to the system.
        # c = a*v1 + b*v2 + c_coeff*v3 with constraints.
        
        # Simpler: the circumcenter is in the direction of the vector:
        # c_dir = (v2 - v1) x (v3 - v1) ... no that's the normal.
        
        # Let's use: the circumcenter c satisfies c·v1 = c·v2 = c·v3.
        # This means c is orthogonal to (v1-v2) and (v1-v3).
        # So c is parallel to (v1-v2) x (v1-v3).
        
        c_dir = np.cross(v1 - v2, v1 - v3)
        c_norm = np.linalg.norm(c_dir)
        if c_norm < 1e-12:
            # Degenerate triangle
            continue
        c = c_dir / c_norm
        
        # Ensure c is on the same side of the facet as the sphere center outward
        # The facet normal (pointing outward from the convex hull) should point away from origin
        # Actually, for a point on the sphere, the circumcenter should be the point on the sphere
        # that is the "center" of the cap. Let's check: the circumcenter should satisfy
        # c·v1 >= 0 (since the cap is less than a hemisphere for good coverings).
        
        # Actually, we need the circumcenter that is the center of the spherical cap.
        # The two candidates are c and -c. The correct one is the one that is
        # "inside" the cap, i.e., c·v1 > 0 (the angle from c to v1 is < 90 degrees).
        
        if c.dot(v1) < 0:
            c = -c
        
        # The angular radius is the angle between c and any vertex
        cos_theta = c.dot(v1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta_deg = np.degrees(theta)
        
        if theta_deg > max_circ_r:
            max_circ_r = theta_deg
    
    return max_circ_r


def _objective(points):
    """Covering radius in degrees (lower is better)."""
    return _circumradius_degrees(points)


def _optimize_from_start(start_points, max_iter=200, deadline=None):
    """
    Local optimization using repulsive-force-guided gradient descent.
    We minimize the covering radius by moving points to reduce the largest circumradius.
    """
    points = start_points.copy()
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_obj = _objective(points)
    
    # Gradient descent with adaptive step size
    lr = 0.05
    patience = 30
    no_improve = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute gradient: for each point, find the facet(s) that determine the covering radius
        # and push the point away from the circumcenter of those facets.
        
        # Compute all circumradii and identify the critical facets
        hull = ConvexHull(points)
        
        # For each point, compute the gradient of the covering radius
        # The covering radius is determined by the maximum circumradius facet.
        # We want to move points to reduce this maximum.
        
        # Strategy: for each critical facet (circumradius near the max), 
        # move its three vertices away from the circumcenter.
        
        grad = np.zeros_like(points)
        
        # Find the max circumradius and which facets achieve it
        circ_radii = []
        facet_centers = []
        facet_indices = []
        
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            
            c_dir = np.cross(v1 - v2, v1 - v3)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-12:
                continue
            c = c_dir / c_norm
            if c.dot(v1) < 0:
                c = -c
            
            cos_theta = np.clip(c.dot(v1), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            circ_radii.append(theta)
            facet_centers.append(c)
            facet_indices.append(simplex)
        
        if not circ_radii:
            break
        
        circ_radii = np.array(circ_radii)
        max_cr = np.max(circ_radii)
        
        # Use soft-max weighting to get a smooth gradient
        # Weight each facet by exp(theta / temperature)
        temperature = 0.01
        weights = np.exp(circ_radii / temperature)
        weights /= np.sum(weights)
        
        # For each facet, the gradient contribution:
        # Moving a vertex v away from the circumcenter c reduces the circumradius.
        # The direction to move is (v - c) projected onto the tangent plane at v.
        for i, (w, c, simplex) in enumerate(zip(weights, facet_centers, facet_indices)):
            for idx in simplex:
                v = points[idx]
                # Direction from c to v, projected onto tangent plane at v
                # Tangent plane at v: orthogonal to v
                d = v - c
                # Project onto tangent plane: d - (d·v)v
                d_tan = d - np.dot(d, v) * v
                d_tan_norm = np.linalg.norm(d_tan)
                if d_tan_norm > 1e-12:
                    d_tan /= d_tan_norm
                grad[idx] += w * d_tan
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm == 0] = 1.0
        grad = grad / grad_norm
        
        # Update points
        new_points = points + lr * grad
        # Project back to sphere
        new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_norms[new_norms == 0] = 1.0
        new_points = new_points / new_norms
        
        new_obj = _objective(new_points)
        
        if new_obj < best_obj - 1e-10:
            best_obj = new_obj
            best_points = new_points.copy()
            points = new_points
            no_improve = 0
            # Decrease learning rate slowly
            lr *= 0.99
        else:
            no_improve += 1
            # Try smaller step
            lr *= 0.9
            if no_improve >= patience:
                break
        
        if lr < 1e-8:
            break
    
    return best_points, best_obj


def _generate_random_start():
    """Generate a random set of N points on the unit sphere."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(np.random.uniform(-1, 1, N))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _generate_fibonacci_start():
    """Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _generate_icosahedron_based_start():
    """Start from icosahedron vertices and subdivide."""
    # 12 icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Use these 12 as a subset and add 38 more via optimization
    # Start with 12 icosahedron points and 38 random points
    remaining = N - 12
    rand_pts = _generate_random_start()
    # Only keep random points that are far from icosahedron vertices
    selected = []
    for _ in range(remaining):
        best_idx = None
        best_dist = -1
        for idx in range(remaining):
            min_d = np.min(np.linalg.norm(verts - rand_pts[idx], axis=1))
            if min_d > best_dist:
                best_dist = min_d
                best_idx = idx
        selected.append(rand_pts[best_idx])
        # Remove that point from consideration
        rand_pts = np.delete(rand_pts, best_idx, axis=0)
    
    all_pts = np.vstack([verts, np.array(selected)])
    return all_pts


def _polish(points, deadline=None):
    """High-precision polish pass with tighter tolerances."""
    points = points.copy()
    best_obj = _objective(points)
    
    # Multiple passes with decreasing step size
    for pass_num in range(5):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        max_iter = 100
        lr = 0.01 / (pass_num + 1)
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            
            grad = np.zeros_like(points)
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            circ_radii = []
            facet_centers = []
            facet_indices = []
            
            for simplex in hull.simplices:
                v1 = points[simplex[0]]
                v2 = points[simplex[1]]
                v3 = points[simplex[2]]
                
                c_dir = np.cross(v1 - v2, v1 - v3)
                c_norm = np.linalg.norm(c_dir)
                if c_norm < 1e-12:
                    continue
                c = c_dir / c_norm
                if c.dot(v1) < 0:
                    c = -c
                
                cos_theta = np.clip(c.dot(v1), -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
                circ_radii.append(theta)
                facet_centers.append(c)
                facet_indices.append(simplex)
            
            if not circ_radii:
                break
            
            circ_radii = np.array(circ_radii)
            temperature = 0.005
            weights = np.exp(circ_radii / temperature)
            weights /= np.sum(weights)
            
            for i, (w, c, simplex) in enumerate(zip(weights, facet_centers, facet_indices)):
                for idx in simplex:
                    v = points[idx]
                    d = v - c
                    d_tan = d - np.dot(d, v) * v
                    d_tan_norm = np.linalg.norm(d_tan)
                    if d_tan_norm > 1e-12:
                        d_tan /= d_tan_norm
                    grad[idx] += w * d_tan
            
            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norm[grad_norm == 0] = 1.0
            grad = grad / grad_norm
            
            new_points = points + lr * grad
            new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_norms[new_norms == 0] = 1.0
            new_points = new_points / new_norms
            
            new_obj = _objective(new_points)
            
            if new_obj < best_obj - 1e-12:
                best_obj = new_obj
                points = new_points
            else:
                lr *= 0.95
                if lr < 1e-10:
                    break
    
    return points, best_obj


def construct():
    """
    Main function: find 50 points on the unit sphere minimizing the covering radius.
    Uses multiple diverse starting configurations with repulsive-force-guided optimization
    and a final exact-radius polish pass.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_generate_fibonacci_start())
    
    # 2. Icosahedron-based
    starts.append(_generate_icosahedron_based_start())
    
    # 3. Random starts
    for i in range(5):
        starts.append(_generate_random_start())
    
    # Optimize from each start
    best_points = None
    best_obj = np.inf
    
    for s_idx, start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        # Allocate time per start
        remaining_starts = len(starts) - s_idx
        time_per_start = min(20.0, (deadline - 20.0 - time.time()) / remaining_starts)
        start_deadline = time.time() + time_per_start
        
        points, obj = _optimize_from_start(start, max_iter=300, deadline=start_deadline)
        
        if obj < best_obj:
            best_obj = obj
            best_points = points.copy()
    
    # Final polish pass on the best result
    if best_points is not None:
        if time.time() < deadline - 5.0:
            best_points, best_obj = _polish(best_points, deadline=deadline)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
