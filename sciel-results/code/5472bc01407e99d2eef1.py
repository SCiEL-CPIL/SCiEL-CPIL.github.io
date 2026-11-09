"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import math

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import math

TIME_LIMIT = 130.0  # safety margin before 150s kill


def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on unit sphere using Fibonacci spiral with optional angular offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _covering_radius_chord(points, test_points):
    """Compute covering radius as max chord distance (proportional to angular distance)."""
    tree = cKDTree(points)
    dists, _ = tree.query(test_points)
    return np.max(dists)


def _spherical_circumcenter_of_triangle(p1, p2, p3):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    Returns the circumcenter as a unit vector and the circumradius (angular distance in radians).
    If the triangle is degenerate, returns None.
    """
    # Normalize
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    p3 = p3 / np.linalg.norm(p3)
    
    # Check for degeneracy
    cross12 = np.cross(p1, p2)
    cross23 = np.cross(p2, p3)
    cross31 = np.cross(p3, p1)
    
    # The circumcenter is proportional to the sum of these cross products
    # weighted by the sine of the opposite angles
    c1 = np.linalg.norm(cross12)
    c2 = np.linalg.norm(cross23)
    c3 = np.linalg.norm(cross31)
    
    if c1 < 1e-12 or c2 < 1e-12 or c3 < 1e-12:
        return None, None
    
    # For a spherical triangle, the circumcenter can be found using the formula:
    # The circumcenter is the intersection of the perpendicular bisectors
    # A simpler approach: use the fact that the circumcenter equidistant from all three vertices
    
    # Using the formula: circumcenter = (sin(A)*p1 + sin(B)*p2 + sin(C)*p3) / |...|
    # where A, B, C are the angles opposite to p1, p2, p3
    
    # Actually, a more robust method: solve the system where |c - p1| = |c - p2| = |c - p3|
    # This gives linear equations: c·(p1-p2) = 0.5*|p1-p2|^2 - 0.5*|p2-p2|^2 ... 
    
    # Let's use a different approach: the circumcenter lies in the plane spanned by the three points
    # and equidistant from all three.
    
    # The circumcenter c satisfies:
    # c·p1 = c·p2 = c·p3 = cos(r) where r is the circumradius
    # And |c| = 1
    
    # From c·p1 = c·p2: c·(p1-p2) = 0
    # From c·p2 = c·p3: c·(p2-p3) = 0
    # So c is perpendicular to (p1-p2) and (p2-p3), hence c is parallel to (p1-p2)×(p2-p3)
    
    d1 = p1 - p2
    d2 = p2 - p3
    c_candidate = np.cross(d1, d2)
    c_norm = np.linalg.norm(c_candidate)
    
    if c_norm < 1e-12:
        return None, None
    
    c_candidate = c_candidate / c_norm
    
    # Now we need to determine the sign and the actual circumradius
    # The circumcenter should be on the same side as the triangle's interior
    # Let's compute the angular distances from c_candidate to each vertex
    cos_r1 = np.clip(np.dot(c_candidate, p1), -1.0, 1.0)
    cos_r2 = np.clip(np.dot(c_candidate, p2), -1.0, 1.0)
    cos_r3 = np.clip(np.dot(c_candidate, p3), -1.0, 1.0)
    
    r1 = np.arccos(cos_r1)
    r2 = np.arccos(cos_r2)
    r3 = np.arccos(cos_r3)
    
    # The circumradius should be the same for all three (within numerical error)
    # If not, we may have the wrong sign
    max_diff = max(abs(r1 - r2), abs(r2 - r3), abs(r1 - r3))
    if max_diff > 0.1:
        # Try the opposite direction
        c_candidate = -c_candidate
        cos_r1 = np.clip(np.dot(c_candidate, p1), -1.0, 1.0)
        cos_r2 = np.clip(np.dot(c_candidate, p2), -1.0, 1.0)
        cos_r3 = np.clip(np.dot(c_candidate, p3), -1.0, 1.0)
        r1 = np.arccos(cos_r1)
        r2 = np.arccos(cos_r2)
        r3 = np.arccos(cos_r3)
        max_diff = max(abs(r1 - r2), abs(r2 - r3), abs(r1 - r3))
        if max_diff > 0.1:
            return None, None
    
    circumradius = max(r1, r2, r3)
    return c_candidate, circumradius


def _compute_facet_circumradii(points):
    """
    Compute the circumradius (in radians) of each facet of the convex hull.
    Returns a list of (facet_indices, circumradius) tuples.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    
    facets = []
    for simplex in hull.simplices:
        if len(simplex) == 3:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            cc, cr = _spherical_circumcenter_of_triangle(p1, p2, p3)
            if cc is not None:
                facets.append((simplex, cr, cc))
        elif len(simplex) == 4:
            # Tetrahedron: use the largest face
            max_cr = 0
            best_face = None
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        p1, p2, p3 = points[simplex[i]], points[simplex[j]], points[simplex[k]]
                        cc, cr = _spherical_circumcenter_of_triangle(p1, p2, p3)
                        if cc is not None and cr > max_cr:
                            max_cr = cr
                            best_face = (simplex[i], simplex[j], simplex[k])
            if best_face is not None:
                facets.append((np.array(best_face), max_cr, None))
    
    return facets


def _optimize(points, test_points, max_iter=500, initial_step=0.03, decay=0.997, deadline=None):
    """Optimize points to minimize covering radius using targeted force heuristic."""
    points = points.copy()
    n = len(points)
    
    best_points = points.copy()
    best_radius = _covering_radius_chord(points, test_points)
    
    tree = cKDTree(points)
    step_size = initial_step
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find nearest candidate for each test point
        dists, idx = tree.query(test_points)
        current_radius = np.max(dists)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Identify worst test points (top 10% by distance)
        n_worst = max(1, len(test_points) // 10)
        worst_indices = np.argsort(dists)[-n_worst:]
        
        # For each candidate, compute force based on:
        # 1. Weighted sum of vectors to worst test points in its cell
        # 2. Slight repulsion from other candidates to prevent collapse
        
        forces = np.zeros((n, 3))
        
        # Accumulate weighted vectors for worst points
        for j in worst_indices:
            i = idx[j]
            vec = test_points[j] - points[i]
            forces[i] += vec
        
        # Add mild repulsion from other candidates to maintain spread
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-8:
                        forces[i] += 0.01 * d / dist
        
        # Normalize forces and apply
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_forces = forces / safe_norms
        
        # Adaptive step: scale by current radius relative to best
        radius_ratio = current_radius / (best_radius + 1e-10)
        effective_step = step_size * min(1.0, radius_ratio)
        
        points += effective_step * unit_forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        tree = cKDTree(points)
        step_size *= decay
    
    return best_points


def _spherical_gradient_descent(points, max_iter=200, step_size=0.05, deadline=None):
    """
    Spherical Riemannian gradient descent to minimize the maximum facet circumradius.
    Uses the exact spherical circumcenter of the worst facet to compute gradients.
    """
    points = points.copy()
    n = len(points)
    best_points = points.copy()
    best_max_cr = float('inf')
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all facet circumradii
        facets = _compute_facet_circumradii(points)
        if not facets:
            break
        
        # Find the worst facet
        worst_facet_idx = np.argmax([f[1] for f in facets])
        worst_simplex, worst_cr, worst_cc = facets[worst_facet_idx]
        
        if worst_cr < best_max_cr:
            best_max_cr = worst_cr
            best_points = points.copy()
        
        # Compute gradient: move the vertices of the worst facet toward the circumcenter
        # The gradient for each vertex is the direction from the vertex to the circumcenter
        # projected onto the tangent space at the vertex
        
        forces = np.zeros((n, 3))
        
        for vi in worst_simplex:
            v = points[vi]
            if worst_cc is not None:
                # Direction from vertex to circumcenter
                d = worst_cc - v
                # Project onto tangent space at v: d - (d·v)v
                d_tangent = d - np.dot(d, v) * v
                d_norm = np.linalg.norm(d_tangent)
                if d_norm > 1e-10:
                    forces[vi] += d_tangent / d_norm
            else:
                # Fallback: move away from centroid of the facet
                centroid = np.mean(points[worst_simplex], axis=0)
                centroid /= np.linalg.norm(centroid)
                d = centroid - v
                d_tangent = d - np.dot(d, v) * v
                d_norm = np.linalg.norm(d_tangent)
                if d_norm > 1e-10:
                    forces[vi] += d_tangent / d_norm
        
        # Normalize and apply
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_forces = forces / safe_norms
        
        points += step_size * unit_forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= 0.995
    
    return best_points, best_max_cr


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Test points: use a moderate density for speed
    N_test = 3000
    test_points = _fibonacci_sphere(N_test)
    
    best_points = None
    best_radius = float('inf')
    
    # Multiple restarts with different initializations
    restarts = [
        ("fibonacci", _fibonacci_sphere(N, offset=0.0)),
        ("fibonacci_rot", _fibonacci_sphere(N, offset=np.pi / 7)),
        ("random", _random_sphere(N, seed=42)),
        ("random2", _random_sphere(N, seed=123)),
        ("random3", _random_sphere(N, seed=456)),
    ]
    
    for name, init_points in restarts:
        if time.time() > deadline - 10:
            break
        
        # Coarse optimization
        opt_points = _optimize(init_points, test_points, max_iter=300, initial_step=0.04, 
                               decay=0.996, deadline=deadline)
        
        radius = _covering_radius_chord(opt_points, test_points)
        if radius < best_radius:
            best_radius = radius
            best_points = opt_points.copy()
    
    # Fine polish: use denser test points for the best configuration
    if best_points is not None and time.time() < deadline - 20:
        N_test_fine = 10000
        test_points_fine = _fibonacci_sphere(N_test_fine)
        
        polished = _optimize(best_points, test_points_fine, max_iter=200, initial_step=0.01,
                             decay=0.998, deadline=deadline)
        
        radius_fine = _covering_radius_chord(polished, test_points_fine)
        # Compare using same metric (chord distance)
        if radius_fine < best_radius:
            best_points = polished
    
    # Spherical gradient descent to minimize facet circumradii
    if best_points is not None and time.time() < deadline - 15:
        sg_points, sg_cr = _spherical_gradient_descent(best_points, max_iter=100, step_size=0.03, deadline=deadline)
        
        # Compare using chord distance metric
        radius_sg = _covering_radius_chord(sg_points, test_points_fine)
        if radius_sg < best_radius:
            best_points = sg_points
    
    return best_points
# EVOLVE-BLOCK-END
