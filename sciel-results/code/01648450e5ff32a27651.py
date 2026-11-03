"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0])
    return v / norm

def _spherical_distance(p, q):
    """Angular distance in radians between two unit vectors."""
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum over all facets of the convex hull 
    of the angular distance from the facet center to its vertices.
    For a spherical triangle with vertices A, B, C, the circumradius is 
    the angular distance from the circumcenter to any vertex.
    If the triangle is acute, circumcenter is inside. If obtuse, it's outside,
    but the farthest point in the cap is still the circumcenter (which may be 
    on the opposite side, but for covering radius we care about the empty cap).
    Actually, the covering radius is determined by the largest empty circle.
    The center of the largest empty circle is either:
    1. The midpoint of the longest edge of the Delaunay triangulation (if the 
       circumcircle of the triangle containing that edge is not empty, but 
       actually the covering radius is the max circumradius of the Delaunay 
       triangles).
    
    Correction: The covering radius is the maximum circumradius of the facets 
    of the convex hull (Delaunay triangulation on the sphere).
    For a spherical triangle with side lengths a, b, c (angular distances),
    the circumradius R can be computed.
    However, a simpler and robust way is to compute the circumcenter of each 
    spherical triangle and find the max distance from the circumcenter to 
    its vertices. If the triangle is obtuse, the circumcenter lies outside 
    the triangle, and the max distance to a vertex is still the circumradius, 
    but the "empty cap" property might be violated if the circumcenter is 
    not in the interior of the triangle. In that case, the covering radius 
    for that facet is actually half the longest side.
    
    Let's use a robust method:
    For each facet (triangle), compute the circumcenter.
    If the circumcenter is inside the triangle (all angles < 90), the 
    circumradius is the distance from circumcenter to any vertex.
    If the triangle is obtuse, the circumcenter is outside, and the 
    "worst point" in the cap is the midpoint of the longest side.
    The covering radius for that facet is half the length of the longest side.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        # Fallback if hull fails (e.g., points in a hemisphere)
        # Use a simple O(N^2) check for max empty cap approximation
        # But this should not happen if points are well-spread.
        # Return a large value to penalize.
        return np.pi / 2.0

    max_radius = 0.0
    for tri in facets:
        A, B, C = points[tri[0]], points[tri[1]], points[tri[2]]
        
        # Compute side lengths (angular distances)
        a = _spherical_distance(B, C)
        b = _spherical_distance(A, C)
        c = _spherical_distance(A, B)
        
        # Check if triangle is obtuse
        # A triangle is obtuse if the square of the longest side is greater 
        # than the sum of the squares of the other two sides (in Euclidean space).
        # On the sphere, the condition for the circumcenter being inside is 
        # that all angles are < 90 degrees.
        # We can check if the circumcenter is inside by verifying that the 
        # barycentric coordinates are positive, or more simply, by checking 
        # if the dot product of the circumcenter with the normal of the 
        # triangle is positive and the distances to edges are consistent.
        
        # A simpler robust check:
        # Compute circumcenter O.
        # If O is inside the triangle, the covering radius for this facet is 
        # the distance from O to A (or B or C).
        # If O is outside, the covering radius is half the longest side.
        
        # Compute circumcenter O of spherical triangle ABC.
        # O is proportional to (A + B + C) only for equilateral triangles.
        # General formula:
        # O = (sin(a) * A + sin(b) * B + sin(c) * C) / |...| ? No, that's for planar.
        
        # Let's use the property that O is equidistant from A, B, C.
        # We can solve for O by intersecting the great circles perpendicular 
        # to the edges at their midpoints.
        # Alternatively, use the formula:
        # O = (A * sin(2*alpha) + B * sin(2*beta) + C * sin(2*gamma)) / |...|
        # where alpha, beta, gamma are the angles at A, B, C.
        
        # Compute angles using spherical law of cosines:
        # cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A)
        def angle_at(A, B, C):
            # Angle at A
            b = _spherical_distance(A, C)
            c = _spherical_distance(A, B)
            a = _spherical_distance(B, C)
            cos_A = (np.cos(b) - np.cos(a)*np.cos(c)) / (np.sin(a)*np.sin(c))
            cos_A = np.clip(cos_A, -1.0, 1.0)
            return np.arccos(cos_A)
        
        alpha = angle_at(A, B, C)
        beta = angle_at(B, A, C)
        gamma = angle_at(C, A, B)
        
        # Circumcenter
        O = (A * np.sin(2*alpha) + B * np.sin(2*beta) + C * np.sin(2*gamma))
        O = _normalize(O)
        
        # Check if O is inside the triangle.
        # A point O is inside spherical triangle ABC if the sum of the 
        # angles AOB, BOC, COA is 2*pi (or if the barycentric coords are positive).
        # A simpler check: The distance from O to A should be the circumradius, 
        # and O should be on the same side of each edge as the opposite vertex.
        
        # Let's just compute the circumradius R = distance(O, A).
        R = _spherical_distance(O, A)
        
        # Check if the triangle is obtuse.
        # If any angle > pi/2, the circumcenter is outside, and the 
        # covering radius for this facet is half the longest side.
        if alpha > np.pi/2 or beta > np.pi/2 or gamma > np.pi/2:
            # Find longest side
            longest = max(a, b, c)
            R = longest / 2.0
        else:
            # O is inside, R is the circumradius
            # But we must ensure O is actually inside. If the triangle is 
            # very small or degenerate, this should hold.
            # If O is outside despite acute angles (which shouldn't happen 
            # for convex hull facets), we fall back to half longest side.
            # For robustness, we can check if the distance from O to the 
            # plane of the triangle is positive (same side as origin? No, 
            # opposite to origin for convex hull).
            # Actually, for a convex hull facet, the normal points outward.
            # O should be on the outward side.
            normal = np.cross(B-A, C-A)
            normal = _normalize(normal)
            # If dot(O, normal) < 0, O is on the inward side, which is 
            # impossible for a circumcenter of a convex hull facet unless 
            # the triangle is obtuse.
            if np.dot(O, normal) < 0:
                longest = max(a, b, c)
                R = longest / 2.0
        
        if R > max_radius:
            max_radius = R

    return max_radius

def _energy(points):
    """
    Energy function to spread points out.
    Sum of inverse distances to avoid clustering.
    """
    dists = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dists[i, j] = _spherical_distance(points[i], points[j])
    # Avoid division by zero
    dists = np.maximum(dists, 1e-6)
    energy = np.sum(1.0 / dists)
    return energy

def _worst_facet_gradient(points):
    """
    Compute the gradient of the covering radius with respect to the points.
    This is complex, so we use a heuristic:
    Identify the worst facet (largest circumradius).
    Move its vertices away from the circumcenter.
    """
    # First, find the worst facet
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return None, None

    max_radius = 0.0
    worst_tri = None
    worst_O = None
    for tri in facets:
        A, B, C = points[tri[0]], points[tri[1]], points[tri[2]]
        a = _spherical_distance(B, C)
        b = _spherical_distance(A, C)
        c = _spherical_distance(A, B)
        
        def angle_at(A, B, C):
            b = _spherical_distance(A, C)
            c = _spherical_distance(A, B)
            a = _spherical_distance(B, C)
            cos_A = (np.cos(b) - np.cos(a)*np.cos(c)) / (np.sin(a)*np.sin(c))
            cos_A = np.clip(cos_A, -1.0, 1.0)
            return np.arccos(cos_A)
        
        alpha = angle_at(A, B, C)
        beta = angle_at(B, A, C)
        gamma = angle_at(C, A, B)
        
        O = (A * np.sin(2*alpha) + B * np.sin(2*beta) + C * np.sin(2*gamma))
        O = _normalize(O)
        
        if alpha > np.pi/2 or beta > np.pi/2 or gamma > np.pi/2:
            longest = max(a, b, c)
            R = longest / 2.0
            # For obtuse, the "center" of the empty cap is the midpoint of 
            # the longest side.
            if longest == a:
                O = _normalize((B + C) / 2.0)
            elif longest == b:
                O = _normalize((A + C) / 2.0)
            else:
                O = _normalize((A + B) / 2.0)
        else:
            R = _spherical_distance(O, A)
            normal = np.cross(B-A, C-A)
            normal = _normalize(normal)
            if np.dot(O, normal) < 0:
                longest = max(a, b, c)
                R = longest / 2.0
                if longest == a:
                    O = _normalize((B + C) / 2.0)
                elif longest == b:
                    O = _normalize((A + C) / 2.0)
                else:
                    O = _normalize((A + B) / 2.0)
        
        if R > max_radius:
            max_radius = R
            worst_tri = tri
            worst_O = O

    if worst_tri is None:
        return None, None

    # Gradient: Move the vertices of the worst facet away from the circumcenter.
    # The direction to move is from O to the vertex, projected onto the tangent plane.
    grad = np.zeros_like(points)
    for idx in worst_tri:
        p = points[idx]
        # Vector from O to p
        v = p - worst_O
        # Project v onto the tangent plane at p
        # Tangent plane is perpendicular to p.
        # v_tangent = v - (v . p) p
        v_tangent = v - np.dot(v, p) * p
        # Normalize
        v_tangent = _normalize(v_tangent)
        # The gradient of the distance to O is v_tangent.
        # We want to MINIMIZE the distance, so we move in the direction 
        # of -v_tangent? No, we want to shrink the facet, so we want to 
        # move the vertices TOWARDS the circumcenter?
        # Wait, if we move vertices towards the circumcenter, the triangle 
        # shrinks, but the circumradius might not shrink if the circumcenter 
        # moves.
        # Actually, to shrink the empty cap, we should move the vertices 
        # AWAY from the circumcenter? No, that would enlarge the cap.
        # We should move the vertices TOWARDS the circumcenter to reduce 
        # the distance.
        # So the gradient descent step is: p_new = p - alpha * grad, 
        # where grad is the direction of increasing distance.
        # The direction of increasing distance from O is v_tangent (from O to p).
        # So we move in the direction -v_tangent.
        grad[idx] -= v_tangent

    return grad, worst_O

def construct():
    start_time = time.time()
    deadline = 145.0

    # Initial points: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = _covering_radius(best_points)

    # Phase 1: Global energy minimization to spread points
    # Use a simple gradient descent on the energy function
    def energy_grad(points):
        # Approximate gradient of energy
        grad = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = _spherical_distance(points[i], points[j])
                    if d < 1e-6:
                        continue
                    # Gradient of 1/d w.r.t. points[i]
                    # d/dp (1/d) = -1/d^2 * d/dp d
                    # d/dp d = (p - q) / d (tangent component)
                    v = points[i] - points[j]
                    v_tangent = v - np.dot(v, points[i]) * points[i]
                    grad[i] -= (1.0 / (d * d)) * (v_tangent / d)
        return grad

    # Run a few steps of gradient descent
    for step in range(100):
        if time.time() > deadline:
            break
        grad = energy_grad(points)
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
        grad = grad / grad_norm
        # Step size
        lr = 0.01
        points = points - lr * grad
        # Re-normalize to sphere
        for i in range(N):
            points[i] = _normalize(points[i])
        
        # Update best
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

    # Phase 2: Worst-facet projection
    # Iteratively identify the worst facet and move its vertices
    for step in range(500):
        if time.time() > deadline:
            break
        
        grad, worst_O = _worst_facet_gradient(points)
        if grad is None:
            break
        
        # Step size for facet projection
        lr = 0.05
        points = points - lr * grad
        for i in range(N):
            points[i] = _normalize(points[i])
        
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        else:
            # If the radius increased, revert and reduce step size
            # Or just accept if it's a small increase to escape local minima
            # For simplicity, we accept and continue, as the gradient 
            # direction is heuristic.
            pass

    return best_points
# EVOLVE-BLOCK-END
