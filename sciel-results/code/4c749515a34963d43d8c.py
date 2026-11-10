"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools


def _compute_circumradii(points):
    """Compute circumradius of each facet of the convex hull of points on unit sphere.
    Returns array of circumradii (angular distance in radians) for each facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([])
    
    facets = hull.simplices  # each row is 3 indices for a triangular facet
    n_facets = len(facets)
    if n_facets == 0:
        return np.array([])
    
    circumradii = np.zeros(n_facets)
    
    for i, tri in enumerate(facets):
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        
        # Compute circumradius of triangle on unit sphere
        # The circumcenter is the point equidistant from all three vertices on the sphere
        # For a spherical triangle, the circumradius is the angular distance from circumcenter to any vertex
        
        # Method: find the point on the sphere equidistant from p0, p1, p2
        # This is the intersection of the sphere with the plane perpendicular to the plane of the triangle
        # passing through the origin (since all points are on unit sphere)
        
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
        # can be found as follows: the three points define a plane. The circumcenter is
        # the point on the unit sphere that lies in the plane perpendicular to the triangle's
        # plane passing through the origin, and is equidistant from all three points.
        
        # Simpler: use the formula for circumradius of a triangle with sides a, b, c
        # R = abc / (4 * area)
        # But on the sphere, we need the spherical circumradius.
        
        # Alternative: the circumradius in angular distance is the angle at the center
        # from the circumcenter to a vertex.
        
        # Let's compute using the cross product method:
        # The circumcenter C satisfies |C - p0|^2 = |C - p1|^2 = |C - p2|^2
        # and |C| = 1.
        
        # Expand: |C|^2 - 2 C·p0 + |p0|^2 = |C|^2 - 2 C·p1 + |p1|^2
        # Since |C|^2 = 1 and |pi|^2 = 1:
        # -2 C·p0 = -2 C·p1 => C·p0 = C·p1
        # Similarly C·p0 = C·p2
        
        # So C is in the null space of [p0 - p1; p0 - p2] (dot product constraints)
        # and |C| = 1.
        
        # The two constraints give us that C is parallel to (p1 - p0) × (p2 - p0)
        # Wait, that's the normal to the triangle plane. Let me reconsider.
        
        # C·p0 = C·p1 and C·p0 = C·p2
        # C·(p0 - p1) = 0 and C·(p0 - p2) = 0
        # So C is orthogonal to both (p0-p1) and (p0-p2), meaning C is parallel to their cross product.
        
        n = np.cross(p0 - p1, p0 - p2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            # Degenerate triangle
            circumradii[i] = np.pi
            continue
        
        # C is parallel to n, and |C| = 1
        # But we need to check the sign: C should be on the same side as the triangle
        # The circumcenter of the spherical triangle should be such that the cap is empty.
        # Let's try both directions and pick the one that gives a valid circumcenter.
        
        c1 = n / n_norm
        c2 = -c1
        
        # For each candidate, compute the angular distance to p0
        # The circumradius is the max angular distance to any vertex (should be equal if correct)
        
        # Actually, for a spherical triangle, the circumradius is well-defined.
        # Let's compute the angle: cos(theta) = C · p0
        # theta = arccos(C · p0)
        
        dot1 = np.clip(np.dot(c1, p0), -1.0, 1.0)
        theta1 = np.arccos(dot1)
        
        dot2 = np.clip(np.dot(c2, p0), -1.0, 1.0)
        theta2 = np.arccos(dot2)
        
        # The circumradius should be the same for all three vertices.
        # Pick the direction that gives a consistent (smaller or appropriate) radius.
        # Typically, we want the circumcenter to be inside the cap, so the radius should be
        # the angular distance from the circumcenter to the vertices.
        
        # Let's verify with all three vertices for c1:
        d0 = np.clip(np.dot(c1, p0), -1.0, 1.0)
        d1 = np.clip(np.dot(c1, p1), -1.0, 1.0)
        d2 = np.clip(np.dot(c1, p2), -1.0, 1.0)
        
        # The circumradius is arccos of these dot products (should be equal)
        r0 = np.arccos(d0)
        r1 = np.arccos(d1)
        r2 = np.arccos(d2)
        
        # If they're approximately equal, c1 is the circumcenter
        if np.max([r0, r1, r2]) - np.min([r0, r1, r2]) < 1e-10:
            circumradii[i] = r0
        else:
            # Try c2
            d0 = np.clip(np.dot(c2, p0), -1.0, 1.0)
            d1 = np.clip(np.dot(c2, p1), -1.0, 1.0)
            d2 = np.clip(np.dot(c2, p2), -1.0, 1.0)
            r0 = np.arccos(d0)
            r1 = np.arccos(d1)
            r2 = np.arccos(d2)
            
            if np.max([r0, r1, r2]) - np.min([r0, r1, r2]) < 1e-10:
                circumradii[i] = r0
            else:
                # Fallback: use the average
                circumradii[i] = (r0 + r1 + r2) / 3.0
    
    return circumradii


def _objective(x, N):
    """Objective: max circumradius (in radians) of the convex hull facets.
    x is a flat array of 3*N values representing N unit vectors.
    """
    points = x.reshape(N, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    points = points / norms
    
    circumradii = _compute_circumradii(points)
    if len(circumradii) == 0:
        return np.pi  # Degenerate
    
    return np.max(circumradii)


def _optimize_from_start(start_points, N, time_budget):
    """Optimize from a starting configuration using L-BFGS-B.
    Returns the optimized points.
    """
    start_time = time.time()
    
    x0 = start_points.flatten()
    
    # Use L-BFGS-B with bounds to keep points away from origin
    # Actually, we'll normalize in the objective, so no bounds needed
    # But we need to handle the non-smooth max operation
    
    result = None
    best_val = _objective(x0, N)
    best_x = x0.copy()
    
    # Try L-BFGS-B
    try:
        res = minimize(
            _objective,
            x0,
            args=(N,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_val:
            # Normalize the result
            points = res.x.reshape(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-15, 1.0, norms)
            points = points / norms
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return best_x.reshape(N, 3)
    
    # Try Nelder-Mead as a second pass
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 5:
        try:
            res2 = minimize(
                _objective,
                best_x,
                args=(N,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    # Final normalization
    points = best_x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    points = points / norms
    
    return points


def _fibonacci_spiral(N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(N, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    points = rng.randn(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _icosahedron_based(N):
    """Generate points based on an icosahedron subdivision."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide each face into smaller triangles
    # For N=50, we need to add 38 more points
    # Simple approach: use Fibonacci as fallback
    return _fibonacci_spiral(N)


def construct():
    """Construct 50 points on the unit sphere to minimize covering radius."""
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe (150s limit)
    
    best_points = None
    best_score = np.pi  # Worst possible is pi radians
    
    # Generate multiple starting configurations
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # Fibonacci with different offset
    starts.append(_fibonacci_spiral(N, offset=0.0))
    
    # Random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline - 60:
            break
        starts.append(_random_points(N, seed=seed))
    
    # Optimize from each start
    for idx, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        # Allocate time per optimization
        remaining = deadline - time.time()
        n_starts_remaining = len(starts) - idx
        time_budget = max(5.0, (remaining - 10) / n_starts_remaining)
        
        try:
            optimized = _optimize_from_start(start, N, time_budget)
            
            # Evaluate
            score = _objective(optimized.flatten(), N)
            
            if score < best_score:
                best_score = score
                best_points = optimized.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to Fibonacci
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_points = best_points / norms
    
    # Ensure no coincident points
    # Check for very close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-8:
                # Nudge point j slightly
                best_points[j] = best_points[j] + 1e-6 * np.random.randn(3)
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
