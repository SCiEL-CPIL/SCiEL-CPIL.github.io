"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys


def _normalize(v):
    """Normalize each row of v to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Uses the circumradius of each facet of the convex hull.
    The covering radius is the maximum circumradius over all facets.
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute-force nearest neighbor on a fine grid
        # Sample many points on the sphere
        n_samples = 5000
        # Fibonacci sphere for sampling
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # For each sample, find nearest point
        # Use chunked computation to avoid memory issues
        max_dist = 0.0
        chunk_size = 1000
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = samples[start:end]
            # Compute distances
            dists = 1.0 - np.clip(np.dot(chunk, points.T), -1.0, 1.0)
            min_dists = np.min(dists, axis=1)
            max_dist = max(max_dist, np.max(min_dists))
        return np.degrees(np.arccos(np.clip(1.0 - max_dist, -1.0, 1.0)))
    
    # For each facet, compute the circumradius
    # The circumradius of a spherical triangle with vertices a, b, c
    # can be found from the dihedral angles or by finding the point equidistant from all three.
    # Alternative: the covering radius is the max over facets of the angular distance
    # from the facet's "center" (circumcenter on the sphere) to its vertices.
    # 
    # A simpler approach: for each facet, the worst point is the circumcenter of the spherical triangle.
    # The angular radius is the angle from the circumcenter to any vertex.
    #
    # Actually, the standard approach: the covering radius equals the maximum circumradius of the
    # Delaunay triangulation facets on the sphere. For a convex polytope inscribed in the unit sphere,
    # each facet is a spherical polygon. The covering radius for that facet is the circumradius
    # of the spherical polygon.
    #
    # For a triangle with vertices on the unit sphere, the circumcenter on the sphere is the
    # normalized cross product approach... actually, the spherical circumcenter is the point
    # equidistant (angularly) from all three vertices. This is the normalization of the vector
    # that is the "circumcenter" in 3D projected to the sphere.
    #
    # For a spherical triangle, the circumradius R satisfies:
    #   sin(R) / sin(a) = sin(R) / sin(b) = sin(R) / sin(c) ... no, that's for planar.
    #
    # Let me use a different approach: for each facet, find the point on the sphere farthest
    # from the facet's plane... no, that's not right either.
    #
    # The correct approach: for each facet of the convex hull, the point on the sphere farthest
    # from the nearest configuration point within that facet's region is the spherical circumcenter.
    # The angular distance from this circumcenter to any vertex of the facet is the circumradius.
    #
    # For a triangle with unit vectors a, b, c, the spherical circumcenter n satisfies:
    #   n . a = n . b = n . c = cos(R)
    # This means n is proportional to the solution of the linear system.
    # n is in the span of a, b, c. Let n = alpha*a + beta*b + gamma*c.
    # Then n.a = alpha + beta*(b.a) + gamma*(c.a) = cos(R)
    #          alpha*(a.b) + beta + gamma*(c.b) = cos(R)
    #          alpha*(a.c) + beta*(b.c) + gamma = cos(R)
    #
    # This is a 3x3 linear system. Let's solve it.
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        if len(verts) < 3:
            continue
        # For triangles (most common in 3D convex hull)
        if len(verts) == 3:
            a, b, c = verts
            # Build the system M * [alpha, beta, gamma] = [cos(R), cos(R), cos(R)]
            # M[i,j] = verts[i] . verts[j]
            M = np.dot(verts, verts.T)
            rhs = np.ones(3)  # We'll solve for the direction, then normalize
            try:
                coeffs = np.linalg.solve(M, rhs)
                n = np.dot(coeffs, verts)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                # The angular distance from n to any vertex
                cos_R = np.dot(n, a)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            except np.linalg.LinAlgError:
                # Singular matrix, fallback
                continue
        else:
            # For higher-dimensional facets (rare in 3D), use a brute-force approach
            # Sample points in the facet's spherical polygon and find the max distance
            # For simplicity, just check the vertices and their midpoints
            pass
    
    return np.degrees(max_circumradius)


def _objective_and_grad(x):
    """
    Objective: covering radius in radians (minimize).
    x is a flat array of 3*N coordinates.
    """
    points = x.reshape(N, 3)
    points = _normalize(points)
    
    # Compute covering radius
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, return a large value
        return 10.0, np.zeros_like(x)
    
    max_circumradius = 0.0
    # Track which facet gives the max to compute gradient
    max_facet = None
    
    for idx, simplex in enumerate(hull.simplices):
        verts = points[simplex]
        if len(verts) != 3:
            continue
        a, b, c = verts
        M = np.dot(verts, verts.T)
        rhs = np.ones(3)
        try:
            coeffs = np.linalg.solve(M, rhs)
            n = np.dot(coeffs, verts)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            cos_R = np.dot(n, a)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_circumradius:
                max_circumradius = R
                max_facet = (idx, simplex, n, verts)
        except np.linalg.LinAlgError:
            continue
    
    # Gradient computation is complex; use finite differences for now
    # Or just use the value and let L-BFGS-B use numerical gradients
    return max_circumradius, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_points = _normalize(best_points)
    best_score = _covering_radius(best_points)
    
    # Try multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate random starting points
        if restart == 0:
            x0 = best_points.flatten()
        else:
            # Random points on sphere
            x0 = rng.normal(size=(N, 3))
            x0 = _normalize(x0)
            x0 = x0.flatten()
        
        # Use L-BFGS-B with numerical gradients
        # We'll use a simpler objective: minimize the sum of squared max distances
        # Actually, let's use the covering radius directly with L-BFGS-B
        
        # For L-BFGS-B, we need a smooth objective. The covering radius is not smooth.
        # Use a soft-max approximation: sum of exp(beta * R_i) / beta for large beta
        # But this is also not trivial to compute.
        
        # Alternative: use a penalty-based approach or just use the covering radius
        # with a wrapper that handles non-smoothness.
        
        # Let's try a different approach: minimize the maximum of pairwise distances
        # to the Voronoi cell centers. But that's complex.
        
        # Simpler: use scipy.optimize.minimize with method='L-BFGS-B' and 
        # let it use numerical gradients. The objective is the covering radius.
        
        def obj(x):
            points = x.reshape(N, 3)
            points = _normalize(points)
            return _covering_radius(points)
        
        try:
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if result.success or result.fun < best_score:
                new_points = _normalize(result.x.reshape(N, 3))
                new_score = _covering_radius(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            continue
    
    # Final polish: try a few more iterations with tighter tolerance
    if time.time() < deadline - 5.0:
        def obj(x):
            points = x.reshape(N, 3)
            points = _normalize(points)
            return _covering_radius(points)
        
        try:
            result = minimize(
                obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            if result.fun < best_score:
                best_points = _normalize(result.x.reshape(N, 3))
                best_score = _covering_radius(best_points)
        except Exception:
            pass
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
