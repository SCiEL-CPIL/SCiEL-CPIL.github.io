"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import sys

# Deadline guard: 150s wall-clock, return best before ~145s
_DEADLINE = 145.0


def _covering_radius(pts):
    """Compute the covering radius (max angular distance to nearest point) in radians.
    Uses the convex hull: the covering radius is the max circumradius of the spherical
    facets. For a point configuration on the unit sphere, the largest empty cap radius
    equals the max over hull facets of the circumradius of that facet's vertices.
    """
    pts = np.asarray(pts, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, None]
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force on a fine grid
        # Use a reasonably fine icosahedral subdivision
        # For speed, sample ~50k points
        n_theta = 200
        n_phi = 400
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        samples = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        # Compute min distance to each sample
        # pts shape (N,3), samples shape (M,3)
        # dist = sqrt(2 - 2*dot)
        dot = samples @ pts.T  # (M, N)
        dists = np.sqrt(np.maximum(0, 2 - 2 * dot))
        return np.max(dists.min(axis=1))
    
    # For each facet, compute the circumradius (in angular terms)
    # The facet is a spherical polygon. The circumradius of a spherical triangle
    # (or polygon) is the angular distance from the circumcenter to a vertex.
    # For a planar convex hull facet with vertices v1,...,vk, the circumcenter in 3D
    # is the point equidistant from all vertices. The angular circumradius is
    # arccos(dot(circumcenter, v_i)) for any vertex.
    
    # Actually, for the covering radius, we need the maximum over all points on the
    # sphere of the distance to the nearest site. This equals the maximum over all
    # Voronoi cell vertices (which are circumcenters of hull facets) of the distance
    # to the nearest site.
    
    # For a hull facet with vertices at indices, compute the circumcenter.
    max_cr = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]  # (k, 3) where k is typically 3 for tetrahedral hull
        # Compute circumcenter of the spherical polygon formed by these vertices
        # For a triangle on the sphere, the circumcenter is the normalized solution
        # to the system: |c - v_i|_2 = const for all i, with |c|=1
        # Equivalently, c is proportional to the cross product approach or
        # solve (v_i - v_j) . c = 0 for all i,j, |c|=1
        
        if len(verts) == 3:
            # Triangle: circumcenter is along the normal to the plane, at the
            # appropriate distance. Actually for a spherical triangle, the circumcenter
            # is the unit vector in the direction of the "spherical circumcenter".
            # For a planar triangle with vertices a,b,c on the unit sphere,
            # the circumcenter of the spherical triangle is the unit vector
            # proportional to a*(b×c) + b*(c×a) + c*(a×b) ... no that's not right.
            # 
            # The circumcenter of a spherical triangle is the point equidistant from
            # all three vertices. It lies on the polar great circles of the sides.
            # A simpler approach: the circumcenter is the normalized vector
            # (a × b) × (b × c) ... let me think differently.
            #
            # For three points a, b, c on the unit sphere, the circumcenter is:
            # c = normalize( (a × b) × (b × c) ) ... no.
            #
            # Actually, the circumcenter of a spherical triangle with vertices a,b,c
            # is the unit vector u such that u·a = u·b = u·c. This means u is
            # orthogonal to (a-b), (b-c), (c-a). So u is parallel to the normal of
            # the plane through a,b,c, which is (a-b)×(b-c).
            # But we also need to pick the correct sign (the one on the same side
            # as the triangle interior).
            #
            # Wait, that's the circumcenter of the planar triangle. For the spherical
            # triangle, the circumcenter is indeed along the normal to the plane of
            # the three points. The angular circumradius is arccos(|u·a|) where u
            # is the unit normal (choosing the sign so that u·a > 0).
            
            n = np.cross(verts[0] - verts[1], verts[1] - verts[2])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            u = n / n_norm
            # Choose sign so that u points in the same general direction as the vertices
            if np.dot(u, verts[0]) < 0:
                u = -u
            cr = np.arccos(np.clip(np.dot(u, verts[0]), -1.0, 1.0))
            if cr > max_cr:
                max_cr = cr
        elif len(verts) == 4:
            # Quadrilateral: compute circumradius by checking if it's cyclic on sphere
            # For a spherical quadrilateral, the circumcenter is the point equidistant
            # from all 4 vertices. This is more complex. We can use the fact that
            # if the 4 points are coplanar (which they are, being a hull facet),
            # the circumcenter is the same as the planar circumcenter projected.
            # For a planar quadrilateral, the circumcenter exists only if it's cyclic.
            # In general, we can use the max of the circumradii of the two triangles
            # formed by a diagonal, or better, solve for the point equidistant from all.
            #
            # Simpler: use the fact that the circumcenter of the spherical polygon
            # is the same as the circumcenter of any three non-collinear vertices
            # if the polygon is cyclic. Otherwise, use a least-squares approach.
            #
            # For robustness, let's use the first three vertices to get an approximate
            # circumcenter, then refine.
            n = np.cross(verts[0] - verts[1], verts[1] - verts[2])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            u = n / n_norm
            if np.dot(u, verts[0]) < 0:
                u = -u
            # Check if this u is approximately equidistant from all vertices
            dots = np.array([np.dot(u, v) for v in verts])
            if np.max(np.abs(dots - dots[0])) < 1e-10:
                cr = np.arccos(np.clip(dots[0], -1.0, 1.0))
            else:
                # Not cyclic; use the max angular distance from u to any vertex
                cr = np.max(np.arccos(np.clip(dots, -1.0, 1.0)))
            if cr > max_cr:
                max_cr = cr
        else:
            # General polygon: use least-squares to find the point equidistant
            # from all vertices in the angular sense.
            # Minimize sum (arccos(u·v_i) - r)^2 ... this is complex.
            # Instead, find u that minimizes the max of arccos(u·v_i).
            # Use the normal to the plane as a starting point.
            n = np.cross(verts[0] - verts[1], verts[1] - verts[2])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            u = n / n_norm
            if np.dot(u, verts[0]) < 0:
                u = -u
            dots = np.array([np.dot(u, v) for v in verts])
            cr = np.max(np.arccos(np.clip(dots, -1.0, 1.0)))
            if cr > max_cr:
                max_cr = cr
    
    return max_cr


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_step(pts, lr=0.01, iters=1000):
    """
    Perform gradient-based optimization to minimize the covering radius.
    Uses a smooth surrogate: soft-max of the angular distances.
    """
    start_time = time.time()
    pts = np.array(pts, dtype=np.float64)
    
    # Use a soft-max parameter for smoothness
    beta = 50.0  # Higher = closer to true max
    
    def objective_and_grad(x):
        if time.time() > _DEADLINE:
            return None, None
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1)
        p = p / norms[:, None]
        
        # Compute angular distances between all pairs
        dot = p @ p.T  # (N, N)
        np.fill_diagonal(dot, 2.0)  # Avoid division issues on diagonal
        dists = np.arccos(np.clip(dot, -1.0, 1.0))
        
        # Soft-max: log-sum-exp for the max
        # For each point y (we'll use the points themselves as a proxy, but really
        # we need to consider the whole sphere). 
        # Better approach: the covering radius is determined by the Voronoi cell
        # vertices. But for a smooth optimization, we can use a grid-based approach
        # or the hull-based approach.
        #
        # Actually, let's use a different strategy: minimize the sum of squared
        # distances to the nearest point for a set of test points on the sphere.
        # This is a smooth approximation to the covering radius.
        
        # Generate test points on the sphere (fixed, not part of optimization)
        # Use a well-distributed set
        n_test = 2000
        # Use Fibonacci spiral for test points
        j = np.arange(n_test) + 0.5
        z_t = 1.0 - 2.0 * j / n_test
        phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
        r_t = np.sqrt(1.0 - z_t * z_t)
        test_pts = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
        
        # Compute distances from each test point to each site
        dot_t = test_pts @ p.T  # (n_test, N)
        dists_t = np.arccos(np.clip(dot_t, -1.0, 1.0))
        
        # For each test point, find the nearest site
        min_dists = dists_t.min(axis=1)  # (n_test,)
        
        # Objective: soft-max of min_dists
        # Use log-sum-exp for numerical stability
        m = min_dists.max()
        log_sum_exp = m + np.log(np.sum(np.exp(beta * (min_dists - m))))
        obj = log_sum_exp / beta
        
        # Gradient: this is complex. Let's use finite differences for simplicity
        # but that's slow. Instead, let's use a simpler gradient-free approach.
        return obj, None
    
    # Use Nelder-Mead or Powell for gradient-free optimization
    # This is more robust for non-smooth objectives
    
    x0 = pts.ravel()
    
    def obj_func(x):
        if time.time() > _DEADLINE:
            return 1e10
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1)
        p = p / norms[:, None]
        return _covering_radius(p)
    
    # Use Powell's method which doesn't need gradients
    try:
        result = minimize(obj_func, x0, method='Powell', 
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.success or result.fun < obj_func(x0):
            p_opt = result.x.reshape(-1, 3)
            norms = np.linalg.norm(p_opt, axis=1)
            p_opt = p_opt / norms[:, None]
            return p_opt
    except Exception:
        pass
    
    return pts


def construct():
    start_time = time.time()
    
    # Initial configuration
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    
    # Try multiple optimization approaches
    # 1. Powell's method
    try:
        if time.time() < _DEADLINE - 30:
            optimized = _gradient_step(best_pts)
            cr = _covering_radius(optimized)
            if cr < best_cr:
                best_pts = optimized
                best_cr = cr
    except Exception:
        pass
    
    # 2. Local search: perturb points and accept if improvement
    if time.time() < _DEADLINE - 10:
        # Simple simulated annealing or hill climbing
        current_pts = best_pts.copy()
        current_cr = best_cr
        step_size = 0.1
        n_iter = 5000
        
        for it in range(n_iter):
            if time.time() > _DEADLINE - 5:
                break
            
            # Pick a random point and perturb it
            idx = np.random.randint(N)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            new_pts = current_pts.copy()
            new_pts[idx] = current_pts[idx] + step_size * direction
            # Project back to unit sphere
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_cr = _covering_radius(new_pts)
            
            if new_cr < current_cr:
                current_pts = new_pts
                current_cr = new_cr
                if current_cr < best_cr:
                    best_pts = current_pts.copy()
                    best_cr = current_cr
            else:
                # Occasionally accept worse solutions (simulated annealing)
                if np.random.rand() < np.exp(-10 * (new_cr - current_cr)):
                    current_pts = new_pts
                    current_cr = new_cr
            
            # Adaptive step size
            if it % 500 == 0:
                step_size *= 0.8
    
    # Final polish: small local optimization
    if time.time() < _DEADLINE - 5:
        try:
            x0 = best_pts.ravel()
            
            def obj_func(x):
                if time.time() > _DEADLINE:
                    return 1e10
                p = x.reshape(-1, 3)
                norms = np.linalg.norm(p, axis=1)
                p = p / norms[:, None]
                return _covering_radius(p)
            
            result = minimize(obj_func, x0, method='Powell',
                             options={'maxiter': 50, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_cr:
                p_opt = result.x.reshape(-1, 3)
                norms = np.linalg.norm(p_opt, axis=1)
                p_opt = p_opt / norms[:, None]
                best_pts = p_opt
                best_cr = result.fun
        except Exception:
            pass
    
    # Ensure output is valid
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, None]
    
    return best_pts
# EVOLVE-BLOCK-END
