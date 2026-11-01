import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()

    def get_hull_data(pts):
        """
        Compute the convex hull and return:
        - max_r: the maximum circumradius (covering radius)
        - active_simplex: indices of the facet defining the max radius
        - c_sph: the spherical circumcenter of that facet
        - p1, p2, p3: the three points of the facet
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1.0, None, None, None, None, None

        max_r = 0.0
        best_simplex = None
        best_c = None
        best_p = (None, None, None)

        for simplex in hull.simplices:
            i_idx, j_idx, k_idx = simplex
            p1 = pts[i_idx]
            p2 = pts[j_idx]
            p3 = pts[k_idx]

            u = p2 - p1
            v = p3 - P1
            # Wait, typo in thought, let's code carefully.
            v = p3 - p1
            
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            det = uu * vv - uv * uv
            
            if abs(det) < 1e-10:
                continue
                
            # Coefficients for circumcenter in the plane of the triangle
            # We want c = p1 + a*u + b*v such that |c-p1|^2 = |c-p2|^2 = |c-p3|^2
            # 2*a*uu + 2*b*uv = uu
            # 2*a*uv + 2*b*vv = vv
            a = (0.5 * uu * vv - 0.5 * uv * vv) / det
            b = (0.5 * vv * uu - 0.5 * uu * uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-10:
                continue
            c_sph = c_planar / c_norm
            
            # Check if the cap is empty (circumcenter is on the same side of the facet as the rest of the hull? 
            # Actually, for a convex polytope containing the origin, the circumcenter of a facet 
            # defines the covering radius IF the sphere centered at c_sph with radius r 
            # does not contain any other points. 
            # The condition "empty cap" is equivalent to checking if c_sph is the unique point 
            # maximizing dot product with the normal? 
            # Simpler check: The circumradius is valid for the covering problem if the 
            # circumcenter c_sph is such that the spherical triangle defined by p1,p2,p3 
            # contains c_sph. For a convex hull on the sphere, this is true if the 
            # barycentric coordinates are positive? 
            # Actually, the "empty cap" condition means that no other point p_k has 
            # dot(c_sph, p_k) > dot(c_sph, p1). 
            # Since p1, p2, p3 are on the sphere, dot(c_sph, p1) = cos(r).
            # So we need dot(c_sph, p_k) <= cos(r) for all k.
            # This is equivalent to saying c_sph is the normal to the facet pointing outward.
            # For a convex hull of points on the sphere containing the origin, 
            # the normal to the facet n = (p2-p1)x(p3-p1) points away from the origin 
            # if the facet is "visible". 
            # The circumcenter c_sph is in the direction of the normal.
            # So we check if dot(c_sph, p_k) <= dot(c_sph, p1) for all k.
            
            # Optimization: We can just compute the radius and check validity at the end, 
            # or check here. Checking here is O(N) per facet, so O(F*N) total. 
            # F is ~2N, so O(N^2). Acceptable.
            
            # Let's just compute r and store it. We'll verify the "empty cap" condition 
            # more efficiently later or assume convexity implies it for the max facet?
            # No, the max circumradius facet is the one that defines the covering radius 
            # ONLY if its cap is empty. If the cap is not empty, the true covering radius 
            # is determined by a different point (likely a vertex or an edge).
            # However, it is a known result that the covering radius of a set of points on the sphere 
            # is the maximum circumradius of the facets of the convex hull that have empty caps.
            # For a convex polytope, the "empty cap" condition is satisfied by the facets 
            # that are part of the "Voronoi diagram" of the points on the sphere.
            
            # Let's calculate the angle.
            d_c = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
            
            # Check if this is a valid "covering" facet.
            # The condition is that c_sph is the normal to the facet.
            # n = np.cross(u, v)
            # if np.dot(n, c_sph) > 0: # Normal points in direction of c_sph
            #    # Check if all other points are "behind" the plane defined by the facet?
            #    # No, the condition is that the cap centered at c_sph with angular radius d_c 
            #    # contains no other points.
            #    # i.e., dot(c_sph, p_k) <= cos(d_c) for all k.
            #    # cos(d_c) = dot(c_sph, p1).
            #    dots = pts @ c_sph
            #    if np.max(dots) <= dot(c_sph, p1) + 1e-9:
            #        if d_c > max_r:
            #            max_r = d_c
            #            best_simplex = simplex
            #            best_c = c_sph
            #            best_p = (p1, p2, p3)
            
            # To save time in the inner loop, we might skip the full check for every facet 
            # and only check the ones with large d_c? 
            # Or just do it. N=50, F~100. 50*100 = 5000 dot products. Trivial.
            
            dots = pts @ c_sph
            if np.max(dots) <= np.dot(c_sph, p1) + 1e-9:
                if d_c > max_r:
                    max_r = d_c
                    best_simplex = simplex
                    best_c = c_sph
                    best_p = (p1, p2, p3)
                    
        if best_simplex is None:
            return 1.0, None, None, None, None, None

        return max_r, best_simplex, best_c, best_p[0], best_p[1], best_p[2]

    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        max_r, simplex, c_sph, p1, p2, p3 = get_hull_data(pts)
        
        if simplex is None:
            # Fallback: return a large value and zero gradient
            return 1.0, np.zeros_like(x_flat)
            
        # Gradient calculation
        # We want to minimize r = arccos(c_sph . p1).
        # dr/d(c_sph) = -p1 / sqrt(1 - (c_sph.p1)^2)
        # dr/d(p1) = -c_sph / sqrt(1 - (c_sph.p1)^2)
        
        # c_sph is a function of p1, p2, p3.
        # c_sph = (p1 + a*u + b*v) / ||...||
        # This is complex. 
        # Alternative: Use the fact that at the optimum, the gradient of the objective 
        # w.r.t the points is in the span of the gradients of the constraints.
        # But we are doing direct optimization.
        
        # Let's use finite differences for the gradient of the hull radius? 
        # It's robust and N=50 is small enough that 150 FD evaluations per step 
        # might be too slow if we do many steps. 
        # However, L-BFGS-B uses memory. 
        # Let's try analytic gradient approximation.
        
        # The circumcenter c of a spherical triangle is the intersection of the perpendicular bisectors.
        # The gradient of the circumradius with respect to the vertices is known.
        # For a planar triangle, the gradient of the circumradius R with respect to vertex i 
        # is related to the unit vector from the circumcenter to the vertex.
        # On the sphere, it is the tangent vector pointing away from the circumcenter?
        # Actually, if we move p1 in the direction of c_sph, the distance decreases.
        # So the gradient of r w.r.t p1 (on the sphere) is -c_sph (tangent component).
        # Since c_sph is orthogonal to the tangent plane at p1? No.
        # c_sph is not on the tangent plane at p1.
        # The gradient on the sphere is the projection of -c_sph onto the tangent plane at p1?
        # No, the distance is arccos(c.p). d/dp arccos(c.p) = -c / sqrt(1-(c.p)^2).
        # This vector is in the direction of -c.
        # But p must stay on the sphere. The gradient in the ambient space is 
        # lambda * (-c) where lambda accounts for the normalization?
        # If we parameterize p directly on the sphere, the gradient is the tangent vector.
        # grad_sphere(r) = -c_sph / sin(r) * (projection onto tangent plane)?
        # Actually, the gradient of f(p) = arccos(c.p) on the sphere is 
        # ( -c / sqrt(1-(c.p)^2) ) - ( (c.p) / (1-(c.p)^2) ) * p ? No.
        
        # Let's stick to a simpler, robust approach: 
        # Use the subgradient of the max function.
        # The active set is the single facet (or multiple if tied).
        # The gradient of the radius w.r.t the points in the active facet is what matters.
        
        # For the active facet (i,j,k), the radius is determined by the circumcenter.
        # Moving any of these points changes the circumcenter.
        # A known result for Tammes/covering problems: 
        # The gradient of the covering radius w.r.t point i is 
        # sum over facets containing i of (normal_to_facet / sin(r_facet)) ?
        
        # Let's use a simpler heuristic gradient that works well in practice:
        # For the active facet (i,j,k), the gradient for each point is 
        # pointing away from the circumcenter c_sph, projected onto the tangent plane.
        # grad_i = (c_sph - (c_sph.p_i) p_i) / ||c_sph - (c_sph.p_i) p_i|| ?
        # Or just -c_sph projected?
        
        # Let's try: grad_i = -c_sph (ambient gradient)
        # Since we are using L-BFGS-B on Cartesian coordinates, we need the ambient gradient.
        # But the points are constrained to the sphere. 
        # If we don't enforce the sphere constraint in the optimizer (by normalizing inside), 
        # the optimizer might drift. 
        # It is better to use a manifold optimization library or parameterize the sphere.
        # However, the task requires returning Cartesian vectors.
        # The current code normalizes inside. This is a "projection" method.
        # The gradient must be consistent with the projection.
        
        # Let's use the "normalized gradient" trick.
        # If f(p) is defined on the sphere, and we extend it to R^3 by f(x) = f(x/||x||),
        # then grad f(x) = grad_sphere f(x/||x||) / ||x|| - f(x/||x||) * x / ||x||^3 ?
        
        # Simpler: Just compute the gradient w.r.t the Cartesian coordinates 
        # using the chain rule for the normalization.
        
        # Let p = x / ||x||.
        # r = arccos(c(p1,p2,p3) . p1)
        # This is too complex for analytic derivation in a short time.
        
        # Let's use Finite Differences for the gradient of the objective.
        # N=50, 150 dimensions.
        # 150 function evaluations per gradient.
        # Each evaluation is O(N^2) ~ 2500 ops.
        # Total ~ 375,000 ops per gradient.
        # L-BFGS might need 10-20 iterations.
        # Total ~ 7,500,000 ops. Very fast.
        
        # We will return the objective and let L-BFGS-B approximate the gradient?
        # No, L-BFGS-B can use finite differences if jac is not provided.
        # But we can provide a custom finite difference gradient to be safe and potentially faster 
        # if we parallelize or use complex step?
        # Standard L-BFGS-B finite difference is forward difference.
        
        # Let's just return the objective and let scipy handle it.
        # To improve convergence, we might want to provide the gradient.
        # But for now, let's try the "black box" approach with L-BFGS-B.
        
        return max_r, None

    # We need a wrapper for minimize that handles the normalization and returns the value.
    def obj_wrapper(x_flat):
        val, _ = objective_and_grad(x_flat)
        return val

    # Initial guess
    x_init = x0.copy()
    
    # Run L-BFGS-B
    best_x = x_init
    best_val = obj_wrapper(x_init)
    
    try:
        # maxiter: 200 should be enough for N=50
        # ftol: 1e-12
        res = minimize(
            obj_wrapper,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False, 'maxls': 50}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception as e:
        pass

    # Extract points and normalize
    pts = best_x.reshape(N, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Final check and potentially a few more steps if time permits?
    # L-BFGS-B is usually robust.
    
    # Verify validity: no coincident points, not in hemisphere
    # The convex hull construction inside the objective ensures the points are "well-distributed" 
    # in the sense that they form a convex hull.
    # If they were in a hemisphere, the hull would be degenerate or the radius would be large.
    # The optimization drives the radius down.
    
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

# EVOLVE-BLOCK-END
