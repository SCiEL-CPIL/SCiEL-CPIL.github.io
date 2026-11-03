"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere.
    Returns a large value if the configuration is invalid."""
    pts = np.asarray(points)
    if pts.shape != (N, 3):
        return np.pi  # invalid
    if not np.all(np.isfinite(pts)):
        return np.pi
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi
    pts = pts / norms
    # Check not all in a hemisphere: convex hull should enclose origin
    try:
        hull = ConvexHull(pts)
        # Check if origin is inside the hull
        # A point is inside the convex hull if it's on the same side of all facets
        # For a hull containing the origin, the origin should be inside
        # Simple check: if the hull's volume is positive and origin is inside
        # Use the fact that for a sphere-centered hull, the covering radius
        # equals the max circumradius of facets
        # Actually, let's compute the max distance from origin to any facet
        # The covering radius is the maximum over all facets of the distance
        # from the origin to the facet plane (for facets that face away from origin)
        # Wait, the covering radius is the largest angular distance from any point
        # on the sphere to its nearest chosen point. This equals the maximum
        # circumradius of the spherical facets.
        
        # For each facet (triangle on the convex hull), compute the circumradius
        # of the spherical triangle, which corresponds to the angular covering radius
        # for that facet.
        
        # Get facet vertices
        simplices = hull.simplices  # shape (F, 3)
        max_r = 0.0
        for tri in simplices:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # The circumradius of the spherical triangle formed by a, b, c
            # The covering radius for this facet is the angle from the origin
            # to the circumcenter of the spherical triangle.
            # The circumcenter of the spherical triangle is the point on the sphere
            # equidistant from a, b, c.
            # For three points on the unit sphere, the circumcenter (if it exists)
            # is in the direction of the cross product of (b-a) and (c-a) normalized... 
            # Actually, the circumcenter of the spherical triangle is the unit vector
            # in the direction of the normal to the plane containing a, b, c,
            # but only if the triangle is acute. Let me think more carefully.
            
            # The farthest point from {a,b,c} on the sphere within the spherical triangle
            # is the circumcenter if it lies inside the triangle, otherwise it's a vertex.
            # The circumradius is the max angular distance from the circumcenter to a, b, c.
            
            # Compute the circumcenter of the spherical triangle.
            # The circumcenter is the intersection of the perpendicular bisectors.
            # For three points on the unit sphere, the circumcenter direction is:
            # n = (a × b) × (b × c) × (c × a) ... this is getting complicated.
            
            # Simpler approach: the circumradius of the spherical triangle with
            # side lengths α, β, γ is given by the formula involving the spherical
            # circumradius. But an easier numerical approach:
            # The circumcenter is the point x on the unit sphere minimizing max(|x-a|, |x-b|, |x-c|).
            # This is the center of the smallest enclosing circle on the sphere.
            
            # For three points, the circumcenter is the unit vector in the direction
            # of (a × b) × (b × c) ... let me use a different approach.
            
            # The plane through a, b, c has normal n = (b-a) × (c-a).
            # The intersection of this plane with the sphere gives a circle.
            # The center of this circle (in 3D) is the projection of the origin onto the plane.
            # The circumcenter on the sphere is along this normal direction.
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                # Collinear, use max distance to a
                max_r = max(max_r, np.arccos(np.clip(np.dot(a, b), -1, 1)),
                           np.arccos(np.clip(np.dot(b, c), -1, 1)),
                           np.arccos(np.clip(np.dot(c, a), -1, 1)))
                continue
            
            # The circumcenter is in the direction of n or -n.
            # We need the one that is "inside" the spherical triangle.
            # The circumradius is the angle between the circumcenter and any of a, b, c.
            # Since the circumcenter is equidistant from a, b, c, we just need to
            # check which sign of n gives a point closer to the triangle.
            
            # The distance from origin to the plane is |n| / |n| ... wait.
            # The plane equation: n · x = n · a
            # Distance from origin to plane: |n · a| / |n|
            # The circumcenter on the sphere is at distance |n·a|/|n| from origin in the n direction.
            # But the circumcenter must be on the unit sphere.
            
            # Actually, for three points on the unit sphere, the circumcenter of the
            # spherical triangle is the unit vector in the direction of:
            # a*(sin(2*angle_b) ... ) - this is getting too complex.
            
            # Let me just use a simple numerical approach: sample the circumcenter
            # as the unit vector in the direction of the cross product of the two
            # "bisector" directions.
            
            # Simpler: the circumcenter is the unit vector u such that u·a = u·b = u·c = cos(R)
            # where R is the circumradius. This means u is in the null space of [a-b; b-c]
            # and has unit norm. So u is parallel to (a-b) × (b-c) = (a-b) × (c-b).
            
            u = np.cross(a - b, c - b)
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-12:
                # Degenerate
                max_r = max(max_r, np.arccos(np.clip(np.dot(a, b), -1, 1)),
                           np.arccos(np.clip(np.dot(b, c), -1, 1)),
                           np.arccos(np.clip(np.dot(c, a), -1, 1)))
                continue
            u = u / u_norm
            
            # Check if u is on the same side as the triangle interior
            # The circumradius is arccos(u · a) = arccos(u · b) = arccos(u · c)
            r1 = np.arccos(np.clip(np.dot(u, a), -1, 1))
            r2 = np.arccos(np.clip(np.dot(u, b), -1, 1))
            r3 = np.arccos(np.clip(np.dot(u, c), -1, 1))
            # All should be approximately equal
            r_facet = max(r1, r2, r3)
            
            # But we need to check if the circumcenter is inside the spherical triangle.
            # If not, the covering radius for this facet is determined by a vertex or edge.
            # For now, use the max of the circumradius and the max pairwise distance.
            # Actually, if the circumcenter is outside, the farthest point is a vertex.
            # The maximum angular distance from the circumcenter to a, b, c is r_facet.
            # But if the circumcenter is outside the triangle, the farthest point on the
            # spherical triangle from the circumcenter might be a vertex.
            # For covering radius, we want the max over the entire spherical triangle
            # of the distance to the nearest of {a, b, c}.
            # This is either the circumradius (if circumcenter is inside) or the max
            # of the distances to the vertices (which is just the max pairwise angular distance).
            
            # Let's check if the circumcenter is inside the spherical triangle.
            # A point is inside a spherical triangle if it's on the same side of each
            # edge as the opposite vertex.
            inside = True
            edges = [(a, b, c), (b, c, a), (c, a, b)]
            for e1, e2, ev in edges:
                # Normal to the great circle through e1 and e2
                n_edge = np.cross(e1, e2)
                n_edge_norm = np.linalg.norm(n_edge)
                if n_edge_norm < 1e-12:
                    continue
                # Check if u and ev are on the same side
                if np.dot(n_edge, u) * np.dot(n_edge, ev) < 0:
                    inside = False
                    break
            
            if not inside:
                # The farthest point is a vertex, so the covering radius is the max
                # angular distance between any two vertices
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                r_facet = max(d_ab, d_bc, d_ca)
            
            max_r = max(max_r, r_facet)
        
        return np.degrees(max_r)
    except Exception:
        return np.pi


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 110s out of 120s, safety margin
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    
    # Gradient-free optimization using a smooth surrogate for the covering radius
    # We use a soft-max of the distances to approximate the max distance
    # Objective: minimize soft-max of angular distances from test points to nearest config point
    # But we don't have test points. Instead, we use the convex hull approach.
    
    # Alternative: use a repulsion-based energy that encourages uniform distribution
    # E = sum_{i<j} f(d_ij) where f is a function that penalizes clustering
    
    # Use the fact that minimizing the covering radius is related to maximizing
    # the minimum pairwise distance, but that's not quite right.
    
    # Let's use a direct approach: iteratively move each point to reduce the
    # covering radius by moving it away from the largest empty region.
    
    # Actually, let's use a smooth objective based on the Gram matrix.
    # The Gram matrix G = P^T P has diagonal 1 and off-diagonal cos(d_ij).
    # We want the points to be as uniformly distributed as possible.
    # One measure: minimize the variance of the off-diagonal entries of G,
    # which encourages uniform pairwise distances.
    # But this doesn't directly minimize the covering radius.
    
    # Better: use a repulsive potential. Each pair of points repels each other
    # with a force that depends on their distance. Points that are too close
    # repel more strongly, encouraging uniform distribution.
    
    def energy(pts):
        """Compute a smooth energy that correlates with covering radius."""
        # Pairwise dot products
        G = pts @ pts.T
        np.fill_diagonal(G, 0.0)
        # Use a repulsive potential: sum of exp(-alpha * d_ij) for i != j
        # where d_ij is the angular distance
        # This penalizes close points
        alpha = 5.0
        D = np.arccos(np.clip(G, -1, 1))
        # Sum of exp(-alpha * D) off-diagonal
        np.fill_diagonal(D, np.inf)
        E = np.sum(np.exp(-alpha * D))
        return E
    
    def energy_grad(pts):
        """Compute gradient of the energy."""
        G = pts @ pts.T
        D = np.arccos(np.clip(G, -1, 1))
        np.fill_diagonal(D, np.inf)
        # dE/d(pts[i]) = sum_j exp(-alpha*D_ij) * alpha * dD_ij/d(pts[i])
        # dD_ij/d(pts[i]) = -d_ij / (sin(d_ij) * d(pts[i])) ... this is getting complex
        # Let's use a simpler approach: numerical gradient or a different energy
        
        # Alternative energy: sum of (1 - cos(d_ij))^2 which is smoother
        # E = sum_{i<j} (1 - G_ij)^2
        # dE/d(pts[i]) = 2 * sum_{j != i} (1 - G_ij) * d(1-G_ij)/d(pts[i])
        # d(1-G_ij)/d(pts[i]) = -d(G_ij)/d(pts[i]) = -pts[j]
        # So dE/d(pts[i]) = -2 * sum_{j != i} (1 - G_ij) * pts[j]
        
        G = pts @ pts.T
        # For each i, grad_i = -2 * sum_{j != i} (1 - G_ij) * pts[j]
        # = -2 * (sum_j (1 - G_ij) * pts[j] - (1 - G_ii) * pts[i])
        # G_ii = 1, so (1 - G_ii) = 0
        # = -2 * sum_j (1 - G_ij) * pts[j]
        # = -2 * (sum_j pts[j] - sum_j G_ij * pts[j])
        # = -2 * (sum_j pts[j] - G @ pts)
        
        S = np.sum(pts, axis=0)  # shape (3,)
        Gpts = G @ pts  # shape (N, 3)
        grad = -2.0 * (S[np.newaxis, :] - Gpts)  # shape (N, 3)
        return grad
    
    # Use L-BFGS-B to minimize the energy
    # But we need to project back to the sphere after each step
    
    def sphere_energy_and_grad(pts_flat):
        pts = pts_flat.reshape(N, 3)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        
        G = pts @ pts.T
        S = np.sum(pts, axis=0)
        Gpts = G @ pts
        grad = -2.0 * (S[np.newaxis, :] - Gpts)
        
        # Project gradient back to tangent space
        # The constraint is ||pts[i]|| = 1, so the gradient should be projected
        # to be orthogonal to pts[i]
        for i in range(N):
            proj = np.dot(grad[i], pts[i])
            grad[i] -= proj * pts[i]
        
        # Energy
        np.fill_diagonal(G, 0.0)
        E = np.sum((1.0 - G) ** 2)
        return E, grad.flatten()
    
    # Optimize
    x0 = best_pts.flatten()
    result = minimize(sphere_energy_and_grad, x0, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success or result.fun < energy(best_pts):
        new_pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        new_pts = new_pts / norms
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr:
            best_pts = new_pts
            best_cr = new_cr
    
    # Check time
    if time.time() > deadline:
        return best_pts
    
    # Try a second optimization with a different energy
    def energy2_and_grad(pts_flat):
        pts = pts_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        
        G = pts @ pts.T
        # Use a different potential: sum of exp(-beta * d_ij)
        D = np.arccos(np.clip(G, -1, 1))
        np.fill_diagonal(D, np.inf)
        beta = 3.0
        E = np.sum(np.exp(-beta * D))
        
        # Gradient: dE/d(pts[i]) = sum_j exp(-beta*D_ij) * (-beta) * dD_ij/d(pts[i])
        # dD_ij/d(pts[i]) = -pts[j] / sin(D_ij) for j != i
        # So dE/d(pts[i]) = beta * sum_j exp(-beta*D_ij) * pts[j] / sin(D_ij)
        
        sinD = np.sin(D)
        sinD[sinD < 1e-10] = 1e-10
        W = np.exp(-beta * D) / sinD
        np.fill_diagonal(W, 0.0)
        grad = beta * (W @ pts)
        
        # Project to tangent space
        for i in range(N):
            proj = np.dot(grad[i], pts[i])
            grad[i] -= proj * pts[i]
        
        return E, grad.flatten()
    
    x0 = best_pts.flatten()
    result2 = minimize(energy2_and_grad, x0, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result2.success or result2.fun < energy2_and_grad(best_pts.flatten())[0]:
        new_pts = result2.x.reshape(N, 3)
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        new_pts = new_pts / norms
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr:
            best_pts = new_pts
            best_cr = new_cr
    
    return best_pts
# EVOLVE-BLOCK-END
