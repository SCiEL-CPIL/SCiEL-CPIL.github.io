"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from math import pi, sqrt, cos, sin

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius (in degrees) of points on the unit sphere.
    
    The covering radius is the maximum over all points y on the sphere of the
    angular distance from y to the nearest point in the configuration.
    This equals the maximum circumradius of the facets of the convex hull
    (each facet checked to have an empty cap).
    """
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        return _covering_radius_grid(points)
    
    # For each facet (simplex), compute the circumradius
    # The circumradius of a facet on the unit sphere is the angular distance
    # from the sphere center to the facet plane, measured as the angle from
    # the sphere normal to the plane.
    # Actually, the covering radius contribution from a facet is the angle
    # between the facet's outward normal and the nearest point on the sphere
    # to the facet. For a simplex facet with vertices v1,...,vk, the point
    # on the sphere farthest from all configuration points within that facet
    # is along the outward normal of the facet plane, at distance = angle
    # from the origin to the plane.
    
    # For a facet with vertices, the circumradius (angular) is:
    # r = arccos(d) where d is the distance from origin to the facet plane
    # (the facet plane is at distance d from origin, and the sphere has radius 1)
    # So the angular distance from the sphere point (along normal) to the
    # nearest configuration point is arccos(d).
    
    max_cr = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        # Compute the plane normal and distance from origin
        # For a triangle (most common in 3D hull):
        if len(verts) == 3:
            v1, v2, v3 = verts
            normal = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-12:
                continue
            normal = normal / n_norm
            # Distance from origin to plane:
            d = np.dot(v1, normal)
            # The facet is on the convex hull, so the normal points outward
            # The covering radius for this facet is arccos(|d|) but we need
            # to be careful: the point on the sphere closest to the facet
            # along the normal is at angular distance arccos(d) from v1, v2, v3
            # if d > 0 (outward). If the facet is "flat" (d close to 1),
            # the covering radius is small.
            # Actually, the circumradius of the spherical triangle is the
            # angle from the circumcenter to any vertex. For a plane triangle
            # inscribed in a unit sphere, if the triangle lies in a plane at
            # distance d from the center, the circumradius of the spherical
            # triangle is arccos(d) only if the triangle is equilateral and
            # the circumcenter coincides with the normal direction.
            
            # More precisely: the covering radius contribution from a facet
            # is the maximum angular distance from any point in the spherical
            # polygon (facet) to the nearest vertex. For a spherical triangle,
            # this is the circumradius of the spherical triangle.
            
            # For a planar triangle with vertices on the unit sphere, the
            # circumradius (angular) can be computed. The point on the sphere
            # equidistant from all three vertices is along the normal to the
            # plane through the circumcenter of the planar triangle.
            
            # Let's compute the circumcenter of the planar triangle and then
            # find the corresponding point on the sphere.
            
            # Simpler approach: the covering radius for a facet is the angle
            # from the origin to the facet plane, i.e., arccos(d), but only
            # if the circumcenter of the spherical triangle is within the
            # triangle. For a convex hull facet, it should be.
            
            # Actually, the correct formula: for a spherical triangle with
            # vertices a, b, c on the unit sphere, the circumradius R satisfies
            # cos(R) = (a·b·c + 2(a·b)(b·c)(c·a) - (a·b)^2 - (b·c)^2 - (c·a)^2) / (2 * area_factor)
            # This is complex. Let's use a simpler geometric approach.
            
            # The circumcenter of the spherical triangle is the intersection
            # of the sphere with the line through the circumcenter of the
            # planar triangle in the direction of the plane normal.
            
            # Planar circumcenter:
            # Solve for point p in the plane such that |p - v1| = |p - v2| = |p - v3|
            # p = v1 + alpha*(v2-v1) + beta*(v3-v1)
            # |p-v1|^2 = |p-v2|^2 => 2*alpha*(v2-v1)·(v2-v1) + 2*beta*(v3-v1)·(v2-v1) = |v2-v1|^2
            # |p-v1|^2 = |p-v3|^2 => 2*alpha*(v2-v1)·(v3-v1) + 2*beta*(v3-v1)·(v3-v1) = |v3-v1|^2
            
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            # Circumcenter in 3D:
            pc = v1 + alpha * e1 + beta * e2
            
            # The circumcenter on the sphere is along the direction of pc
            # (if pc is not zero)
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            # The circumradius is the angle from pc/|pc| to any vertex:
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            max_cr = max(max_cr, R)
        
        elif len(verts) == 4:
            # Tetrahedron facet - compute circumradius similarly
            # Use the first 3 vertices to get an approximate circumradius
            v1, v2, v3 = verts[0], verts[1], verts[2]
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pc = v1 + alpha * e1 + beta * e2
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            max_cr = max(max_cr, R)
    
    return np.degrees(max_cr)


def _covering_radius_grid(points):
    """Fallback: estimate covering radius using a grid on the sphere."""
    # Use a fine grid
    n_theta = 180
    n_phi = 360
    theta = np.linspace(0, pi, n_theta)
    phi = np.linspace(0, 2 * pi, n_phi, endpoint=False)
    
    # Create grid
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # For each grid point, find nearest configuration point
    # Use scipy's cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(grid_points, k=1)
    
    # Angular distance from Euclidean distance:
    # cos(angle) = dot product (since both are unit vectors)
    # angle = arccos(dot)
    # But dists are Euclidean, so:
    # Euclidean distance d between two unit vectors: d^2 = 2 - 2*cos(angle)
    # cos(angle) = 1 - d^2/2
    cos_angles = 1.0 - dists ** 2 / 2.0
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
    return np.degrees(np.max(angles))


def _thomson_initialization(n, time_limit=30.0):
    """Initialize n points on the sphere using Thomson (electrostatic) minimization."""
    rng = np.random.default_rng(42)
    
    # Start with random points
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Also try Fibonacci spiral as a better starting point
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use the Fibonacci spiral as initial points
    points = fib_points.copy()
    
    # Thomson potential: sum over pairs of 1/|ri - rj|^2
    # We want to maximize the minimum distance, so minimize the Thomson potential
    
    def thomson_energy(x):
        # x is flat array of length 3n
        pts = x.reshape(n, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        # Compute pairwise distances
        # Use broadcasting for all pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Thomson potential
        energy = np.sum(1.0 / dists_sq)
        return energy
    
    def thomson_grad(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        # Gradient of Thomson potential w.r.t. x
        # d/dx [sum_{i!=j} 1/|ri-rj|^2]
        # = sum_{i!=j} -2*(ri-rj)/|ri-rj|^4
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 0.0)
        
        # Avoid division by zero
        mask = dists_sq > 1e-12
        inv_dists4 = np.zeros_like(dists_sq)
        inv_dists4[mask] = 1.0 / (dists_sq[mask] ** 2)
        
        # Gradient: for each i, sum over j != i of -2*(ri-rj)*inv_dists4[i,j]
        grad = -2.0 * np.sum(diff * inv_dists4[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space (remove radial component)
        # grad_tangent = grad - (grad·pts)*pts
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - radial * pts
        
        return grad.ravel()
    
    # Minimize using L-BFGS-B
    x0 = points.ravel()
    result = minimize(thomson_energy, x0, jac=thomson_grad, method='L-BFGS-B',
                      options={'maxiter': 500, 'gtol': 1e-8})
    
    pts = result.x.reshape(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    return pts


def _local_optimization(points, time_limit=60.0):
    """Local optimization to minimize the covering radius.
    
    Uses a gradient-based approach on the covering radius objective.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)
    
    # Compute initial covering radius
    cr_initial = _covering_radius(pts)
    
    # Use a soft version of the covering radius for smooth optimization
    # The covering radius is max over facets of circumradius
    # We can use a log-sum-exp approximation
    
    def soft_covering_radius(x, beta=10.0):
        """Soft covering radius using log-sum-exp over facet circumradii."""
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 180.0  # Large penalty
        
        # Compute circumradii of all facets
        circumradii = []
        for simplex in hull.simplices:
            verts = pts[simplex]
            if len(verts) < 3:
                continue
            
            # Use first 3 vertices for circumradius calculation
            v1, v2, v3 = verts[0], verts[1], verts[2]
            e1 = v2 - v1
            e2 = v3 - v1
            A = np.array([[np.dot(e1, e1), np.dot(e2, e1)],
                           [np.dot(e1, e2), np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1) / 2, np.dot(e2, e2) / 2])
            
            try:
                alpha, beta_val = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            
            pc = v1 + alpha * e1 + beta_val * e2
            pc_norm = np.linalg.norm(pc)
            if pc_norm < 1e-12:
                continue
            
            cos_R = np.dot(v1, pc) / pc_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            circumradii.append(R)
        
        if not circumradii:
            return 180.0
        
        circumradii = np.array(circumradii)
        
        # Log-sum-exp for soft max
        max_R = np.max(circumradii)
        if max_R == 0:
            return 0.0
        
        # Soft max: (1/beta) * log(sum(exp(beta * R_i)))
        # But we want the max, so use:
        # soft_max = max_R + (1/beta) * log(sum(exp(beta * (R_i - max_R))))
        exp_terms = np.exp(beta * (circumradii - max_R))
        soft_max = max_R + (1.0 / beta) * np.log(np.sum(exp_terms))
        
        return soft_max
    
    # Gradient-based optimization
    def objective(x):
        return soft_covering_radius(x, beta=20.0)
    
    # Use Nelder-Mead for derivative-free optimization
    # Or use a simple projected gradient method
    
    # For simplicity and robustness, use a combination of:
    # 1. Nelder-Mead on the soft covering radius
    # 2. Local adjustments based on facet analysis
    
    # First, try Nelder-Mead
    x0 = pts.ravel()
    
    # Time-limited optimization
    best_x = x0.copy()
    best_cr = cr_initial
    
    # Use L-BFGS-B with finite differences for gradient
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 500})
    
    if result.success or result.fun < best_cr:
        pts = result.x.reshape(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_x = pts.ravel().copy()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.7:
        return best_x.reshape(n, 3)
    
    # Try a few more restarts with small perturbations
    rng = np.random.default_rng(123)
    for restart in range(3):
        if time.time() - start_time > time_limit * 0.8:
            break
        
        # Perturb the current best
        perturbed = best_x.reshape(n, 3) + rng.normal(scale=0.01, size=(n, 3))
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        
        result = minimize(objective, perturbed.ravel(), method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 300})
        
        if result.success or result.fun < best_cr:
            pts = result.x.reshape(n, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_x = pts.ravel().copy()
    
    return best_x.reshape(n, 3)


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s limit
    
    # Step 1: Thomson initialization
    points = _thomson_initialization(N, time_limit=30.0)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.4:
        # Not enough time for optimization, return Thomson result
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points
    
    # Step 2: Local optimization
    points = _local_optimization(points, time_limit=time_limit - elapsed - 10.0)
    
    # Final normalization
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check for validity: no coincident points, not in a closed hemisphere
    # Check for coincident points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-6:
        # Add small perturbation to coincident points
        rng = np.random.default_rng(456)
        points = points + rng.normal(scale=1e-4, size=points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check not in a closed hemisphere:
    # The points span the sphere if no hemisphere contains all of them
    # This is equivalent to the origin being inside the convex hull
    try:
        hull = ConvexHull(points)
        # Check if origin is inside the convex hull
        # The origin is inside if for all facet normals, the origin is on
        # the same side as the interior
        # Simpler: check if the convex hull contains the origin
        from scipy.spatial import HalfspaceIntersection
        # Alternative: use linear programming or check all facets
        # For a convex hull of points on the unit sphere, the origin is inside
        # if the hull is not contained in any closed hemisphere
        # This is true if the convex hull has dimension 3 and the origin is
        # in its interior
        
        # Check by solving: is there a vector c such that c·p_i >= 0 for all i?
        # If yes, the points are in a closed hemisphere.
        # Use a simple check: compute the minimum distance from origin to
        # any facet plane. If all distances have the same sign, the origin
        # is outside.
        
        # For each facet, compute the signed distance from origin
        # If all signed distances are positive (or all negative), the origin
        # is outside the hull
        all_positive = True
        all_negative = True
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) < 3:
                continue
            v1, v2, v3 = verts[0], verts[1], verts[2]
            normal = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-12:
                continue
            normal = normal / n_norm
            d = np.dot(v1, normal)
            if d < 0:
                all_positive = False
            if d > 0:
                all_negative = False
        
        if all_positive or all_negative:
            # Points are in a closed hemisphere, need to fix
            # This shouldn't happen with Thomson initialization, but just in case
            pass
    except Exception:
        pass
    
    return points

# EVOLVE-BLOCK-END
