import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _circumcenter_3d(A, B, C):
    """
    Compute the circumcenter of triangle ABC on the unit sphere.
    Returns the normalized vector pointing to the circumcenter.
    If the triangle is degenerate, returns the normalized centroid.
    """
    # Circumcenter in 3D for points on unit sphere
    # The circumcenter is proportional to A + B + C + t * (A x B + B x C + C x A)
    # But for unit sphere, a simpler approach:
    # Solve for point O such that |O - A|^2 = |O - B|^2 = |O - C|^2 and O is on the plane of ABC
    # Actually, for points on unit sphere, the circumcenter direction is:
    # O ∝ A + B + C + λ * (A × B + B × C + C × A)
    # Let's use the standard formula.
    
    # Vector approach:
    # The circumcenter C satisfies:
    # (C - A) · (B - A) = (B - A)^2 / 2
    # (C - A) · (C - A) = (C - A) · (B - A) + ...
    
    # Simpler: use the fact that for unit sphere, the circumcenter is in the direction of
    # the vector perpendicular to the plane that passes through the centroid... no.
    
    # Standard formula for circumcenter of triangle with vertices a, b, c:
    # d = 2 * (a · (b × c))
    # If d is near 0, degenerate.
    # circumcenter = (|a|^2 (b × c) + |b|^2 (c × a) + |c|^2 (a × b)) / d
    # But this gives the circumcenter in 3D space, not necessarily on the sphere.
    
    # For points on the unit sphere, the circumcenter on the sphere is the normalized
    # version of the 3D circumcenter if it lies on the sphere, but actually the circumcenter
    # of a spherical triangle is the pole of the great circle... no.
    
    # Let me think differently. The covering radius for a facet is the angular distance
    # from the pole (normalized normal) to the farthest vertex. But actually, the worst point
    # in a facet is the circumcenter of the spherical triangle formed by the three vertices.
    
    # For a spherical triangle with vertices A, B, C on the unit sphere, the circumcenter
    # (the point equidistant from A, B, C on the sphere) is given by:
    # O ∝ A + B + C + k * (A × B + B × C + C × A)
    # where k is chosen so that O · A = O · B = O · C.
    
    # Actually, a simpler approach: the circumcenter of a spherical triangle is the
    # normalized vector:
    # O = normalize( (1-cos_ab) * C + (1-cos_bc) * A + (1-cos_ca) * B )
    # No, that's the incenter.
    
    # Let's use the formula:
    # For points a, b, c on unit sphere, the circumcenter is:
    # O = (a + b + c) / |a + b + c| if the triangle is acute
    # But this is only true if the circumcenter is the centroid direction, which is not generally true.
    
    # Correct approach: The circumcenter of a spherical triangle satisfies:
    # O · A = O · B = O · C = cos(R)
    # and O is a unit vector.
    # This is a system of 3 linear equations in O (with the constraint |O|=1).
    # O · A = c, O · B = c, O · C = c
    # So O is in the null space of [A-B; B-C] and satisfies O·A = c.
    
    # Let's solve: O = A + u*(B-A) + v*(C-A)
    # O·B = O·A => (A + u*(B-A) + v*(C-A))·B = A·B
    # A·B + u*(B-A)·B + v*(C-A)·B = A·B
    # u*(B·B - A·B) + v*(C·B - A·B) = 0
    # Similarly for C.
    
    # Let a = A, b = B, c = C
    # u*(1 - a·b) + v*(b·c - a·b) = 0
    # u*(a·c - a·b) + v*(1 - a·c) = 0
    
    ab = np.dot(A, B)
    bc = np.dot(B, C)
    ca = np.dot(C, A)
    
    M = np.array([[1 - ab, bc - ab],
                  [ca - ab, 1 - ca]])
    # Solve M @ [u, v]^T = 0
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if abs(det) < _EPS:
        # Degenerate, use centroid
        O = A + B + C
        O /= (np.linalg.norm(O) + _EPS)
        return O
    
    # For homogeneous system, use SVD or just take the null vector
    # The null vector is proportional to [M[1,1], -M[0,1]] or use cross product
    u = M[1,1]
    v = -M[0,1]
    
    O = A + u*(B - A) + v*(C - A)
    norm_O = np.linalg.norm(O)
    if norm_O < _EPS:
        O = A + B + C
        O /= (np.linalg.norm(O) + _EPS)
        return O
    O /= norm_O
    return O

def _hull_data(pts):
    """
    Compute convex hull and return facet data.
    Returns:
        faces: array of shape (F, 3) with vertex indices
        normals: array of shape (F, 3) with outward normals
        circumcenters: array of shape (F, 3) with circumcenter directions
        radii: array of shape (F,) with circumradius (angular distance from circumcenter to vertices)
        vertex_facets: for each vertex, list of facet indices containing it
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return None, None, None, None, None
            
        faces = faces[valid]
        A = A[valid]
        B = B[valid]
        C = C[valid]
        cross = cross[valid]
        norms = norms[valid]
        
        normals = cross / (norms + _EPS)
        
        # Ensure outward direction
        centroids = (A + B + C) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # Compute circumcenters and radii
        F = len(faces)
        circumcenters = np.zeros((F, 3))
        radii = np.zeros(F)
        
        for i in range(F):
            cc = _circumcenter_3d(A[i], B[i], C[i])
            circumcenters[i] = cc
            # Radius is angular distance from circumcenter to any vertex
            cos_r = np.clip(np.dot(cc, A[i]), -1.0, 1.0)
            radii[i] = np.arccos(cos_r)
        
        # Build vertex-to-facet mapping
        vertex_facets = [[] for _ in range(N)]
        for i, f in enumerate(faces):
            for v in f:
                vertex_facets[v].append(i)
        
        return faces, normals, circumcenters, radii, vertex_facets
    except Exception:
        return None, None, None, None, None

def _soft_hull_objective_and_grad(pts, temperature=0.5):
    """
    Compute a smooth approximation of the covering radius and its gradient.
    Uses log-sum-exp over facet circumradii.
    
    Returns:
        obj: scalar objective
        grad: array of shape (N, 3) with gradient
    """
    faces, normals, circumcenters, radii, vertex_facets = _hull_data(pts)
    if faces is None:
        return np.pi, np.zeros((N, 3))
        
    F = len(radii)
    if F == 0:
        return np.pi, np.zeros((N, 3))
    
    # Log-sum-exp for stability
    max_r = np.max(radii)
    exp_r = np.exp(radii - max_r)
    log_sum_exp = max_r + np.log(np.sum(exp_r))
    
    # The soft objective is a smooth approximation of the max
    # For gradient, we use the softmax weights
    weights = exp_r / np.sum(exp_r)
    
    # Gradient: for each facet, the gradient of the circumradius with respect to vertices
    # The circumradius R = arccos(cc · A) where cc is the circumcenter
    # dR/dA = -cc / (sqrt(1 - (cc·A)^2)) = -cc / sin(R)
    # But we also need d(cc)/dA which is complex.
    
    # Simpler approach: use the fact that moving a vertex toward the circumcenter
    # reduces the circumradius. So the gradient direction for vertex v in facet f
    # is proportional to (circumcenter_f - v) / ||circumcenter_f - v||
    
    grad = np.zeros((N, 3))
    for i in range(F):
        w = weights[i]
        cc = circumcenters[i]
        R = radii[i]
        if R < _EPS:
            continue
        sin_R = np.sin(R)
        if sin_R < _EPS:
            continue
        for v_idx in faces[i]:
            # Gradient contribution: move vertex toward circumcenter
            # The rate of change of R with respect to moving vertex v is:
            # dR/dv = -cc / sin(R) (approximately, ignoring d(cc)/dv)
            # So gradient of R w.r.t. v is -cc/sin(R)
            # To minimize R, we move v in the direction of -grad = cc/sin(R)
            # But we want to move v toward cc, so direction is (cc - v)
            # Let's use: grad_v += w * (cc - v) / ||cc - v||
            diff = cc - pts[v_idx]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > _EPS:
                grad[v_idx] += w * diff / norm_diff
    
    # Project gradient to tangent space (remove radial component)
    for i in range(N):
        radial = np.dot(grad[i], pts[i])
        grad[i] -= radial * pts[i]
    
    return log_sum_exp, grad

def _riemannian_grad_step(pts, grad, step_size):
    """
    Perform a Riemannian gradient step on the sphere.
    Uses exponential map: new_pt = cos(s)*pt + sin(s)*grad/s
    where s = step_size.
    """
    grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
    valid = grad_norms > _EPS
    grad_unit = np.zeros_like(grad)
    grad_unit[valid] = grad[valid] / grad_norms[valid]
    
    s = step_size
    new_pts = pts * np.cos(s)[:, None] + grad_unit * np.sin(s)[:, None]
    new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
    return new_pts

def _compute_hull_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    """
    faces, normals, circumcenters, radii, vertex_facets = _hull_data(pts)
    if faces is None:
        return np.pi
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
        
        # Riemannian gradient descent with adaptive step size
        step_size = 0.1
        prev_obj = np.inf
        no_improve = 0
        
        for it in range(200):
            if time.time() > deadline - 10.0:
                break
            
            obj, grad = _soft_hull_objective_and_grad(pts, temperature=0.5)
            
            if not np.isfinite(obj):
                break
                
            # Adaptive step size: decrease if objective increases
            if obj > prev_obj + 1e-6:
                step_size *= 0.8
                no_improve += 1
            else:
                if obj < prev_obj - 1e-8:
                    no_improve = 0
                step_size = min(step_size * 1.05, 0.5)
            
            prev_obj = obj
            
            # Take gradient step
            pts = _riemannian_grad_step(pts, grad, step_size)
            
            # Check convergence
            if step_size < 1e-6:
                break
        
        # Evaluate
        curr_r = _compute_hull_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
    
    # Final polish: more iterations with smaller steps on best config
    if time.time() < deadline - 5.0:
        pts = best_pts.copy()
        step_size = 0.01
        for it in range(100):
            if time.time() > deadline - 2.0:
                break
            obj, grad = _soft_hull_objective_and_grad(pts, temperature=0.3)
            if not np.isfinite(obj):
                break
            pts = _riemannian_grad_step(pts, grad, step_size)
            curr_r = _compute_hull_radius(pts)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
    
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
