import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 145.0  # Safety margin from 150s limit

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii (vectorized).
    Returns: hull, facets, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None, None
    
    facets = hull.simplices
    n_facets = len(facets)
    if n_facets == 0:
        return hull, facets, np.zeros((0, 3)), np.zeros(0), None
        
    a = points[facets[:, 0]]  # (F, 3)
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # Vectorized circumcenter computation
    # Normal to the plane defined by a, b, c
    v1 = a - b
    v2 = b - c
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1)
    
    # Handle degenerate facets (collinear points)
    valid = norm_n > 1e-12
    
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    if np.any(valid):
        # Normalize normal vector
        cc = n[valid] / norm_n[valid, np.newaxis]
        
        # Ensure circumcenter faces outward (toward 'a')
        # The circumcenter C satisfies dot(C, a) = dot(C, b) = dot(C, c)
        # For a point on the sphere, the circumcenter of the triangle formed by 3 points
        # on the sphere lies on the line passing through the origin and the normal to the plane.
        # Specifically, C = k * n_hat.
        # We need to find k such that |C - a|^2 = |C - b|^2 = |C - c|^2.
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
        # is the intersection of the perpendicular bisectors.
        # A simpler formula for the circumcenter of a triangle on the unit sphere:
        # C = (a * sin(A) + b * sin(B) + c * sin(C)) / sin(A+B+C) ... this is complex.
        
        # Alternative: Solve for C = x*a + y*b + z*c? No, C is not necessarily in the plane spanned by a,b,c relative to origin?
        # Wait, a, b, c are on the sphere. The circumcenter of the spherical triangle is the point on the sphere
        # equidistant from a, b, c.
        # Let C be the circumcenter. C is in the span of a, b, c? Yes, because the perpendicular bisector of chord ab passes through origin and is perpendicular to ab.
        # Actually, the set of points equidistant from a and b is the plane passing through origin and perpendicular to a-b.
        # So C is the intersection of 3 such planes? No, 2 planes define a line through origin. The 3rd is redundant if consistent.
        # The line is along the normal to the plane abc.
        # So C = k * n_hat.
        # We require dot(C, a) = dot(C, b).
        # k * dot(n_hat, a) = k * dot(n_hat, b) => dot(n_hat, a) = dot(n_hat, b).
        # This is always true for the normal n = (a-b) x (b-c).
        # Let's check: dot(n, a) = dot((a-b)x(b-c), a) = dot(a, (a-b)x(b-c)).
        # dot(n, b) = dot(b, (a-b)x(b-c)).
        # It turns out that for any vector v, dot(v, (a-b)x(b-c)) is proportional to the scalar triple product.
        # The distance from C to a is cos(theta).
        # We need to find k such that C is on the unit sphere? No, the circumcenter of the *spherical* triangle is on the sphere.
        # The circumradius R is the angle between C and a.
        # C = n_hat / ||n_hat||? No, n_hat is already unit.
        # Let's verify if n_hat is the circumcenter.
        # dot(n_hat, a) = dot(n_hat, b) = dot(n_hat, c).
        # Yes, the normal to the plane containing a,b,c is equidistant from a,b,c.
        # So C = n_hat or -n_hat.
        # We choose the sign such that it is "outside" the triangle, i.e., dot(C, a) > 0 (assuming acute triangle logic, but generally we want the center of the cap).
        # The covering radius is the angle to the *farthest* point in the facet.
        # The circumcenter of the facet is the point on the sphere that is the center of the minimal cap covering the facet vertices.
        # So C should be the one that minimizes the max angle to vertices.
        # Since dot(C, a) = dot(C, b) = dot(C, c) = cos(R), R is the circumradius.
        # We want the smallest R, so we want the largest cos(R).
        # So we choose the sign of n_hat such that dot(n_hat, a) is positive.
        
        dots = np.sum(cc * a[valid], axis=1)
        # If dot is negative, flip the center
        cc[dots < 0] *= -1
        
        centers[valid] = cc
        
        # Calculate radii (angular distance from center to vertex a)
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
        
        # For invalid facets, set radius to 0 or large? 
        # If degenerate, it's likely a numerical artifact or a very small facet.
        # We should probably ignore them or assign a small radius.
        # However, if the hull is valid, facets shouldn't be degenerate unless points are nearly collinear.
        # We'll leave radii as 0 for invalid, which means they won't be the "worst" unless all are 0.
        
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
        
    return hull, facets, centers, radii, worst_idx

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, facets, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    """
    hull, facets, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Use log-sum-exp trick for numerical stability
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    n_facets = len(facets)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-15:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r) projected to tangent space
        grad_contrib = -C / sin_r
        
        simplex = facets[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Project gradient to tangent space at p
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _electrostatic_relax(points, time_budget, n_iters=50):
    """
    Electrostatic repulsion relaxation: minimize sum of 1/r^2 between pairs
    to spread points evenly.
    """
    start_time = time.time()
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    for it in range(n_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Compute pairwise repulsion gradient
        grads = np.zeros_like(current)
        for i in range(N):
            for j in range(i+1, N):
                diff = current[i] - current[j]
                dist_sq = np.dot(diff, diff)
                dist_sq = max(dist_sq, 1e-10)
                # Force is proportional to 1/dist^2 in direction of diff
                force = diff / (dist_sq * np.sqrt(dist_sq))
                grads[i] += force
                grads[j] -= force
        
        # Normalize gradient to unit step
        gnorm = np.linalg.norm(grads)
        if gnorm < 1e-14:
            break
        
        step = 0.01
        new_points = current + step * grads / gnorm
        new_points = _normalize(new_points)
        
        rad = _compute_covering_radius(new_points)
        if rad < best_radius - 1e-14:
            current = new_points
            best_radius = rad
            best_points = new_points.copy()
        else:
            # Halve step size if no improvement
            step *= 0.5
    
    return best_points, best_radius

def _lbfgs_stage(points, time_budget, beta=100.0):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_soft_r = _compute_covering_radius(points)
    
    def objective_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = _normalize(x.reshape(N, 3))
        soft_r, grads = _soft_max_radius_and_grad(pts, beta=beta)
        grad_flat = grads.reshape(-1)
        return soft_r, grad_flat
    
    x0 = best_pts.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.fun < best_soft_r - 1e-12:
            new_pts = _normalize(result.x.reshape(N, 3))
            new_rad = _compute_covering_radius(new_pts)
            if new_rad < best_soft_r:
                best_pts = new_pts
                best_soft_r = new_rad
    except Exception:
        pass
    
    return best_pts

def _targeted_worst_facet_polish(points, time_budget):
    """
    Directly optimize the worst facet by moving its vertices to minimize
    the circumradius of that specific facet.
    This is a targeted local refinement.
    """
    start_time = time.time()
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    for it in range(200):
        if time.time() - start_time > time_budget:
            break
        
        hull, facets, centers, radii, worst_idx = _compute_facet_data(current)
        if hull is None or worst_idx is None:
            break
        
        worst_r = radii[worst_idx]
        C = centers[worst_idx]
        simplex = facets[worst_idx]
        
        # We want to reduce worst_r.
        # The gradient of the radius w.r.t. each vertex p_k is -C/sin(r) projected to tangent.
        # But this is for the *current* radius.
        # Moving vertices towards C reduces the angle.
        
        # Let's try a small step towards C for each vertex of the worst facet.
        step_size = 0.02
        moved = False
        for k in range(3):
            p_idx = simplex[k]
            p = current[p_idx]
            
            # Direction: move p toward C
            # Project C onto tangent space at p? No, just move towards C in 3D then normalize.
            # Or move along the geodesic towards C.
            # Vector from p to C on sphere:
            # v = C - p
            # project v to tangent at p: v_t = v - (v.dot(p))p
            v = C - p
            v_t = v - np.dot(v, p) * p
            v_norm = np.linalg.norm(v_t)
            
            if v_norm > 1e-10:
                new_p = p + step_size * v_t / v_norm
                current[p_idx] = new_p
                moved = True
        
        if not moved:
            break
            
        current = _normalize(current)
        rad = _compute_covering_radius(current)
        
        if rad < best_radius - 1e-14:
            best_radius = rad
            best_points = current.copy()
        else:
            # Revert if it made things worse
            current = best_points.copy()
            # Try smaller step next time? Or break?
            # If we just reverted, the worst facet might have changed or stayed same.
            # Let's try a smaller step for the next iteration.
            step_size *= 0.7
            if step_size < 1e-5:
                break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_points = None
    best_radius = np.inf

    # Generate diverse seeds
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    n_seeds = len(seeds)
    
    # Stage 1: For each seed, apply electrostatic relax + L-BFGS + Targeted Polish
    # Reserve ~30s for final polish stages
    stage1_deadline = deadline - 30.0

    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 5:
            break
        
        if i < n_seeds - 1:
            time_budget = min(6.0, remaining / (n_seeds - i))
        else:
            time_budget = min(10.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Electrostatic relaxation for initial spread
        if time_budget > 3.0:
            pts, _ = _electrostatic_relax(seed_pts, min(1.5, time_budget - 2.0))
        else:
            pts = seed_pts
        
        # Apply L-BFGS stage
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3.0:
            pts = _lbfgs_stage(pts, min(3.0, remaining - 2.0), beta=80.0)
        
        # Targeted worst facet polish
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3.0:
            pts, rad = _targeted_worst_facet_polish(pts, min(2.5, remaining - 1.0))
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Iterative targeted polish with increasing precision
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            # First pass: moderate step
            budget = min(8.0, remaining - 10)
            if budget > 3:
                pts, rad = _targeted_worst_facet_polish(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Second pass: L-BFGS with high beta to smooth out
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(6.0, remaining - 8)
            if budget > 3:
                pts = _lbfgs_stage(best_points, budget, beta=300.0)
                rad = _compute_covering_radius(pts)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Third pass: Fine targeted polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(5.0, remaining - 5)
            if budget > 2:
                pts, rad = _targeted_worst_facet_polish(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(3.0, remaining - 3)
            if budget > 1:
                pts, rad = _targeted_worst_facet_polish(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized and valid
    best_points = _normalize(best_points)
    
    return best_points
