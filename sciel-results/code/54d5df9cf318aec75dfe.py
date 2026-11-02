"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms


def _covering_radius_degrees(points):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest point in the configuration.
    This equals the largest circumradius of the facets of the convex hull,
    where each facet is checked to have an empty cap.
    """
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0

    max_radius = 0.0
    # For each facet, compute the circumradius
    # A facet is defined by its vertices
    for simplex in hull.simplices:
        verts = points[simplex]
        # Compute circumcenter of the simplex (triangle in 3D)
        # For a triangle with vertices a, b, c on the unit sphere,
        # the circumradius (angular) is related to the circumcenter
        # The circumcenter in 3D for a triangle on the sphere:
        # We need to find the point on the sphere farthest from the triangle's plane
        # that is equidistant from all three vertices.
        
        # For a triangle on the unit sphere, the circumcenter (in angular sense)
        # is the point on the great circle perpendicular to the triangle's plane.
        
        # Compute the normal to the triangle
        a, b, c = verts[0], verts[1], verts[2]
        edge1 = b - a
        edge2 = c - a
        normal = np.cross(edge1, edge2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            continue
        normal = normal / normal_norm
        
        # The circumcenter of the triangle on the sphere is at angle θ from each vertex
        # where cos(θ) = (a·normal) / sqrt(1 - (a·normal)^2) ... this is tricky.
        
        # Alternative: the circumradius of a spherical triangle can be computed
        # from the side lengths. For a triangle with sides a, b, c (angular),
        # the circumradius R satisfies:
        # sin(R) = abc / (4 * area) where area is the spherical excess... 
        # Actually, let's use a more direct approach.
        
        # The circumcenter of a spherical triangle is the pole of the great circle
        # circumscribing it. For a triangle with vertices a, b, c on the unit sphere,
        # the circumradius R satisfies:
        # a·b = b·c = c·a = cos(R) ... no, that's only for equilateral.
        
        # Let me use the formula: for a triangle with vertices a, b, c,
        # the circumradius R is given by:
        # R = arcsin(|a × b × c| / (4 * |a × b| * |b × c| * |c × a|)) ... not quite.
        
        # Actually, a cleaner approach: the circumcenter u satisfies
        # u·a = u·b = u·c = cos(R)
        # and |u| = 1.
        # So u is in the plane spanned by a, b, c (since it's equidistant).
        # u = αa + βb + γc
        # u·a = α + β(b·a) + γ(c·a) = cos(R)
        # u·b = α(a·b) + β + γ(c·b) = cos(R)
        # u·c = α(a·c) + β(b·c) + γ = cos(R)
        # |u|^2 = 1
        
        # This is getting complex. Let me use a simpler numerical approach:
        # The covering radius for a facet is the angular distance from the
        # circumcenter to any vertex. We can compute this by finding the
        # point on the sphere that maximizes the minimum distance to the
        # three vertices, but that's the same problem.
        
        # Let me just compute the circumradius directly from the triangle's
        # side lengths using the spherical law of sines/cosines.
        
        # Sides (angular distances between vertices):
        cos_ab = np.clip(np.dot(a, b), -1.0, 1.0)
        cos_bc = np.clip(np.dot(b, c), -1.0, 1.0)
        cos_ca = np.clip(np.dot(c, a), -1.0, 1.0)
        side_ab = np.arccos(cos_ab)
        side_bc = np.arccos(cos_bc)
        side_ca = np.arccos(cos_ca)
        
        # Spherical circumradius formula:
        # For a spherical triangle with sides a, b, c and angles A, B, C,
        # the circumradius R is given by:
        # sin(R) = (sin(a) * sin(b) * sin(c)) / (4 * sin(s))
        # where s = (a+b+c)/2 ... no, that's not right either.
        
        # Let me use: R = arccos( (cos(a) + cos(b) + cos(c)) / (1 + 2*cos(a)*cos(b)*cos(c)) )
        # Hmm, I'm not sure about that formula.
        
        # Let me just do a numerical optimization for the circumcenter.
        # The circumcenter u maximizes min(u·a, u·b, u·c) subject to |u|=1.
        # At the optimum, u·a = u·b = u·c = cos(R).
        
        # Solve: u·a = u·b, u·a = u·c, |u|^2 = 1
        # This means u is orthogonal to (a-b) and (a-c).
        # So u is parallel to (a-b) × (a-c) = normal (up to sign).
        # u = ±normal
        # But we need u·a = u·b = u·c, and u·a = ±(normal·a).
        # For the circumcenter to be on the "correct" side (the side where
        # the cap is empty), we pick the sign that gives u·a > 0 (or the
        # one that's inside the convex hull).
        
        # Actually, for a convex hull facet, the circumcenter that matters
        # is the one on the outside of the hull (the one where the cap is
        # empty). The covering radius is the angular distance from this
        # circumcenter to the nearest vertex.
        
        # u = normal or u = -normal
        # The circumradius R = arccos(|u·a|) where u is the circumcenter.
        
        dot_a = np.dot(a, normal)
        dot_b = np.dot(b, normal)
        dot_c = np.dot(c, normal)
        
        # Check if all dots have the same sign (they should for a proper facet)
        if abs(dot_a) < 1e-10:
            continue
        
        # The circumcenter is at angular distance arccos(|dot_a|) from a
        # (since u·a = ±dot_a and |u|=1, |a|=1)
        circumradius = np.arccos(np.clip(abs(dot_a), 0.0, 1.0))
        
        if circumradius > max_radius:
            max_radius = circumradius
    
    return np.degrees(max_radius)


def _pairwise_distances(points):
    """Compute pairwise angular distances."""
    points = _normalize(points)
    dots = np.clip(points @ points.T, -1.0, 1.0)
    return np.arccos(dots)


def _objective(x):
    """
    Objective: minimize the covering radius.
    We use a smooth approximation via log-sum-exp of squared distances.
    """
    points = _normalize(x.reshape(N, 3))
    # Compute pairwise distances
    dots = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(dots)
    # We want to minimize the maximum distance from any point to its nearest neighbor
    # This is equivalent to minimizing the covering radius.
    # For each point i, the nearest neighbor distance is min_j dist(i,j) for j != i
    # The covering radius is related to the Delaunay triangulation, but as an
    # approximation, we can use the minimum pairwise distance as a proxy.
    # Actually, the covering radius is NOT the minimum pairwise distance.
    # The covering radius is the largest circumradius of the hull facets.
    # A better proxy: maximize the minimum pairwise distance (Tammes problem)
    # is different from minimizing the covering radius (covering problem).
    
    # For the covering problem, we want to minimize the largest empty cap.
    # A good smooth objective is the sum of exp(k * d_ij^2) for all pairs,
    # which encourages all pairwise distances to be similar and large.
    # But this is more like the Tammes problem.
    
    # Let me think differently. The covering radius R satisfies:
    # For every point y on the sphere, there exists a point p_i with d(y, p_i) <= R.
    # This is equivalent to: the union of caps of radius R around each p_i covers the sphere.
    # The boundary of the covered region is determined by the Voronoi diagram.
    # The covering radius is the maximum over all Voronoi vertices of the distance
    # to the nearest point.
    
    # For a smooth objective, let's use:
    # f(x) = log-sum-exp(k * min_j d_ij^2) over all i
    # This approximates the maximum of the minimum distances.
    
    # For each point i, min distance to other points:
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    # Smooth max via log-sum-exp
    k = 50.0
    max_min_dist = np.log(np.sum(np.exp(k * min_dists**2))) / k
    # This is a lower bound on the actual max, but it's smooth.
    # Actually, log-sum-exp gives an upper bound on the max.
    # Let's just return the max for simplicity, and rely on the optimizer.
    return np.max(min_dists)


def _objective_smooth(x):
    """
    Smooth objective: minimize a smooth approximation of the covering radius.
    Uses log-sum-exp to smooth the max operation.
    """
    points = _normalize(x.reshape(N, 3))
    dots = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    # Smooth max via log-sum-exp
    k = 100.0
    # Scale to avoid overflow
    scaled = k * min_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum_exp / k


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_x = x0.copy()
    best_score = _covering_radius_degrees(best_x)
    
    # Try a few random restarts with optimization
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x = x0.copy()
        else:
            # Random perturbation
            x = x0 + 0.1 * rng.standard_normal((N, 3))
            x = _normalize(x)
        
        # Use L-BFGS-B with the smooth objective
        try:
            result = minimize(
                _objective_smooth,
                x.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = _normalize(result.x.reshape(N, 3))
            score = _covering_radius_degrees(x_opt)
            
            if score < best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Final polish: try to improve with a more targeted approach
    if time.time() < deadline - 10:
        # Try SLSQP with tighter constraints
        try:
            # Constraint: all pairwise distances should be >= some threshold
            # This is the Tammes problem constraint, which helps with covering too.
            x = best_x.flatten()
            
            # Use a simple gradient-based refinement
            # Move each point away from its nearest neighbor
            for _iter in range(50):
                if time.time() > deadline:
                    break
                points = _normalize(x.reshape(N, 3))
                dots = np.clip(points @ points.T, -1.0, 1.0)
                dists = np.arccos(dots)
                np.fill_diagonal(dists, np.inf)
                min_idx = np.argmin(dists, axis=1)
                min_dists = np.min(dists, axis=1)
                
                # Gradient of the objective w.r.t. each point
                # For simplicity, move each point away from its nearest neighbor
                grad = np.zeros_like(x)
                for i in range(N):
                    j = min_idx[i]
                    # Move point i away from point j
                    direction = points[i] - points[j]
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-10:
                        direction = direction / direction_norm
                    # Also account for the constraint that points stay on the sphere
                    grad[i*3:(i+1)*3] = direction * min_dists[i]
                
                # Gradient step
                lr = 0.01
                x = x - lr * grad
                x = _normalize(x.reshape(N, 3)).flatten()
            
            x_opt = _normalize(x.reshape(N, 3))
            score = _covering_radius_degrees(x_opt)
            if score < best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
