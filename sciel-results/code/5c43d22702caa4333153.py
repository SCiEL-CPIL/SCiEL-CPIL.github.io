"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere.
    Uses the convex hull facets to find the largest empty cap."""
    from scipy.spatial import ConvexHull
    
    # Normalize points
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Check for coincident points
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return 180.0  # Invalid: coincident points
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere iff there exists a plane through origin
    # such that all points are on one side
    # Equivalent: the origin is NOT in the convex hull of the points
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # The covering radius is the maximum over all facets of the circumradius
    # For each facet, the circumradius is the angular distance from the facet's
    # circumcenter (on the sphere) to the facet vertices.
    
    # Get the facets
    simplices = hull.simplices
    
    max_radius = 0.0
    
    for simplex in simplices:
        # For a simplex on the unit sphere, the circumcenter is the normalized
        # sum of the vertices (for equilateral) or we need to compute it properly.
        # Actually, for a spherical simplex, the circumcenter is the point equidistant
        # from all vertices. For a triangle on the unit sphere, we can compute it.
        
        # For a general simplex of dimension d, the circumcenter on the sphere
        # is the normalized vector that is equidistant from all vertices.
        # For a triangle (2-simplex) on the unit sphere:
        # The circumcenter is proportional to a + b + c if the triangle is acute,
        # but we need to be more careful.
        
        # Actually, the circumradius of a spherical polygon is the maximum angular
        # distance from the circumcenter to any vertex. The circumcenter of a spherical
        # simplex is the intersection of the perpendicular bisectors.
        
        # For a triangle with vertices a, b, c on the unit sphere:
        # The circumcenter is along the direction of the normal to the plane containing
        # the triangle, but that's not quite right for the sphere.
        
        # Let me think differently. The covering radius is the maximum over all points
        # y on the sphere of min_i angle(y, x_i). This equals the maximum over all
        # Voronoi vertices (which are circumcenters of Delaunay triangles) of the
        # distance to the nearest point.
        
        # For each facet of the convex hull, the "circumcenter" in the spherical sense
        # is the point on the sphere equidistant from the facet's vertices.
        
        # For a triangle with vertices a, b, c:
        # We want y such that |y-a| = |y-b| = |y-c| and |y|=1.
        # This means y is equidistant from a, b, c.
        # y · a = y · b = y · c
        # So y is in the null space of [a-b; b-c] (in the sense that y·(a-b)=0 and y·(b-c)=0)
        # So y is parallel to (a-b) × (b-c) = normal to the plane.
        # But wait, that gives the center of the circumscribed circle in 3D, not on the sphere.
        
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
        # is the normalized version of the 3D circumcenter of the triangle.
        
        # The 3D circumcenter of triangle a, b, c is the point equidistant from a, b, c in R^3.
        # This is the intersection of the perpendicular bisector planes.
        
        # For a triangle on the unit sphere, the 3D circumcenter O satisfies:
        # |O - a|^2 = |O - b|^2 = |O - c|^2
        # O·O - 2O·a + |a|^2 = O·O - 2O·b + |b|^2
        # Since |a| = |b| = |c| = 1:
        # -2O·a = -2O·b => O·(a-b) = 0
        # Similarly O·(b-c) = 0
        # So O is parallel to (a-b) × (b-c).
        
        # The 3D circumcenter is t * n where n = (a-b) × (b-c) and t is chosen so that
        # |t*n - a|^2 is minimized... actually it's determined by the condition.
        
        # Let n = (a-b) × (b-c), normalized.
        # O = t * n (if the triangle is not degenerate)
        # |O - a|^2 = |t*n - a|^2 = t^2 - 2t(n·a) + 1
        # For this to be equal for b and c, we need n·a = n·b = n·c.
        # n·(a-b) = 0 and n·(b-c) = 0, which is true by construction.
        # So n·a = n·b = n·c = some constant k.
        # The circumradius in 3D is |O - a| = sqrt(t^2 - 2tk + 1).
        # But we want the point on the sphere, so we normalize O to get y = O/|O|.
        # The angular distance from y to a is arccos(y·a) = arccos(k/|t*n|) = arccos(k/(|t|*|n|))
        # Since n is unit, |n|=1, so y·a = k/t.
        
        # Actually, let me just compute it numerically.
        
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the normal to the plane
        n = np.cross(a - b, b - c)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # The circumcenter direction is n (or -n)
        # We need to determine the sign and the distance.
        # The circumcenter O = t * n
        # Condition: |O - a|^2 = |O - b|^2 is automatically satisfied.
        # We need |O - a|^2 to give the correct circumradius.
        
        # Actually, the circumcenter of the triangle in R^3 is unique.
        # O = t*n where t = (n·a) / (1 - (n·a)^2) ... no, that's not right.
        
        # Let me solve it properly.
        # O = t*n
        # |O-a|^2 = t^2 - 2t(n·a) + 1
        # The circumradius R satisfies R^2 = t^2 - 2t(n·a) + 1
        # But we also know that for the circumcenter, the distance to each vertex is the same.
        # This is already satisfied.
        
        # The issue is: what is t? The circumcenter is the center of the unique sphere
        # passing through a, b, c. Since a, b, c are on the unit sphere centered at origin,
        # the circumcenter of the triangle (in the plane of the triangle) is different
        # from the center of the sphere.
        
        # I think I'm overcomplicating this. Let me use a different approach.
        
        # The covering radius is the maximum, over all points y on the sphere, of
        # min_i arccos(y·x_i).
        
        # The maximum occurs at a point equidistant from at least 3 of the x_i's
        # (a Voronoi vertex) or at a point equidistant from 2 (on an edge) or 1 (unlikely for max).
        
        # For each triple (i,j,k), find the point y equidistant from x_i, x_j, x_k
        # and check if it's the nearest to those three (i.e., it's a Voronoi vertex).
        
        # But that's O(N^3) which is 130^3 = 2.2 million, too slow.
        
        # Let me use the convex hull approach properly.
        
        # For each facet (triangle) of the convex hull, the circumcenter of the
        # spherical triangle is the point on the sphere equidistant from the three vertices.
        # This is the Voronoi vertex corresponding to that Delaunay triangle.
        
        # For triangle with vertices a, b, c on unit sphere:
        # The circumcenter y satisfies y·a = y·b = y·c and |y| = 1.
        # Let y = alpha*a + beta*b + gamma*c
        # y·a = alpha + beta*(b·a) + gamma*(c·a)
        # y·b = alpha*(a·b) + beta + gamma*(c·b)
        # y·c = alpha*(a·c) + beta*(b·c) + gamma
        
        # Set y·a = y·b:
        # alpha + beta*(b·a) + gamma*(c·a) = alpha*(a·b) + beta + gamma*(c·b)
        # alpha*(1 - a·b) + beta*(a·b - 1) + gamma*(a·c - b·c) = 0
        # alpha*(1 - a·b) = beta*(1 - a·b) + gamma*(b·c - a·c)
        
        # This is getting messy. Let me just use the formula:
        # The circumcenter is proportional to the vector:
        # y_dir = (|b|^2|c|^2(a - b) + |a|^2|c|^2(b - c) + |a|^2|b|^2(c - a)) / (something)
        
        # Since all are unit vectors:
        # y_dir = |c|^2(a-b) + |a|^2(b-c) + |a|^2(c-a) ... no.
        
        # Standard formula for circumcenter of triangle with vertices A, B, C:
        # O = A + (|B|^2 * cross(C-A, A-B) + |C|^2 * cross(A-B, B-A)) / (2 * cross(C-A, B-A) · something)
        
        # Let me just use a numerical approach: solve for y such that
        # y·a = y·b, y·b = y·c, |y| = 1.
        # From y·a = y·b: y·(a-b) = 0
        # From y·b = y·c: y·(b-c) = 0
        # So y is parallel to (a-b) × (b-c).
        
        y_dir = np.cross(a - b, b - c)
        y_norm = np.linalg.norm(y_dir)
        if y_norm < 1e-10:
            continue
        y = y_dir / y_norm
        
        # Check which side: y should be on the same side as the facet normal
        # The facet normal points outward from the convex hull
        # The circumcenter should be on the sphere, so we just use y or -y
        # We need the one that is "outside" the triangle in the spherical sense
        
        # Compute the angular distances
        d_a = np.arccos(np.clip(y @ a, -1, 1))
        d_b = np.arccos(np.clip(y @ b, -1, 1))
        d_c = np.arccos(np.clip(y @ c, -1, 1))
        
        # Also check -y
        neg_y = -y
        d_a2 = np.arccos(np.clip(neg_y @ a, -1, 1))
        d_b2 = np.arccos(np.clip(neg_y @ b, -1, 1))
        d_c2 = np.arccos(np.clip(neg_y @ c, -1, 1))
        
        # The circumcenter should be equidistant, so d_a ≈ d_b ≈ d_c
        # Use the one that gives a valid circumcenter (the one farther from the origin in some sense)
        # Actually both y and -y are equidistant from a, b, c.
        # We want the one that is the Voronoi vertex, which is the one NOT in the convex hull
        # of the other points... this is complex.
        
        # For the covering radius, we want the maximum over all such circumcenters.
        # Let's use max of the two candidates.
        
        r1 = max(d_a, d_b, d_c)
        r2 = max(d_a2, d_b2, d_c2)
        
        # The circumradius of the facet is the larger of the two? No.
        # The circumcenter is the one that is equidistant AND is the farthest from the
        # other points (i.e., it's the Voronoi vertex).
        
        # For now, let's use the maximum of r1 and r2 as an upper bound.
        # Actually, the correct circumcenter is the one that is on the "outside" of the
        # convex hull. The convex hull facet has an outward normal. The circumcenter
        # should be on the same side.
        
        # The outward normal of the facet is n (computed above, pointing outward).
        # The circumcenter y should satisfy y·n > 0 (roughly).
        
        if y @ n > 0:
            facet_radius = r1
        else:
            facet_radius = r2
        
        if facet_radius > max_radius:
            max_radius = facet_radius
    
    return np.degrees(max_radius)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    # Local optimization using gradient descent on the sphere
    # Objective: maximize the minimum distance from each point to its nearest neighbor
    # This is a proxy for minimizing the covering radius
    
    def objective(params):
        # params: flattened N*3 array
        p = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        # Compute pairwise distances
        # d[i,j] = arccos(p_i · p_j)
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.pi)
        
        # Minimum distance for each point
        min_dists = np.min(dists, axis=1)
        
        # We want to maximize the minimum of min_dists
        # Use a soft version for differentiability
        # Or just use the min (non-smooth but works with subgradient methods)
        
        return -np.min(min_dists)
    
    def gradient(params):
        p = params.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.pi)
        
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        
        # Gradient of -min(min_dists)
        # d/dp_i [-min_j dist(i,j)] = -d/dp_i dist(i, j_i) where j_i is the nearest neighbor
        # d/dp_i arccos(p_i · p_j) = -(p_j - (p_i·p_j)p_i) / sqrt(1 - (p_i·p_j)^2)
        
        grad = np.zeros_like(p)
        for i in range(N):
            j = min_indices[i]
            if i == j:
                continue
            dot_ij = np.clip(p[i] @ p[j], -1.0, 1.0)
            denom = np.sqrt(max(1e-10, 1.0 - dot_ij**2))
            # Gradient of arccos(dot) w.r.t. p_i
            grad_p_i = -(p[j] - dot_ij * p[i]) / denom
            # We're minimizing -min, so gradient is -grad_p_i
            grad[i] += -grad_p_i
        
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= (grad[i] @ p[i]) * p[i]
        
        return grad.flatten()
    
    # Use L-BFGS-B with projection onto sphere
    # Actually, let's use a simpler approach: gradient descent with re-normalization
    
    lr = 0.01
    for iteration in range(2000):
        if time.time() > deadline - 30:
            break
        
        # Compute gradient
        params = best_pts.flatten()
        p = best_pts.copy()
        
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.pi)
        
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        
        # Gradient step
        grad = np.zeros_like(p)
        for i in range(N):
            j = min_indices[i]
            if i == j:
                continue
            dot_ij = np.clip(p[i] @ p[j], -1.0, 1.0)
            denom = np.sqrt(max(1e-10, 1.0 - dot_ij**2))
            grad_p_i = -(p[j] - dot_ij * p[i]) / denom
            grad[i] += -grad_p_i
        
        # Project to tangent space
        for i in range(N):
            grad[i] -= (grad[i] @ p[i]) * p[i]
        
        # Update
        new_pts = p + lr * grad
        # Re-normalize
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check if improved
        new_radius = _compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_pts = new_pts
            best_radius = new_radius
        
        # Decrease learning rate
        lr *= 0.999
    
    # Final polish with finer steps
    lr = 0.001
    for iteration in range(1000):
        if time.time() > deadline - 5:
            break
        
        p = best_pts.copy()
        
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.pi)
        
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        
        grad = np.zeros_like(p)
        for i in range(N):
            j = min_indices[i]
            if i == j:
                continue
            dot_ij = np.clip(p[i] @ p[j], -1.0, 1.0)
            denom = np.sqrt(max(1e-10, 1.0 - dot_ij**2))
            grad_p_i = -(p[j] - dot_ij * p[i]) / denom
            grad[i] += -grad_p_i
        
        for i in range(N):
            grad[i] -= (grad[i] @ p[i]) * p[i]
        
        new_pts = p + lr * grad
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = _compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_pts = new_pts
            best_radius = new_radius
    
    return best_pts

# EVOLVE-BLOCK-END
