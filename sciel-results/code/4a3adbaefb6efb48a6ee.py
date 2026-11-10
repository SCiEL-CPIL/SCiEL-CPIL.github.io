"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This equals the maximum circumradius of the
    spherical Voronoi cells, which can be found by checking the circumcircles of
    the spherical Delaunay triangles.
    
    For a triangle with vertices a, b, c on the unit sphere, the circumradius (in terms
    of angular distance from the circumcenter to each vertex) can be computed.
    """
    # Compute pairwise angular distances
    # For efficiency, we'll use the spherical Delaunay approach
    
    # Project to 3D and use Delaunay triangulation
    # The spherical Delaunay is related to the planar Delaunay of the projected points
    # but for a sphere, we can use the fact that the Voronoi vertices are the circumcenters
    
    # Alternative: compute the convex hull and find the maximum circumradius of facets
    # The covering radius is the maximum over all facets of the circumradius
    # where the circumcenter is on the sphere (i.e., the facet's plane doesn't pass through origin)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max nearest-neighbor distance
        from scipy.spatial.distance import cdist
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        return np.max(np.arccos(np.clip(np.min(dist, axis=1), -1, 1)))
    
    # For each facet (triangle), compute the circumradius
    # The circumradius of a spherical triangle is related to the circumradius of the 
    # planar triangle formed by the three points
    
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is the point equidistant (in angle) from a, b, c
        # It lies in the plane spanned by a, b, c (normalized)
        
        # The circumcenter u satisfies: u·a = u·b = u·c = cos(r)
        # where r is the circumradius
        
        # We can find u as the normalized cross product of (b-a) and (c-a)
        # But more robustly: solve for u in span{a,b,c}
        
        # Use the formula: circumcenter is proportional to the vector
        # that is equidistant from all three points
        # 
        # Let u = alpha*a + beta*b + gamma*c
        # u·a = u·b = u·c
        # 
        # This gives us a linear system. Let's use a different approach:
        # The circumcenter is in the direction of the normal to the plane 
        # through a, b, c... no, that's the center of the circle, not necessarily
        # on the sphere in the right way.
        
        # Better approach: the circumradius can be computed from the side lengths
        # and the area of the spherical triangle.
        
        # Side lengths (angular):
        ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        # Spherical excess / area
        # Using L'Huilier's formula or the formula involving the inradius
        # 
        # Actually, let's use a direct approach: find the circumcenter by solving
        # the system where u is in the plane of a,b,c and equidistant from all three.
        
        # The circumcenter u (on unit sphere) satisfies:
        # u·a = u·b = u·c
        # |u| = 1
        # 
        # u = t1*a + t2*b + t3*c for some t1, t2, t3
        # 
        # u·a = u·b:  t1 + t2*(a·b) + t3*(c·a) = t1*(a·b) + t2 + t3*(b·c)
        # u·a = u·c:  t1 + t2*(a·b) + t3*(c·a) = t1*(a·c) + t2*(b·c) + t3
        # 
        # Let's set up the linear system:
        # (1 - a·b)*t1 + (a·b - 1)*t2 + (c·a - b·c)*t3 = 0
        # (1 - c·a)*t1 + (b·c - a·b)*t2 + (c·a - 1)*t3 = 0
        # 
        # Plus the normalization constraint.
        
        ab_dot = np.clip(np.dot(a, b), -1, 1)
        bc_dot = np.clip(np.dot(b, c), -1, 1)
        ca_dot = np.clip(np.dot(c, a), -1, 1)
        
        # Set up 2x2 system for t1, t2 in terms of t3=1
        # From u·a = u·b:
        # t1*(1-ab_dot) + t2*(ab_dot-1) + (ca_dot - bc_dot) = 0
        # From u·a = u·c:
        # t1*(1-ca_dot) + t2*(bc_dot-ab_dot) + (ca_dot-1) = 0
        
        A = np.array([
            [1 - ab_dot, ab_dot - 1],
            [1 - ca_dot, bc_dot - ab_dot]
        ])
        b_vec = np.array([
            -(ca_dot - bc_dot),
            -(ca_dot - 1)
        ])
        
        try:
            if np.abs(np.linalg.det(A)) < 1e-12:
                # Degenerate case
                # Use fallback: average of the three points
                u = (a + b + c) / np.linalg.norm(a + b + c)
            else:
                t1, t2 = np.linalg.solve(A, b_vec)
                t3 = 1.0
                u = t1*a + t2*b + t3*c
                norm_u = np.linalg.norm(u)
                if norm_u < 1e-12:
                    u = (a + b + c) / np.linalg.norm(a + b + c)
                else:
                    u = u / norm_u
        except np.linalg.LinAlgError:
            u = (a + b + c) / np.linalg.norm(a + b + c)
        
        # Check if u is on the correct side (same side as the triangle center)
        # The centroid of the triangle
        centroid = (a + b + c) / np.linalg.norm(a + b + c)
        if np.dot(u, centroid) < 0:
            u = -u
        
        r = np.arccos(np.clip(np.dot(u, a), -1, 1))
        max_r = max(max_r, r)
    
    return max_r

def _repulsive_force_dynamics(pts, num_steps=5000, dt=0.01, cooling=True):
    """Repulsive force-based dynamics to spread points on the sphere.
    
    Each point experiences repulsion from other points, with the force
    scaled to encourage uniform distribution.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for step in range(num_steps):
        # Compute forces
        forces = np.zeros_like(pts)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Vector from i to j
                v = pts[j] - pts[i]
                dist_sq = np.dot(v, v)
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                
                # Angular distance
                cos_theta = np.clip(np.dot(pts[i], pts[j]), -1, 1)
                theta = np.arccos(cos_theta)
                
                # Repulsive force: stronger when close
                # Use a potential like 1/theta^2 or exp(-theta)
                force_mag = 1.0 / (theta * theta + 1e-6)
                
                # Direction: push i away from j
                # The direction on the sphere is the projection of v onto the tangent plane
                v_tangent = v - np.dot(v, pts[i]) * pts[i]
                if np.linalg.norm(v_tangent) > 1e-12:
                    forces[i] += force_mag * v_tangent / np.linalg.norm(v_tangent)
        
        # Update positions
        if cooling:
            current_dt = dt * (1.0 - step / num_steps) + 0.001
        else:
            current_dt = dt
        
        pts = pts + current_dt * forces
        pts = _normalize(pts)
        
        # Check for time limit
        if step % 100 == 0:
            if time.time() > 130:
                break
    
    return pts

def _local_search(pts, max_iter=10000):
    """Local optimization using gradient-free methods.
    
    We minimize the covering radius by adjusting points in the direction
    that reduces the maximum circumradius.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    best_r = _covering_radius(pts)
    
    for it in range(max_iter):
        if time.time() > 135:
            break
        
        # Find the worst facet
        # For simplicity, do a random local search
        i = np.random.randint(n)
        # Try moving point i in a random direction on the tangent plane
        random_dir = np.random.randn(3)
        random_dir -= np.dot(random_dir, pts[i]) * pts[i]
        random_dir /= np.linalg.norm(random_dir)
        
        # Try a small move
        for step_size in [0.01, 0.005, 0.001]:
            new_pts = pts.copy()
            new_pts[i] = _normalize(pts[i] + step_size * random_dir)
            new_r = _covering_radius(new_pts)
            if new_r < best_r:
                pts = new_pts
                best_r = new_r
                break
    
    return pts, best_r

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    # Run repulsive dynamics
    pts_dyn = _repulsive_force_dynamics(pts, num_steps=3000, dt=0.02)
    r_dyn = _covering_radius(pts_dyn)
    if r_dyn < best_r:
        best_pts = pts_dyn
        best_r = r_dyn
    
    if time.time() > deadline - 10:
        return _normalize(best_pts)
    
    # Local search refinement
    pts_ls, r_ls = _local_search(best_pts, max_iter=5000)
    if r_ls < best_r:
        best_pts = pts_ls
        best_r = r_ls
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
