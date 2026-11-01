"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in degrees from the convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    
    # Get facet normals and offsets
    # For each facet, the circumradius is the distance from the facet plane to the farthest point on the sphere
    # which equals the angle between the facet normal and the origin projected
    
    # Actually, for a unit sphere, the covering radius is the maximum over all facets of
    # the angular distance from the facet's circumcenter to the vertex
    # The circumradius of a spherical triangle can be computed from its angles
    
    # Simpler approach: use Delaunay triangulation and compute circumradius for each triangle
    try:
        tri = Delaunay(pts)
    except Exception:
        return 90.0
    
    max_cr = 0.0
    for simplex in tri.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumradius of the spherical triangle
        # The circumcenter is the normalized cross product of (v0-v2) x (v1-v2)
        # but for unit vectors, we can use the formula
        
        # Side lengths (angular distances)
        d01 = np.arccos(np.clip(np.dot(v0, v1), -1, 1))
        d12 = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
        d02 = np.arccos(np.clip(np.dot(v0, v2), -1, 1))
        
        # For a spherical triangle, the circumradius R satisfies:
        # sin(R) = sin(a)*sin(b)*sin(c) / (2*sin(s)*sin(s-a)*sin(s-b)*sin(s-c))
        # where a,b,c are side lengths and s is semiperimeter
        # But this is complex. Alternative: find circumcenter as the point equidistant from all three vertices
        
        # The circumcenter n satisfies: n·v0 = n·v1 = n·v2 = cos(R)
        # So n is in the span of v0, v1, v2 and equidistant from all three
        # n = alpha*v0 + beta*v1 + gamma*v2 with n·v0 = n·v1 = n·v2
        
        # Using the formula: circumcenter direction is proportional to
        # (v0 × v1) × (v1 × v2) ... actually let's use a different approach
        
        # The circumcenter is the normalized vector:
        # n = (v0 + v1 + v2) is NOT correct in general
        
        # Correct approach: solve for n such that |n - v_i| is constant for i=0,1,2
        # This means n·v_i = constant for all i
        # So (n·v0 - n·v1) = 0 and (n·v0 - n·v2) = 0
        # n · (v0 - v1) = 0 and n · (v0 - v2) = 0
        # So n is perpendicular to (v0-v1) and (v0-v2), i.e., n || (v0-v1) × (v0-v2)
        
        cross = np.cross(v0 - v1, v0 - v2)
        if np.linalg.norm(cross) < 1e-15:
            continue
        n = cross / np.linalg.norm(cross)
        
        # Check which hemisphere - the circumcenter should be on the same side as the triangle
        # The triangle center is roughly (v0+v1+v2)/3
        center = (v0 + v1 + v2) / 3.0
        if np.dot(n, center) < 0:
            n = -n
        
        # The circumradius is the angle between n and any vertex
        cos_R = np.clip(np.dot(n, v0), -1, 1)
        R = np.arccos(cos_R)
        max_cr = max(max_cr, R)
    
    return np.degrees(max_cr)

def _lloyd_step(pts, num_iters=50):
    """Perform Lloyd's algorithm iterations on the sphere."""
    pts = _normalize(pts.copy())
    
    for _ in range(num_iters):
        # Compute Voronoi cells using Delaunay
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell vertices (circumcenters of adjacent triangles)
        new_pts = np.zeros_like(pts)
        
        for i in range(len(pts)):
            # Find all triangles containing point i
            cell_vertices = []
            for simplex in tri.simplices:
                if i in simplex:
                    v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                    cross = np.cross(v0 - v1, v0 - v2)
                    if np.linalg.norm(cross) > 1e-15:
                        n = cross / np.linalg.norm(cross)
                        center = (v0 + v1 + v2) / 3.0
                        if np.dot(n, center) < 0:
                            n = -n
                        cell_vertices.append(n)
            
            if len(cell_vertices) == 0:
                new_pts[i] = pts[i]
            else:
                cell_vertices = np.array(cell_vertices)
                # Average the cell vertices and normalize
                centroid = np.mean(cell_vertices, axis=0)
                if np.linalg.norm(centroid) > 1e-15:
                    new_pts[i] = centroid / np.linalg.norm(centroid)
                else:
                    new_pts[i] = pts[i]
        
        new_pts = _normalize(new_pts)
        
        # Check convergence
        if np.allclose(new_pts, pts, atol=1e-10):
            pts = new_pts
            break
        pts = new_pts
    
    return pts

def _facet_optimization(pts, max_time=30):
    """Optimize points to minimize the maximum facet circumradius using SLSQP."""
    pts = _normalize(pts.copy())
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > max_time:
            return 180.0
        p = _normalize(x.reshape(N, 3))
        return _covering_radius(p)
    
    # Use Nelder-Mead first for robustness, then SLSQP
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_cr and time.time() - start_time < max_time:
            best_pts = _normalize(res.x.reshape(N, 3))
            best_cr = res.fun
    except Exception:
        pass
    
    # SLSQP refinement
    if time.time() - start_time < max_time:
        try:
            def obj_slsqp(x):
                if time.time() - start_time > max_time:
                    return 180.0
                p = _normalize(x.reshape(N, 3))
                return _covering_radius(p)
            
            def jac_slsqp(x):
                # Numerical gradient
                eps = 1e-7
                grad = np.zeros_like(x)
                f0 = obj_slsqp(x)
                for i in range(len(x)):
                    xp = x.copy()
                    xp[i] += eps
                    grad[i] = (obj_slsqp(xp) - f0) / eps
                return grad
            
            res = minimize(obj_slsqp, best_pts.flatten(), method='SLSQP',
                          jac=jac_slsqp, options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < best_cr:
                best_pts = _normalize(res.x.reshape(N, 3))
                best_cr = res.fun
        except Exception:
            pass
    
    return best_pts, best_cr

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Stage 1: Lloyd's algorithm for global improvement
    if time.time() < deadline - 40:
        pts = _lloyd_step(pts, num_iters=100)
    
    # Stage 2: Facet-based optimization
    if time.time() < deadline - 20:
        pts, cr = _facet_optimization(pts, max_time=min(30, deadline - time.time() - 10))
    
    # Stage 3: Additional Lloyd steps for final polish
    if time.time() < deadline - 5:
        pts = _lloyd_step(pts, num_iters=50)
    
    # Final normalization
    pts = _normalize(pts)
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                # Perturb slightly
                pts[j] += np.random.normal(0, 1e-6, 3)
                pts[j] /= np.linalg.norm(pts[j])
    
    return pts
# EVOLVE-BLOCK-END
