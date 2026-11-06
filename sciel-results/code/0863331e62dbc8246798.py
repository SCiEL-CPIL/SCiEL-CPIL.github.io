"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-10] = 1.0
    return pts / norms[:, None]

def _covering_radius_deg(pts):
    """Compute exact covering radius in degrees. Returns inf if invalid."""
    pts = np.asarray(pts)
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    max_rad = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]
        k = len(simplex)
        G = v @ v.T
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            continue
        s = np.ones(k)
        val = s @ G_inv @ s
        if val <= 0:
            continue
        cos_theta = 1.0 / np.sqrt(val)
        if cos_theta >= 1.0:
            continue
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        rad_deg = np.degrees(theta)
        if rad_deg > max_rad:
            max_rad = rad_deg
    return max_rad

def _get_worst_points(pts, num_worst=20):
    """
    Identify the 'worst' points on the sphere, i.e., the circumcenters of the facets
    of the convex hull that have the largest circumradius.
    Returns a list of (point_on_sphere, facet_indices) tuples.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return []
    
    facets_info = []
    for simplex in hull.simplices:
        v = pts[simplex]
        k = len(simplex)
        G = v @ v.T
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            continue
        s = np.ones(k)
        val = s @ G_inv @ s
        if val <= 0:
            continue
        cos_theta = 1.0 / np.sqrt(val)
        if cos_theta >= 1.0:
            continue
        
        # The circumcenter c is proportional to sum of v_i / cos_theta? 
        # Actually, c = G^{-1} s / (s^T G^{-1} s)^{1/2} * something?
        # Let's derive: c = v^T a, G a = s. a = G^{-1} s.
        # c = v^T G^{-1} s.
        # |c| = 1/sqrt(s^T G^{-1} s).
        # So normalized c = (v^T G^{-1} s) * sqrt(s^T G^{-1} s).
        
        a = G_inv @ s
        c_unnorm = v.T @ a
        norm_c = np.linalg.norm(c_unnorm)
        if norm_c < 1e-10:
            continue
        c = c_unnorm / norm_c
        
        # Check if this cap is empty (no other points inside)
        # For convex hull facets of points on sphere, the circumcenter is the center of the empty cap.
        # We just need to verify it's a valid center.
        
        facets_info.append((np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))), c, simplex))
    
    if not facets_info:
        return []
        
    # Sort by radius (descending)
    facets_info.sort(key=lambda x: x[0], reverse=True)
    return facets_info[:num_worst]

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_cr = _covering_radius_deg(best_pts)
    
    # Riemannian SGD with Simulated Annealing
    # We will iteratively:
    # 1. Find the worst covering points (circumcenters of largest facets).
    # 2. Calculate gradient: for each worst point c, the points in its facet should move away from c.
    #    Actually, moving points away from c reduces the circumradius of that facet.
    #    The gradient of the distance from a point p to c on the sphere is -c (projected to tangent space).
    #    Wait, we want to MINIMIZE the distance from c to nearest point.
    #    If c is the circumcenter of facet F, the points in F are at distance d from c.
    #    To reduce d, we should move points in F AWAY from c? No, that increases distance.
    #    We want to move points in F TOWARDS c? No, that makes the facet smaller?
    #    
    #    Let's think geometrically.
    #    The covering radius is the max distance from any sphere point to nearest config point.
    #    The worst points are the circumcenters of the facets.
    #    If we move the vertices of a facet, the circumcenter moves.
    #    Generally, "repulsion" between points helps spread them out.
    #    But for covering, we want to fill the holes. The holes are the caps centered at circumcenters.
    #    To reduce the size of the hole, we need to move the boundary points (facet vertices) INWARD towards the center?
    #    No, moving them inward would make the triangle smaller, but the circumcenter might move out?
    #    
    #    Actually, a standard heuristic for sphere covering is:
    #    For each worst point c, find the nearest config points.
    #    Move those config points AWAY from c? No, that increases the distance to c.
    #    Move them TOWARDS c? That reduces the distance to c.
    #    BUT, moving them towards c might increase the distance to other worst points.
    #    
    #    Let's use a simpler, robust method: **Lloyd's algorithm for sphere covering** or **Repulsion**.
    #    Actually, the most effective simple method is:
    #    1. Compute the Voronoi cells (facets of convex hull).
    #    2. For each facet, compute the circumcenter.
    #    3. Move each point towards the average of the circumcenters of the facets it belongs to?
    #       This is like Lloyd's algorithm for k-means, but on the sphere.
    #       However, the "center" of a Voronoi cell for covering is the circumcenter (worst point).
    #       If we move points towards the circumcenters of their cells, we are effectively "pulling" the boundary towards the hole center, which should reduce the hole size.
    
    current_pts = pts.copy()
    temp = 0.05
    learning_rate = 0.01
    
    for step in range(5000):
        if time.time() > deadline:
            break
        
        worst_info = _get_worst_points(current_pts, num_worst=50)
        if not worst_info:
            break
            
        # Aggregate gradients for each point
        # For each worst point c, it is the circumcenter of facet F.
        # We want to reduce the distance from c to points in F.
        # Gradient direction for point p in F to reduce dist(p, c) is (c - p) projected to tangent space of p.
        # proj_p(c - p) = (c - p) - p(c^T p)
        
        grads = np.zeros_like(current_pts)
        
        for rad, c, simplex in worst_info:
            # Weight by how "bad" this facet is?
            # Or just uniform?
            # Let's use the radius as a weight.
            weight = rad
            for idx in simplex:
                p = current_pts[idx]
                # Vector from p to c
                v = c - p
                # Project to tangent space at p
                # p is unit vector. Tangent space is orthogonal to p.
                # proj = v - p * (p^T v)
                dot_pv = np.dot(p, v)
                proj = v - p * dot_pv
                grads[idx] += weight * proj
                
        # Normalize gradients
        grad_norms = np.linalg.norm(grads, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grads = grads / grad_norms
        
        # Update points
        # Riemannian gradient step
        new_pts = current_pts + learning_rate * grads
        
        # Project back to sphere
        new_pts = _normalize(new_pts)
        
        # Simulated Annealing perturbation
        # Add small random noise to escape local minima
        noise = np.random.randn(N, 3) * temp
        new_pts = _normalize(new_pts + noise)
        
        # Evaluate
        new_cr = _covering_radius_deg(new_pts)
        
        if new_cr < best_cr:
            best_pts = new_pts.copy()
            best_cr = new_cr
            current_pts = new_pts.copy()
            # Decay learning rate and temperature
            learning_rate *= 0.999
            temp *= 0.999
        else:
            # Accept with probability exp(-delta/temp) if temp > 0
            delta = new_cr - best_cr
            if temp > 1e-6 and np.random.rand() < np.exp(-delta / (temp * 100)): # Scale delta
                current_pts = new_pts.copy()
            else:
                # Reject, keep current
                pass
            # Decay temperature even if rejected to cool down
            temp *= 0.999
            
    return best_pts
# EVOLVE-BLOCK-END
