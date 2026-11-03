"""
Strategy: Direct Facet-Targeting Optimization.
Instead of Lloyd's (centroid-based), we identify the facet with the largest 
circumradius and apply a targeted gradient step to its vertices to shrink it.
We also apply repulsive forces to maintain separation.
This directly minimizes the max covering radius.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _get_max_facet_and_radius(pts):
    """
    Compute the covering radius and identify the facet (indices) that achieves it.
    Returns: (max_radius_rad, facet_indices_tuple, circumcenter_vector)
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, (0, 1, 2), np.array([1.0, 0, 0])
    
    max_radius = 0.0
    max_facet = None
    max_cc = None
    
    for simplex in hull.simplices:
        i, j, k = simplex
        a = pts[i]
        b = pts[j]
        c = pts[k]
        
        # Circumcenter of the triangle on the sphere
        # x is parallel to (a-b) x (a-c)
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # Determine correct direction (outward normal side)
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(x, n) > 0:
            cc = x
        else:
            cc = -x
            
        # Radius is angular distance from cc to any vertex
        dot_a = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(dot_a)
        
        if r > max_radius:
            max_radius = r
            max_facet = (i, j, k)
            max_cc = cc
            
    return max_radius, max_facet, max_cc

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    r, _, _ = _get_max_facet_and_radius(pts)
    return r

def _local_optimize_facet(pts, max_time):
    """
    Local optimization by targeting the largest facet.
    Moves vertices of the largest facet towards the circumcenter (to shrink it)
    and away from other points (to prevent collapse).
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    lr = 0.05  # Learning rate
    cooldown = 0.99
    
    while time.time() - start < max_time and time.time() < _DEADLINE:
        obj, facet, cc = _get_max_facet_and_radius(pts)
        
        if facet is None:
            break
            
        i, j, k = facet
        a, b, c = pts[i], pts[j], pts[k]
        
        # Gradient direction for the largest facet:
        # We want to move a, b, c such that the circumradius decreases.
        # Moving them towards the circumcenter 'cc' generally shrinks the triangle's "spread"
        # relative to that center, but on the sphere, we need to be careful.
        # Actually, if we move vertices towards the circumcenter, the triangle becomes "smaller" 
        # in terms of angular extent from that center? No, the circumcenter is fixed by the vertices.
        # Moving vertices towards the *current* circumcenter reduces the side lengths, 
        # which reduces the circumradius.
        
        # Force 1: Attraction to circumcenter (shrink the hole)
        f_a = cc - a
        f_b = cc - b
        f_c = cc - c
        
        # Force 2: Repulsion from all other points to maintain distribution
        # This prevents the 3 points from collapsing onto cc, which would ruin the covering elsewhere
        rep_a = np.zeros(3)
        rep_b = np.zeros(3)
        rep_c = np.zeros(3)
        
        for m in range(N):
            if m in (i, j, k):
                continue
            pm = pts[m]
            # Repulsive force: move away from pm
            # Use a simple inverse distance or dot product based repulsion
            # If dot < 0, they are far. If dot > 0, they are close.
            # We want to maximize min distance.
            # Gradient of -dot(a, pm) w.r.t a is -pm.
            # So we add -pm to the force? No, we want to move a away from pm.
            # Direction away from pm is -pm (if a is at origin? No).
            # Vector from pm to a is a - pm.
            # We want to add a component of (a - pm).
            # However, on the sphere, the "away" direction is the projection of -pm onto the tangent plane of a.
            # Tangent space at a: T_a = {v | v . a = 0}.
            # Projection of -pm onto T_a: -pm - (-pm . a) * a = -pm + (pm . a) * a.
            
            dot_apm = np.dot(a, pm)
            dot_bpm = np.dot(b, pm)
            dot_cpm = np.dot(c, pm)
            
            # Weight repulsion by how close they are (stronger repulsion when close)
            # Use a soft threshold
            w_a = max(0, 1.0 - dot_apm / 0.5) if dot_apm > 0 else 0.1
            w_b = max(0, 1.0 - dot_bpm / 0.5) if dot_bpm > 0 else 0.1
            w_c = max(0, 1.0 - dot_cpm / 0.5) if dot_cpm > 0 else 0.1
            
            rep_a += w_a * (-pm + dot_apm * a)
            rep_b += w_b * (-pm + dot_bpm * b)
            rep_c += w_c * (-pm + dot_cpm * c)
            
        # Combine forces
        # Attraction to cc is strong to fix the bottleneck
        # Repulsion is weaker to maintain structure
        step_a = 1.0 * f_a + 0.5 * rep_a
        step_b = 1.0 * f_b + 0.5 * rep_b
        step_c = 1.0 * f_c + 0.5 * rep_c
        
        # Project steps onto tangent space
        step_a = step_a - np.dot(step_a, a) * a
        step_b = step_b - np.dot(step_b, b) * b
        step_c = step_c - np.dot(step_c, c) * c
        
        # Normalize steps to control magnitude
        norm_sa = np.linalg.norm(step_a)
        norm_sb = np.linalg.norm(step_b)
        norm_sc = np.linalg.norm(step_c)
        
        if norm_sa > 1e-9: step_a = step_a / norm_sa
        else: step_a = np.zeros(3)
        if norm_sb > 1e-9: step_b = step_b / norm_sb
        else: step_b = np.zeros(3)
        if norm_sc > 1e-9: step_c = step_c / norm_sc
        else: step_c = np.zeros(3)
        
        # Apply update
        new_pts = pts.copy()
        new_pts[i] = a + lr * step_a
        new_pts[j] = b + lr * step_b
        new_pts[k] = c + lr * step_c
        
        new_pts = _to_sphere(new_pts)
        
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            lr = min(lr * 1.01, 0.1) # Slightly increase if improving
        else:
            # If not improving, reduce learning rate and maybe perturb
            lr *= cooldown
            if lr < 0.001:
                break
            # Small random perturbation to escape local minima
            noise = np.random.randn(N, 3) * 0.01
            pts = _to_sphere(pts + noise)
            current_obj = _objective(pts)
            
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses multi-start with diverse initializations and facet-targeting local optimization.
    """
    best_pts = None
    best_obj = np.pi
    
    # Define diverse initial configurations
    def init_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def init_random():
        pts = np.random.randn(N, 3)
        return _to_sphere(pts)

    def init_clustered():
        # Start with clusters to test robustness
        centers = np.random.randn(5, 3)
        centers = _to_sphere(centers)
        pts = np.vstack([c + np.random.randn(10, 3) * 0.1 for c in centers])
        return _to_sphere(pts)

    initializers = [init_fibonacci, init_random, init_clustered, init_random, init_random]
    
    np.random.seed(42)
    
    for init_func in initializers:
        if time.time() > _DEADLINE - 15:
            break
            
        pts = init_func()
        pts = _to_sphere(pts)
        
        # Quick local optimization
        pts, obj = _local_optimize_facet(pts, max_time=20)
        
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
        # If we have significant time left, do a second pass with higher precision
        if time.time() < _DEADLINE - 30:
             pts2, obj2 = _local_optimize_facet(pts, max_time=15)
             if obj2 < best_obj:
                 best_obj = obj2
                 best_pts = pts2.copy()
                 
    # Final validation and cleanup
    if best_pts is None:
        best_pts = init_fibonacci()
        
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
