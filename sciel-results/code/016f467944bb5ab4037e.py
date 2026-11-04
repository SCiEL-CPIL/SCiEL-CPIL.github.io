"""Improved program for sphere covering with 130 points."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, minimize_scalar

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum over all spherical facets of the circumradius.
    For a convex hull of points on the unit sphere, each triangular facet has a circumcenter
    (on the sphere) which is the point farthest from the three vertices.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    # Get the triangular facets
    simplices = hull.simplices  # shape (F, 3)
    
    # For each triangle (a, b, c) on the unit sphere, compute the spherical circumradius.
    # The circumcenter of a spherical triangle is the normalized cross product of the three vertices
    # (which gives the normal to the plane through them, i.e., the direction of the circumcenter).
    # Actually, for a triangle on the unit sphere, the circumcenter is the normalized vector
    # that is equidistant from all three vertices. This is the normalized cross product of (b-a) and (c-a)
    # projected... Actually simpler: the circumcenter direction is proportional to a×b + b×c + c×a? No.
    
    # Better approach: For three points a, b, c on the unit sphere, the circumcenter (as a direction)
    # is the normalized vector: (a × b) × (b × c) ... no.
    # The circumcenter is the point x (unit vector) such that x·a = x·b = x·c.
    # This means x is orthogonal to (a-b) and (a-c), so x is parallel to (a-b) × (a-c).
    # The circumradius is the angle between x and any of a, b, c.
    
    a = pts[simplices[:, 0]]  # (F, 3)
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Compute circumcenter direction for each triangle
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)  # (F, 3)
    
    # Normalize to get circumcenter direction
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    centers = cross / norms
    
    # The circumradius is the angle between center and any vertex (say a)
    dots = np.sum(centers * a, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    circumradii = np.arccos(dots)
    
    # The covering radius is the maximum circumradius over all facets
    return np.max(circumradii)


def _covering_radius_degrees(pts):
    """Covering radius in degrees."""
    return np.degrees(_covering_radius(pts))


def _smooth_covering_surrogate(pts, temperature=0.05):
    """
    A smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used with L-BFGS-B.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros(len(pts))
    
    simplices = hull.simplices
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    centers = cross / norms
    
    dots = np.sum(centers * a, axis=1)
    dots = np.clip(dots, -1.0 + 1e-10, 1.0 - 1e-10)
    circumradii = np.arccos(dots)
    
    # Log-sum-exp for smooth max
    max_c = np.max(circumradii)
    if temperature < 1e-8:
        return max_c, np.zeros(len(pts))
    
    # Soft max: log(sum(exp(circumradii/temperature))) * temperature
    shifted = circumradii / temperature
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    smooth_max = temperature * (log_sum + np.max(shifted))
    
    # Gradient approximation via finite differences is expensive; 
    # instead we'll use a differentiable formulation directly.
    return smooth_max, None


def _optimize_lloyd(pts, max_iter=50, tol=1e-10):
    """
    Lloyd-type iteration for sphere covering:
    1. Compute Voronoi cells (via Delaunay on sphere)
    2. Move each point to the centroid of its Voronoi cell
    """
    pts = _normalize(pts.copy())
    
    for it in range(max_iter):
        if _time_left() < 10:
            break
        
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(pts)
        except Exception:
            break
        
        # For each point, find its Voronoi cell by finding adjacent triangles
        # and computing the cell centroid
        new_pts = np.zeros_like(pts)
        
        for i in range(len(pts)):
            # Find triangles containing point i
            # This is expensive; use a simpler approach
            pass
        
        # Alternative: use the circumcenters of adjacent triangles as "Voronoi vertices"
        # and compute the centroid of the Voronoi cell
        # For simplicity, use a gradient-based approach instead
        break
    
    return pts


def _gradient_step(pts, temperature=0.05, lr=0.01, max_iter=100):
    """
    Gradient descent on the smooth covering radius surrogate.
    Uses the fact that the gradient of the circumradius w.r.t. vertex positions
    can be computed analytically.
    """
    pts = _normalize(pts.copy())
    
    for it in range(max_iter):
        if _time_left() < 15:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        simplices = hull.simplices
        F = len(simplices)
        
        a_idx = simplices[:, 0]
        b_idx = simplices[:, 1]
        c_idx = simplices[:, 2]
        
        a = pts[a_idx]
        b = pts[b_idx]
        c = pts[c_idx]
        
        # Compute circumcenters and circumradii
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        centers = cross / norms
        
        dots = np.sum(centers * a, axis=1)
        dots = np.clip(dots, -1.0 + 1e-10, 1.0 - 1e-10)
        circumradii = np.arccos(dots)
        
        # Soft-max weights
        shifted = circumradii / temperature
        shifted -= np.max(shifted)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Gradient of circumradius w.r.t. each vertex
        # For a triangle with circumcenter x and circumradius r = arccos(x·a):
        # dr/da = -x / sqrt(1 - (x·a)^2) = -x / sin(r)
        # But we need to account for the fact that x depends on a, b, c.
        # This is complex; use finite differences for simplicity.
        
        # Use finite difference gradient
        eps = 1e-5
        grad = np.zeros_like(pts)
        
        # Only compute gradient for the worst few facets for speed
        n_worst = min(20, F)
        worst_idx = np.argsort(circumradii)[-n_worst:]
        worst_weights = weights[worst_idx] / np.sum(weights[worst_idx])
        
        for fi in worst_idx:
            w = worst_weights[list(worst_idx).index(fi)]
            if w < 1e-6:
                continue
            
            fi_a, fi_b, fi_c = simplices[fi]
            
            for vi, vid in [(fi_a, a_idx[fi]), (fi_b, b_idx[fi]), (fi_c, c_idx[fi])]:
                # Perturb vertex vid
                for dim in range(3):
                    pts[vid, dim] += eps
                    pts = _normalize(pts)
                    # Recompute circumradius for this facet
                    aa = pts[fi_a]
                    bb = pts[fi_b]
                    cc = pts[fi_c]
                    v1p = aa - bb
                    v2p = aa - cc
                    crossp = np.cross(v1p, v2p)
                    normp = np.linalg.norm(crossp)
                    if normp < 1e-12:
                        pts[vid, dim] -= eps
                        continue
                    centerp = crossp / normp
                    dotp = np.clip(np.sum(centerp * aa), -1.0 + 1e-10, 1.0 - 1e-10)
                    rp = np.arccos(dotp)
                    
                    pts[vid, dim] -= eps
                    pts = _normalize(pts)
                    
                    grad[vid, dim] += w * (rp - circumradii[fi]) / eps
        
        # Project gradient onto tangent space and take step
        for i in range(len(pts)):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        pts = _normalize(pts - lr * grad)
    
    return pts


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Uses multi-phase optimization:
    1. Fibonacci spiral initialization
    2. L-BFGS-B on smooth surrogate
    3. Local refinement on worst facets
    4. Final polish
    """
    # Phase 1: Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 2: L-BFGS-B optimization
    if _time_left() > 60:
        # Flatten for optimization
        x0 = pts.flatten()
        
        def objective(x):
            if _time_left() < 5:
                return 1e6
            p = _normalize(x.reshape(-1, 3))
            cr, _ = _smooth_covering_surrogate(p, temperature=0.05)
            return cr
        
        def gradient(x):
            if _time_left() < 5:
                return np.zeros_like(x)
            p = _normalize(x.reshape(-1, 3))
            # Finite difference gradient
            eps = 1e-6
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                p_plus = _normalize(x_plus.reshape(-1, 3))
                f_plus, _ = _smooth_covering_surrogate(p_plus, temperature=0.05)
                
                x_minus = x.copy()
                x_minus[i] -= eps
                p_minus = _normalize(x_minus.reshape(-1, 3))
                f_minus, _ = _smooth_covering_surrogate(p_minus, temperature=0.05)
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad
        
        result = minimize(
            objective, x0, method='L-BFGS-B', jac=gradient,
            options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 5000}
        )
        
        if result.success or result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(-1, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
    
    # Phase 3: Local refinement - identify worst facets and optimize those vertices
    if _time_left() > 30:
        for refine_iter in range(5):
            if _time_left() < 15:
                break
            
            pts = best_pts.copy()
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            
            simplices = hull.simplices
            a = pts[simplices[:, 0]]
            b = pts[simplices[:, 1]]
            c = pts[simplices[:, 2]]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            centers = cross / norms
            
            dots = np.sum(centers * a, axis=1)
            dots = np.clip(dots, -1.0 + 1e-10, 1.0 - 1e-10)
            circumradii = np.arccos(dots)
            
            # Find the worst facet
            worst_f = np.argmax(circumradii)
            worst_r = circumradii[worst_f]
            
            # Optimize the three vertices of the worst facet
            fa, fb, fc = simplices[worst_f]
            
            def facet_obj(x):
                p = pts.copy()
                p[fa] = _normalize(x[0:3].reshape(1, -1))[0]
                p[fb] = _normalize(x[3:6].reshape(1, -1))[0]
                p[fc] = _normalize(x[6:9].reshape(1, -1))[0]
                p = _normalize(p)
                try:
                    h = ConvexHull(p)
                    s = h.simplices
                    aa = p[s[:, 0]]
                    bb = p[s[:, 1]]
                    cc = p[s[:, 2]]
                    v1p = aa - bb
                    v2p = aa - cc
                    crossp = np.cross(v1p, v2p)
                    normsp = np.linalg.norm(crossp, axis=1, keepdims=True)
                    normsp[normsp < 1e-12] = 1.0
                    centp = crossp / normsp
                    dotsp = np.sum(centp * aa, axis=1)
                    dotsp = np.clip(dotsp, -1.0 + 1e-10, 1.0 - 1e-10)
                    crp = np.arccos(dotsp)
                    return np.max(crp)
                except Exception:
                    return np.pi
            
            x0 = np.concatenate([pts[fa], pts[fb], pts[fc]])
            res = minimize(facet_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.fun < worst_r:
                pts[fa] = _normalize(res.x[0:3].reshape(1, -1))[0]
                pts[fb] = _normalize(res.x[3:6].reshape(1, -1))[0]
                pts[fc] = _normalize(res.x[6:9].reshape(1, -1))[0]
                pts = _normalize(pts)
                
                new_cr = _covering_radius(pts)
                if new_cr < best_cr:
                    best_pts = pts.copy()
                    best_cr = new_cr
    
    # Phase 4: Final polish with L-BFGS on the full configuration
    if _time_left() > 20:
        x0 = best_pts.flatten()
        
        def final_obj(x):
            if _time_left() < 3:
                return 1e6
            p = _normalize(x.reshape(-1, 3))
            cr, _ = _smooth_covering_surrogate(p, temperature=0.02)
            return cr
        
        result = minimize(
            final_obj, x0, method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        if result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(-1, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
    
    # Ensure all points are on the unit sphere
    best_pts = _normalize(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
