"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import scipy.optimize as optimize

def _normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere 
    to the nearest point in pts. This occurs at the circumcenters of the Delaunay triangles 
    (facets of the convex hull).
    """
    try:
        hull = ConvexHull(pts)
        tri_idx = hull.simplices[:, :3]
        
        max_angle = 0.0
        for tri in tri_idx:
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            
            # Normal to the plane of the triangle
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-10:
                continue
            
            # The circumcenter is parallel to the normal n
            # We have two candidates: C1 = n/n_norm and C2 = -n/n_norm
            C1 = n / n_norm
            C2 = -C1
            
            # Calculate angular distance to p1 for both candidates
            dot1 = np.clip(C1 @ p1, -1.0, 1.0)
            angle1 = np.arccos(dot1)
            
            dot2 = np.clip(C2 @ p1, -1.0, 1.0)
            angle2 = np.arccos(dot2)
            
            # The correct circumcenter is the one with the smaller angle (closer to the triangle)
            angle = min(angle1, angle2)
            
            if angle > max_angle:
                max_angle = angle
                
        return max_angle
    except Exception:
        # Fallback to sampling if hull fails
        i = np.arange(20000) + 0.5
        z = 1.0 - 2.0 * i / 20000
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = samples @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(max_dots)
        return np.max(angles)

def _smooth_covering_radius(pts):
    """
    Compute a smooth approximation of the covering radius for gradient-based optimization.
    Uses a soft-max over the angular distances to the nearest point.
    """
    # Sample points on the sphere
    i = np.arange(5000) + 0.5
    z = 1.0 - 2.0 * i / 5000
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    # Soft-max approximation
    beta = 50.0
    max_angle = np.max(angles, axis=1)
    exp_angles = np.exp(beta * (angles - max_angle[:, np.newaxis]))
    soft_max = max_angle - np.log(np.sum(exp_angles, axis=1)) / beta
    
    return np.mean(soft_max)

def _repulsive_relaxation(pts, max_iter=2000, dt=0.05):
    """
    Optimize points using repulsive forces.
    """
    pts = _normalize(pts.copy())
    for i in range(max_iter):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        forces = diff * inv_dist_cubed[:, :, np.newaxis]
        net_forces = np.sum(forces, axis=1)
        
        pts = pts + dt * net_forces
        pts = _normalize(pts)
        
        dt *= 0.999
        
    return pts

def _optimize_nm(pts, max_time=10.0):
    """
    Local optimization using Nelder-Mead on the exact covering radius.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() > start_time + max_time:
            return 1e6
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius_exact(p)
    
    try:
        # Nelder-Mead settings
        # maxiter is high, but we rely on time check
        result = optimize.minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < 1e5:
            best_pts = _normalize(result.x.reshape(N, 3))
            best_cr = _covering_radius_exact(best_pts)
            return best_pts, best_cr
    except Exception:
        pass
        
    # Fallback to current pts
    return pts, _covering_radius_exact(pts)

def _optimize_bfgs(pts, max_time=10.0):
    """
    Local optimization using BFGS on the smooth covering radius.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() > start_time + max_time:
            return 1e6
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _smooth_covering_radius(p)
    
    def gradient(x):
        if time.time() > start_time + max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _normalize(p)
        
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            f_plus = objective(x_plus)
            f_minus = objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    try:
        result = optimize.minimize(objective, x0, method='BFGS', jac=gradient,
                                   options={'maxiter': 100, 'gtol': 1e-6})
        
        if result.fun < 1e5:
            best_pts = _normalize(result.x.reshape(N, 3))
            best_cr = _covering_radius_exact(best_pts)
            return best_pts, best_cr
    except Exception:
        pass
        
    # Fallback to current pts
    return pts, _covering_radius_exact(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_cr = np.inf
    
    # Multiple restarts
    for restart in range(10):
        if time.time() > deadline - 15.0:
            break
        
        # Initialization
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random
            pts = np.random.randn(N, 3)
        else:
            # Perturb best
            if best_pts is not None:
                pts = best_pts + np.random.randn(N, 3) * 0.1
            else:
                pts = np.random.randn(N, 3)
        
        pts = _normalize(pts)
        
        # Phase 1: Repulsive Relaxation
        relax_time = min(5.0, deadline - 20.0 - (time.time() - start_time))
        if relax_time > 1.0:
            pts = _repulsive_relaxation(pts, max_iter=1000, dt=0.05)
        
        # Phase 2: Local Polish with BFGS on smooth objective
        polish_time = min(5.0, deadline - 10.0 - (time.time() - start_time))
        if polish_time > 1.0:
            pts, cr = _optimize_bfgs(pts, max_time=polish_time)
        else:
            cr = _covering_radius_exact(pts)
        
        # Phase 3: Final Polish with Nelder-Mead on exact objective
        polish_time2 = min(10.0, deadline - 5.0 - (time.time() - start_time))
        if polish_time2 > 1.0:
            pts, cr = _optimize_nm(pts, max_time=polish_time2)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
        # Check time
        if time.time() > deadline - 15.0:
            break
            
    # Final verification
    if best_pts is None:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
