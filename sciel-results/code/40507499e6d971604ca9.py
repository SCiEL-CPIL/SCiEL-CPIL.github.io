"""Heilbronn 13-point convex region optimization."""
import numpy as np
import time
from scipy.optimize import minimize, anneal

N = 13
DEADLINE = 145.0  # safety margin before 150s kill


def _triangle_areas(pts):
    """Compute all triangle areas from N points (N=13 -> 286 triangles)."""
    n = len(pts)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                areas[i, j, k] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return areas


def _hull_area(pts):
    """Convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume = area
    except Exception:
        # Fallback: shoelace on all points (assumes convex)
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0


def _score(pts):
    """Minimum triangle area divided by convex hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull = _hull_area(pts)
    if hull < 1e-12:
        return 0.0
    return min_area / hull


def _soft_min_objective(x, temperature=0.1):
    """Smooth soft-min of triangle areas, penalized by hull area."""
    # x encodes angles (N) and radii (N)
    angles = x[:N]
    radii = x[N:2*N]
    
    # Ensure angles are sorted and within [0, 2*pi)
    angles = np.sort(angles)
    # Normalize angles to [0, 2*pi)
    angles = angles - angles[0]
    if angles[-1] > 2 * np.pi - 1e-6:
        angles = angles / angles[-1] * 2 * np.pi
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + (angles - np.linspace(0, 2 * np.pi, N, endpoint=False))
    
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    areas = _triangle_areas(pts)
    # Soft min: -T * log(sum(exp(-a/T)))
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-(areas - min_a) / temperature)))
    soft_min = min_a - temperature * log_sum
    
    hull = _hull_area(pts)
    if hull < 1e-12:
        return 1e10
    return -soft_min / hull


def _init_regular_perturbed(rng, scale=0.1):
    """Regular 13-gon with random radial and angular perturbations."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    
    # Add perturbations
    if scale > 0:
        angles = angles + rng.gauss(0, scale, N)
        radii = 1.0 + rng.gauss(0, scale * 0.5, N)
        radii = np.clip(radii, 0.3, 1.7)
    
    return np.concatenate([angles, radii])


def _polish_nm(pts, max_iter=500):
    """Nelder-Mead polish on the raw min-triangle-area / hull-area objective."""
    def obj(x):
        a = x[:N]
        r = x[N:2*N]
        a = np.sort(a)
        a = a - a[0]
        if a[-1] > 2 * np.pi - 1e-6:
            a = a / a[-1] * 2 * np.pi
        p = np.column_stack([r * np.cos(a), r * np.sin(a)])
        return -_score(p)
    
    x0 = np.concatenate([np.sort(np.arctan2(pts[:,1], pts[:,0])),
                         np.linalg.norm(pts, axis=1)])
    x0 = x0 - x0[0]
    if x0[-1] > 2*np.pi - 1e-6:
        x0 = x0 / x0[-1] * 2*np.pi
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        a = np.sort(res.x[:N])
        a = a - a[0]
        if a[-1] > 2*np.pi - 1e-6:
            a = a / a[-1] * 2*np.pi
        r = res.x[N:2*N]
        return np.column_stack([r * np.cos(a), r * np.sin(a)])
    except Exception:
        return pts


def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = 0.0
    
    # Generate diverse initial points
    n_starts = 15
    for i in range(n_starts):
        if time.time() - t_start > DEADLINE - 10:
            break
        
        if i == 0:
            x0 = _init_regular_perturbed(rng, scale=0.0)
        else:
            x0 = _init_regular_perturbed(rng, scale=0.05 + 0.03 * i)
        
        # L-BFGS-B optimization
        try:
            res = minimize(_soft_min_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            # Convert back to points
            a = np.sort(res.x[:N])
            a = a - a[0]
            if a[-1] > 2*np.pi - 1e-6:
                a = a / a[-1] * 2*np.pi
            r = res.x[N:2*N]
            pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
            
            # Polish with Nelder-Mead
            pts = _polish_nm(pts, max_iter=300)
            
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Final polish on best
    if best_pts is not None and time.time() - t_start < DEADLINE - 5:
        best_pts = _polish_nm(best_pts, max_iter=1000)
        best_score = _score(best_pts)
    
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
