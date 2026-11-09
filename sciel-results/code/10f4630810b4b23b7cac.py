# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 610.0

def _vertex_score(distances):
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k - 1:k + 3]
    ratio = np.maximum(sel / D, D / sel)
    raw = np.mean(ratio)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(polygon):
    n = len(polygon)
    scores = []
    for i in range(n):
        dists = np.linalg.norm(polygon - polygon[i], axis=1)
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return float(np.min(scores))

def _is_valid(polygon, n=20):
    if not np.all(np.isfinite(polygon)):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-12:
                return False
    pts = np.vstack([polygon, polygon[:1]])
    edges = pts[1:] - pts[:-1]
    if np.any(np.linalg.norm(edges, axis=1) < 1e-12):
        return False
    cross = np.cross(edges[:-1], edges[1:])
    if np.any(np.abs(cross) < 1e-12):
        return False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    dangles = np.diff(angles)
    dangles = np.append(dangles, angles[0] - angles[-1])
    dangles = np.arctan2(np.sin(dangles), np.cos(dangles))
    total = np.sum(dangles)
    if abs(abs(total) - 2 * np.pi) > 1e-6:
        return False
    return True

def _normalized(polygon):
    c = np.mean(polygon, axis=0)
    p = polygon - c
    m = np.mean(np.linalg.norm(p, axis=1))
    if m <= 0.01:
        return None
    return p / m

def _objective(params):
    """Objective for optimization: negative min vertex score with convexity penalty."""
    r = params[:20]
    theta = params[20:]
    # Ensure positive radii
    r = np.maximum(r, 0.05)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    
    # Check convexity
    if not _is_valid(pts):
        # Penalty for non-convex
        pts_c = np.vstack([pts, pts[:1]])
        edges = pts_c[1:] - pts_c[:-1]
        cross = np.cross(edges[:-1], edges[1:])
        penalty = np.sum(np.minimum(cross, 0)) ** 2 + 10.0
        return penalty
    
    nm = _normalized(pts)
    if nm is None:
        return 10.0
    
    score = _min_vertex_score(nm)
    return -score

def _regular_polygon():
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    # Start with regular polygon
    reg = _regular_polygon()
    best = reg.copy()
    best_score = _min_vertex_score(reg)
    
    # Parameters: 20 radii + 20 angles
    n = 20
    a0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r0 = np.ones(n)
    
    # Try several optimization runs with different initializations
    rng = np.random.default_rng(42)
    
    for attempt in range(50):
        if time.time() > _DEADLINE - 10:
            break
        
        # Perturb the regular polygon
        if attempt == 0:
            r_init = r0.copy()
            theta_init = a0.copy()
        elif attempt < 10:
            # Small perturbations
            r_init = r0 + rng.normal(0, 0.05, n)
            theta_init = a0 + rng.normal(0, 0.02, n)
        elif attempt < 30:
            # Medium perturbations
            r_init = r0 + rng.normal(0, 0.15, n)
            theta_init = a0 + rng.normal(0, 0.05, n)
        else:
            # Larger perturbations but keep sorted angles
            r_init = r0 + rng.normal(0, 0.3, n)
            theta_init = a0 + rng.normal(0, 0.1, n)
            # Ensure angles are sorted
            theta_init = np.sort(theta_init)
        
        params_init = np.concatenate([r_init, theta_init])
        
        try:
            result = minimize(
                _objective,
                params_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_score:
                r_opt = np.maximum(result.x[:20], 0.05)
                theta_opt = result.x[20:]
                pts = np.stack([r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)], axis=1)
                
                if _is_valid(pts):
                    nm = _normalized(pts)
                    if nm is not None:
                        s = _min_vertex_score(nm)
                        if s > best_score:
                            best_score = s
                            best = nm
        except Exception:
            continue
    
    return best

# EVOLVE-BLOCK-END
