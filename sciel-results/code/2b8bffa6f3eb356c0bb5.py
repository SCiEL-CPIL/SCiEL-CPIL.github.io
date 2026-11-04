"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    # C(13,3) = 286, so a direct loop is fine
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area of triangle with vertices pts[i], pts[j], pts[k]
                # Using the cross product formula: 0.5 * |cross(v1, v2)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _score(pts):
    """
    Score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = _min_triangle_area(pts)
    if min_area <= 0:
        return 0.0
    
    return min_area / hull_area

def _objective(x):
    """
    Objective for minimization: negative of the score.
    x is a flat array of 2*N coordinates.
    """
    pts = x.reshape(-1, 2)
    s = _score(pts)
    return -s

def _nelder_mead_polish(pts0, maxiter=200, tol=1e-10):
    """
    Polish a starting point using Nelder-Mead on the exact nonsmooth objective.
    """
    x0 = pts0.flatten()
    res = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    best_pts = res.x.reshape(-1, 2)
    best_score = -res.fun
    return best_pts, best_score

def _generate_initial_candidates(rng, n_starts=20):
    """
    Generate diverse initial candidates.
    Mix of random uniform, and some structured patterns.
    """
    candidates = []
    
    # 1. Pure random uniform in [0,1]^2
    for _ in range(n_starts // 2):
        pts = rng.random((N, 2))
        candidates.append(pts)
    
    # 2. Random on a circle (with jitter)
    for _ in range(n_starts // 4):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.random(N) * 0.1
        radii = 1.0 + rng.random(N) * 0.1
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Normalize to unit area roughly
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-6:
                pts = pts / np.sqrt(area)
            else:
                pts = pts / np.sqrt(2)  # fallback
        except Exception:
            pass
        candidates.append(pts)
    
    # 3. Random on a square boundary + center
    for _ in range(n_starts // 4):
        pts = np.zeros((N, 2))
        # Place some on corners
        corners = [[0,0],[1,0],[0,1],[1,1]]
        for i in range(4):
            pts[i] = corners[i]
        # Place rest randomly on edges
        for i in range(4, N):
            edge = rng.randint(0, 3)
            t = rng.random()
            if edge == 0:
                pts[i] = [t, 0]
            elif edge == 1:
                pts[i] = [1, t]
            elif edge == 2:
                pts[i] = [t, 1]
            else:
                pts[i] = [0, t]
        # Normalize
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-6:
                pts = pts / np.sqrt(area)
            else:
                pts = pts / np.sqrt(2)
        except Exception:
            pass
        candidates.append(pts)
    
    # 4. Regular polygon with jitter
    for _ in range(max(1, n_starts // 4)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.random(N) * 0.05
        radii = 1.0 + rng.random(N) * 0.05
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-6:
                pts = pts / np.sqrt(area)
            else:
                pts = pts / np.sqrt(2)
        except Exception:
            pass
        candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(N)
    
    # Generate initial candidates
    n_starts = 30
    candidates = _generate_initial_candidates(rng, n_starts)
    
    best_pts = None
    best_score = -1.0
    
    # First pass: evaluate all candidates and polish the top ones
    scored_candidates = []
    for i, pts in enumerate(candidates):
        if time.time() > deadline - 60:  # Stop generating if time is tight
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        scored_candidates.append((s, i, pts))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Polish the top 5 candidates with Nelder-Mead
    top_k = min(5, len(scored_candidates))
    for s, i, pts in scored_candidates[:top_k]:
        if time.time() > deadline - 20:
            break
        polished_pts, polished_score = _nelder_mead_polish(pts, maxiter=100, tol=1e-9)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
    
    # If we have time, do a second round with more random starts
    if time.time() < deadline - 30:
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            pts = rng.random((N, 2))
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
            # Quick polish if promising
            if s > 0.5 * best_score:
                polished_pts, polished_score = _nelder_mead_polish(pts, maxiter=50, tol=1e-8)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished_pts.copy()
    
    # Final high-precision polish on the best
    if time.time() < deadline - 5 and best_pts is not None:
        final_pts, final_score = _nelder_mead_polish(best_pts, maxiter=200, tol=1e-11)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
    
    if best_pts is None:
        # Fallback: return random points
        return rng.random((N, 2))
    
    return best_pts
# EVOLVE-BLOCK-END
