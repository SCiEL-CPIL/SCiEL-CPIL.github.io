"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

_start_time = time.time()
TIME_LIMIT = 130  # safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return TIME_LIMIT - _elapsed()

def _min_triangle_area(pts):
    """Compute minimum triangle area over all C(N,3) triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a

def _negative_score(pts):
    """Objective to minimize."""
    return -_score(pts)

def construct():
    if _remaining() < 5:
        # Not enough time; return regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Initial: regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts0 = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(pts0)
    best_pts = pts0.copy()
    
    # Try several perturbations of the regular polygon and optimize
    rng = np.random.RandomState(42)
    
    # Differential evolution on the full 2N coordinates
    # Fix p0 at origin to remove translation invariance
    # Fix p1's y=0 and x>0 to remove rotation (partially)
    
    def _decode(x):
        """Decode parameter vector to points with p0 fixed at origin."""
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        # x has 2*(N-1) elements for points 1..N-1
        for i in range(N-1):
            pts[i+1] = x[2*i:2*i+2]
        return pts
    
    def _objective(x):
        pts = _decode(x)
        # Check hull area to avoid degenerate
        try:
            hull = ConvexHull(pts)
            if hull.volume < 1e-6:
                return 1e10
        except Exception:
            return 1e10
        return -_score(pts)
    
    # Bounds: points in [-2, 2]
    bounds = [(-2.0, 2.0)] * (2*(N-1))
    
    # Try differential evolution with limited budget
    if _remaining() > 30:
        try:
            result = differential_evolution(
                _objective, bounds,
                seed=42, maxiter=50, tol=1e-6,
                popsize=15, init='latinhypercube',
                polish=False
            )
            if _remaining() > 10:
                pts_de = _decode(result.x)
                s_de = _score(pts_de)
                if s_de > best_score:
                    best_score = s_de
                    best_pts = pts_de.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best point
    if _remaining() > 10:
        try:
            x0 = np.zeros(2*(N-1))
            for i in range(N-1):
                x0[2*i:2*i+2] = best_pts[i+1]
            
            result = minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            pts_nm = _decode(result.x)
            s_nm = _score(pts_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_pts = pts_nm.copy()
        except Exception:
            pass
    
    # Try a few random restarts with Nelder-Mead if time permits
    attempts = 0
    while _remaining() > 15 and attempts < 3:
        attempts += 1
        # Perturb best points
        perturbation_scale = 0.3
        pts_try = best_pts + rng.randn(N, 2) * perturbation_scale
        # Re-center to keep p0 near origin
        pts_try -= pts_try[0]
        
        x0 = np.zeros(2*(N-1))
        for i in range(N-1):
            x0[2*i:2*i+2] = pts_try[i+1]
        
        try:
            result = minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            pts_opt = _decode(result.x)
            s_opt = _score(pts_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
