"""
Heilbronn convex regions: 13 points, maximize min triangle area / hull area.
Strategy: 
1. Use a differentiable soft-min (log-sum-exp) objective for the minimum triangle area.
2. Optimize using L-BFGS-B with annealing (increasing beta) to escape local minima.
3. Use diverse initializations (symmetric and random) to cover the search space.
4. Final polish with BFGS on the exact non-differentiable objective.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
# Precompute triangle indices
TRIS = list(combinations(range(N), 3))
TRI_I = np.array([t[0] for t in TRIS], dtype=np.int32)
TRI_J = np.array([t[1] for t in TRIS], dtype=np.int32)
TRI_K = np.array([t[2] for t in TRIS], dtype=np.int32)

def _hull_area(pts):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _min_triangle_area(pts):
    """Calculate exact minimum triangle area."""
    x = pts[:, 0]
    y = pts[:, 1]
    v1x = x[TRI_J] - x[TRI_I]
    v1y = y[TRI_J] - y[TRI_I]
    v2x = x[TRI_K] - x[TRI_I]
    v2y = y[TRI_K] - y[TRI_I]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    return np.min(areas)

def _exact_score(pts):
    """Exact score: min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _soft_min_objective(pts_flat, beta):
    """
    Differentiable objective: -soft_min_triangle_area / hull_area.
    We minimize this.
    """
    pts = pts_flat.reshape(-1, 2)
    # Center points to help with numerical stability and hull calculation
    pts = pts - pts.mean(axis=0)
    
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 1e6

    x = pts[:, 0]
    y = pts[:, 1]
    v1x = x[TRI_J] - x[TRI_I]
    v1y = y[TRI_J] - y[TRI_I]
    v2x = x[TRI_K] - x[TRI_I]
    v2y = y[TRI_K] - y[TRI_I]
    cross = np.abs(v1x * v2y - v1y * v2x)
    areas = 0.5 * cross
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, use log-sum-exp trick
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    if max_val < -1e6:
        return 1e6
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    score = soft_min_area / hull_a
    return -score

def _exact_objective(pts_flat):
    """Exact objective for final polish."""
    pts = pts_flat.reshape(-1, 2)
    score = _exact_score(pts)
    if score == 0.0:
        return 1e6
    return -score

def _normalize(pts):
    """Normalize to unit hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return pts
    scale = 1.0 / np.sqrt(hull_a)
    return pts * scale

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    inits.append(_regular_polygon(13, 1.0, 0.0))
    
    # 2. 1 Center + 12 Outer
    outer12 = _regular_polygon(12, 1.0, 0.0)
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([center, outer12]))
    
    # 3. 1 Center + 6 Inner + 6 Outer
    for r_in in [0.3, 0.5, 0.7]:
        outer6 = _regular_polygon(6, 1.0, 0.0)
        inner6 = _regular_polygon(6, r_in, np.pi/6) # Rotated to interleave
        center = np.array([[0.0, 0.0]])
        inits.append(np.vstack([center, outer6, inner6]))
        
    # 4. 2 "Centers" (Dipole) + 11 Outer? No, 13 is prime. 
    # Let's try 1 Center + 11 Outer + 1 Inner?
    # Actually, just random points are good too.
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 2) * 0.5
        inits.append(pts)
        
    # Also try a "star" pattern: alternating radii
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.ones(N)
    radii[::2] = 1.0
    radii[1::2] = 0.5
    inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    best_score = -np.inf
    best_pts = inits[0].copy()
    
    # Optimization Loop
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        current_pts = init.copy()
        current_pts_flat = current_pts.flatten()
        
        # Annealing schedule for beta
        betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                res = minimize(
                    _soft_min_objective, 
                    current_pts_flat, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_pts_flat = res.x
                current_pts = current_pts_flat.reshape(-1, 2)
            except Exception:
                pass
                
        # Evaluate exact score for this candidate
        score = _exact_score(current_pts)
        if score > best_score:
            best_score = score
            best_pts = current_pts.copy()
            
    # Final Polish with BFGS on exact objective
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                _exact_objective, 
                best_pts.flatten(), 
                method='BFGS',
                options={'maxiter': 50, 'disp': False}
            )
            polished_pts = res.x.reshape(-1, 2)
            polished_score = _exact_score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
            
    # Final Polish with Powell (robust to non-differentiability)
    if time.time() < deadline - 1.0:
        try:
            res = minimize(
                _exact_objective, 
                best_pts.flatten(), 
                method='Powell',
                options={'maxiter': 20, 'disp': False}
            )
            polished_pts = res.x.reshape(-1, 2)
            polished_score = _exact_score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass

    # Normalize to unit area
    best_pts = _normalize(best_pts)
    
    return best_pts
