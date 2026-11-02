"""Improved program for Heilbronn convex regions task."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _triangle_areas(pts):
    """
    Compute areas of all triangles formed by triples of points.
    pts: (N, 2) array.
    Returns: array of shape (M,), where M = N*(N-1)*(N-2)/6.
    """
    n = len(pts)
    # Indices for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    pi = pts[i_flat]
    pj = pts[j_flat]
    pk = pts[k_flat]
    
    # Cross product in 2D: (pj - pi) x (pk - pi)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    if len(pts) < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_exact_score(pts_flat):
    """Objective function for minimization: negative of the score."""
    pts = pts_flat.reshape(-1, 2)
    score = _score_points(pts)
    return -score

def _generate_initializations():
    """Generate a diverse set of initial point configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Perturbed Regular 13-gon
    for _ in range(5):
        noise = np.random.normal(0, 0.1, (N, 2))
        inits.append(np.column_stack([np.cos(angles), np.sin(angles)]) + noise)
        
    # 3. Two concentric rings (7 outer, 6 inner)
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = 0.6 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    inits.append(np.vstack([outer_pts, inner_pts]))
    
    # 4. Three rings (5, 4, 4)
    r1_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r2_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    r3_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    r1_pts = np.column_stack([np.cos(r1_angles), np.sin(r1_angles)])
    r2_pts = 0.6 * np.column_stack([np.cos(r2_angles), np.sin(r2_angles)])
    r3_pts = 0.3 * np.column_stack([np.cos(r3_angles), np.sin(r3_angles)])
    inits.append(np.vstack([r1_pts, r2_pts, r3_pts]))
    
    # 5. Random points in a circle
    for _ in range(10):
        radii = np.sqrt(np.random.rand(N))
        angles = 2 * np.pi * np.random.rand(N)
        inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
        
    # 6. Random points in a square
    for _ in range(5):
        inits.append(np.random.rand(N, 2) - 0.5)

    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_pts = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_score
        try:
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass

    inits = _generate_initializations()
    
    # Phase 1: Multi-start Nelder-Mead
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        update_best(init_pts)
        
        x0 = init_pts.flatten()
        
        try:
            # Nelder-Mead is good for non-smooth problems
            result = minimize(
                _neg_exact_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < 0:
                opt_pts = result.x.reshape(-1, 2)
                update_best(opt_pts)
                
        except Exception:
            pass

    # Phase 2: Refinement of the best candidate
    if best_pts is not None and time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _neg_exact_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < 0:
                refined_pts = result.x.reshape(-1, 2)
                update_best(refined_pts)
        except Exception:
            pass

    if best_pts is not None:
        return best_pts
    else:
        return np.random.rand(N, 2)
