import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# Pre-compute indices for all triples
_IDX_I = []
_IDX_J = []
_IDX_K = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _IDX_I.append(i)
            _IDX_J.append(j)
            _IDX_K.append(k)
_IDX_I = np.array(_IDX_I)
_IDX_J = np.array(_IDX_J)
_IDX_K = np.array(_IDX_K)

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
            
        if hull_area < 1e-10:
            return 0.0
        
        # Compute triangle areas
        pi = points[_IDX_I]
        pj = points[_IDX_J]
        pk = points[_IDX_K]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(points_flat):
    """
    Objective for minimization: -score
    points_flat: (28,) array
    """
    points = points_flat.reshape(N, 2)
    s = _score(points)
    if s <= 1e-15:
        return 1e6
    return -s

def _generate_initial_guesses():
    """
    Generate diverse initial guesses for the 14 points.
    """
    guesses = []
    rng = np.random.default_rng(42)
    
    # 1. Regular polygon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    guesses.append(pts_reg.flatten())
    
    # 2. Concentric rings: 7 points on outer, 7 on inner
    # Outer ring radius 1, Inner ring radius 0.5
    # Rotate inner ring by pi/7 to interleave
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi / 7
    
    pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    pts_inner = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    
    # Interleave or just stack? Stacking is fine, optimizer will sort/move them.
    # Actually, for symmetry, interleaving might help convergence.
    pts_ring1 = np.vstack([pts_outer, pts_inner])
    guesses.append(pts_ring1.flatten())
    
    # 3. Concentric rings: 6 points outer, 6 points inner, 2 points center? 
    # Or 5,5,4?
    # Let's try 6 outer, 8 inner? No, 14 points.
    # Try 6 outer (r=1), 6 inner (r=0.6), 2 center-ish?
    # Better: 7 outer, 7 inner with different radii
    for r_inner in [0.4, 0.6, 0.8]:
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        pts_ring = np.vstack([pts_outer, pts_inner])
        guesses.append(pts_ring.flatten())
        
    # 4. Random points in unit disk
    for _ in range(10):
        # Random polar coordinates
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        guesses.append(pts.flatten())
        
    # 5. Regular polygon scaled down (to encourage hull expansion?)
    # Actually, the score normalizes by hull area, so scaling doesn't change score.
    # But it changes the relative positions if not all points are on hull.
    # If we start with all points on hull, the optimizer has to move them in.
    # Starting with some points inside (like rings) gives a better starting point for "moving in".
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    guesses = _generate_initial_guesses()
    
    # Evaluate initial guesses
    for g in guesses:
        pts = g.reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # Optimize from best initial guesses
    # Sort guesses by initial score to prioritize good starts
    scored_guesses = []
    for g in guesses:
        pts = g.reshape(N, 2)
        s = _score(pts)
        scored_guesses.append((s, g))
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Run Powell optimization on top candidates
    # Powell is derivative-free and good for non-smooth functions.
    # We limit iterations to ensure we finish in time.
    
    num_candidates = min(5, len(scored_guesses))
    
    for i in range(num_candidates):
        if time.time() > deadline - 10:
            break
            
        s_init, x_init = scored_guesses[i]
        
        try:
            res = minimize(
                _objective,
                x_init,
                method='Powell',
                options={
                    'maxiter': 200,  # Limit iterations
                    'maxfev': 2000,  # Limit function evaluations
                    'xrtol': 1e-10,
                    'ftol': 1e-12,
                    'disp': False
                }
            )
            
            pts_opt = res.x.reshape(N, 2)
            s_opt = _score(pts_opt)
            
            if s_opt > best_score:
                best_score = s_opt
                best_points = pts_opt.copy()
                
        except Exception as e:
            pass

    # If time remains, do a second pass on the best result found
    if best_points is not None and time.time() < deadline - 5:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _objective,
                x_best,
                method='Powell',
                options={
                    'maxiter': 300,
                    'maxfev': 3000,
                    'xrtol': 1e-12,
                    'ftol': 1e-14,
                    'disp': False
                }
            )
            pts_opt = res.x.reshape(N, 2)
            s_opt = _score(pts_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_points = pts_opt.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
