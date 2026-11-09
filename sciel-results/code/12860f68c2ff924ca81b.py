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
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi / 7
    
    for r_inner in [0.3, 0.5, 0.7, 0.9]:
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        pts_ring = np.vstack([pts_outer, pts_inner])
        guesses.append(pts_ring.flatten())
        
    # 3. Random points in unit disk
    for _ in range(20):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        guesses.append(pts.flatten())
        
    # 4. Regular polygon with slight radial noise
    for _ in range(5):
        r = 1.0 + rng.normal(0, 0.1, N)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        guesses.append(pts.flatten())

    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
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
            
    # Sort guesses by initial score
    scored_guesses = []
    for g in guesses:
        pts = g.reshape(N, 2)
        s = _score(pts)
        scored_guesses.append((s, g))
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Strategy: Run Nelder-Mead on top candidates, then Powell polish.
    # Nelder-Mead is often better for non-smooth objectives in low dimensions.
    
    num_candidates = min(8, len(scored_guesses))
    
    for i in range(num_candidates):
        if time.time() > deadline - 20:
            break
            
        s_init, x_init = scored_guesses[i]
        
        # 1. Nelder-Mead Optimization
        try:
            res_nm = minimize(
                _objective,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'disp': False
                }
            )
            
            pts_nm = res_nm.x.reshape(N, 2)
            s_nm = _score(pts_nm)
            
            if s_nm > best_score:
                best_score = s_nm
                best_points = pts_nm.copy()
                
        except Exception:
            pass
        
        # 2. Powell Polish on the best of (initial, NM)
        current_best_x = x_init
        current_best_s = s_init
        if 'res_nm' in locals() and _score(res_nm.x.reshape(N, 2)) > current_best_s:
             current_best_x = res_nm.x
             current_best_s = _score(current_best_x.reshape(N, 2))
             
        if time.time() > deadline - 10:
            continue
            
        try:
            res_pw = minimize(
                _objective,
                current_best_x,
                method='Powell',
                options={
                    'maxiter': 200,
                    'maxfev': 2000,
                    'xrtol': 1e-10,
                    'ftol': 1e-12,
                    'disp': False
                }
            )
            
            pts_pw = res_pw.x.reshape(N, 2)
            s_pw = _score(pts_pw)
            
            if s_pw > best_score:
                best_score = s_pw
                best_points = pts_pw.copy()
                
        except Exception:
            pass

    # Final Polish if time remains
    if best_points is not None and time.time() < deadline - 5:
        x_best = best_points.flatten()
        try:
            res_final = minimize(
                _objective,
                x_best,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'disp': False
                }
            )
            pts_final = res_final.x.reshape(N, 2)
            s_final = _score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_points = pts_final.copy()
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
