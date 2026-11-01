import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14

def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - p1
            for k in range(j + 1, n):
                v2 = points[k] - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area."""
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    return ma / ha

def _neg_score(params):
    """Negative score for minimization."""
    points = params.reshape(-1, 2)
    return -_score(points)

def _coordinate_descent(points, max_iters=50, step_size=0.05):
    """
    Perform coordinate descent to refine points.
    For each point, try moving it in 8 directions.
    """
    best_points = points.copy()
    best_score = _score(best_points)
    current_step = step_size
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            p = best_points[i].copy()
            best_local = p
            best_local_score = best_score
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_p = p + np.array([dx * current_step, dy * current_step])
                    test_points = best_points.copy()
                    test_points[i] = new_p
                    s = _score(test_points)
                    if s > best_local_score + 1e-9:
                        best_local = new_p
                        best_local_score = s
                        improved = True
            
            if improved:
                best_points[i] = best_local
                best_score = best_local_score
                current_step *= 0.95
            else:
                current_step *= 0.99
                
        if current_step < 1e-5:
            break

    return best_points

def construct():
    deadline = time.time() + 145  # Safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Optimized 2-Ring Configuration (7 outer, 7 inner)
    # 15 variables: 14 angles + 1 radius
    def _ring_score(params):
        angles = params[:14]
        r_inner = np.clip(params[14], 0.1, 0.9)
        
        points = np.zeros((14, 2))
        for i in range(7):
            points[i] = [np.cos(angles[i]), np.sin(angles[i])]
        for i in range(7):
            points[7 + i] = [r_inner * np.cos(angles[7 + i]), r_inner * np.sin(angles[7 + i])]
            
        return -_score(points)
        
    init_angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    init_params = np.append(init_angles, 0.5)
    
    if time.time() < deadline - 60:
        try:
            res = scipy.optimize.minimize(
                _ring_score, init_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            opt_params = res.x
            angles = opt_params[:14]
            r_inner = np.clip(opt_params[14], 0.1, 0.9)
            
            points = np.zeros((14, 2))
            for i in range(7):
                points[i] = [np.cos(angles[i]), np.sin(angles[i])]
            for i in range(7):
                points[7 + i] = [r_inner * np.cos(angles[7 + i]), r_inner * np.sin(angles[7 + i])]
                
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Strategy 2: Optimized 3-Ring Configuration (5, 5, 4)
    # 16 variables: 14 angles + 2 radii
    def _ring3_score(params):
        angles = params[:14]
        r1 = np.clip(params[14], 0.1, 0.9)
        r2 = np.clip(params[15], 0.1, 0.9)
        if r1 < r2:
            r1, r2 = r2, r1
            
        points = np.zeros((14, 2))
        for i in range(5):
            points[i] = [np.cos(angles[i]), np.sin(angles[i])]
        for i in range(5):
            points[5 + i] = [r1 * np.cos(angles[5 + i]), r1 * np.sin(angles[5 + i])]
        for i in range(4):
            points[10 + i] = [r2 * np.cos(angles[10 + i]), r2 * np.sin(angles[10 + i])]
            
        return -_score(points)
        
    if time.time() < deadline - 40 and best_points is not None:
        init_angles_3 = np.linspace(0, 2*np.pi, 14, endpoint=False)
        init_params_3 = np.append(init_angles_3, [0.6, 0.3])
        
        try:
            res = scipy.optimize.minimize(
                _ring3_score, init_params_3,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            opt_params = res.x
            angles = opt_params[:14]
            r1 = np.clip(opt_params[14], 0.1, 0.9)
            r2 = np.clip(opt_params[15], 0.1, 0.9)
            if r1 < r2:
                r1, r2 = r2, r1
                
            points = np.zeros((14, 2))
            for i in range(5):
                points[i] = [np.cos(angles[i]), np.sin(angles[i])]
            for i in range(5):
                points[5 + i] = [r1 * np.cos(angles[5 + i]), r1 * np.sin(angles[5 + i])]
            for i in range(4):
                points[10 + i] = [r2 * np.cos(angles[10 + i]), r2 * np.sin(angles[10 + i])]
                
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Strategy 3: General Coordinate Descent Polish
    if best_points is not None and time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        candidates = [best_points]
        
        for _ in range(3):
            perturb = rng.normal(0, 0.1, size=(14, 2))
            candidates.append(best_points + perturb)
            
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            ha = _hull_area(cand)
            if ha > 1e-10:
                cand = cand * (1.0 / np.sqrt(ha))
            else:
                continue
                
            polished = _coordinate_descent(cand, max_iters=30, step_size=0.1)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_points = polished.copy()

    if best_points is None:
        rng = np.random.RandomState(N)
        best_points = rng.random((N, 2))
        
    return best_points
