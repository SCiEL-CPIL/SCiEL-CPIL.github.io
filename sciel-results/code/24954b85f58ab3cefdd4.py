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

def construct():
    deadline = time.time() + 130  # safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Points on concentric circles with optimized angles
    # Try different ring configurations
    ring_configs = [
        # (num_rings, radii, angles_per_ring)
        (1, [1.0], [14]),  # All on one circle
        (2, [1.0, 0.5], [7, 7]),
        (2, [1.0, 0.6], [7, 7]),
        (2, [1.0, 0.4], [8, 6]),
        (2, [1.0, 0.55], [8, 6]),
        (3, [1.0, 0.65, 0.35], [5, 5, 4]),
        (3, [1.0, 0.6, 0.3], [5, 5, 4]),
        (2, [1.0, 0.5], [10, 4]),
        (2, [1.0, 0.45], [9, 5]),
    ]
    
    for n_rings, radii, counts in ring_configs:
        if time.time() > deadline - 30:
            break
        if sum(counts) != N:
            continue
        
        # Create initial points on concentric circles
        points = []
        for r, c in zip(radii, counts):
            if c > 0:
                # Offset angles to avoid alignment
                base_angle = np.random.uniform(0, 2 * np.pi) if np.random.random() > 0.5 else 0
                angles = np.linspace(0, 2 * np.pi, c, endpoint=False) + base_angle
                for a in angles:
                    points.append([r * np.cos(a), r * np.sin(a)])
        
        points = np.array(points)
        
        # Normalize so hull area is 1
        ha = _hull_area(points)
        if ha > 1e-10:
            points = points * (1.0 / np.sqrt(ha))
        
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        # Local optimization
        if time.time() > deadline - 20:
            continue
        
        try:
            result = scipy.optimize.minimize(
                _neg_score, points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            opt_points = result.x.reshape(-1, 2)
            ha = _hull_area(opt_points)
            if ha > 1e-10:
                opt_points = opt_points * (1.0 / np.sqrt(ha))
            s_opt = _score(opt_points)
            if s_opt > best_score:
                best_score = s_opt
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Regular polygon vertices + optimized interior points
    if time.time() < deadline - 25:
        # 7 vertices on unit circle, 7 interior
        for n_outer in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
            if time.time() > deadline - 20:
                break
            n_inner = N - n_outer
            if n_inner < 0:
                continue
            
            # Outer points: regular polygon
            outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
            outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
            
            # Inner points: start at origin or small circle
            if n_inner > 0:
                inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
                inner = 0.3 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
                points = np.vstack([outer, inner])
            else:
                points = outer
            
            ha = _hull_area(points)
            if ha > 1e-10:
                points = points * (1.0 / np.sqrt(ha))
            
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
            
            if time.time() > deadline - 15:
                continue
            
            try:
                result = scipy.optimize.minimize(
                    _neg_score, points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                opt_points = result.x.reshape(-1, 2)
                ha = _hull_area(opt_points)
                if ha > 1e-10:
                    opt_points = opt_points * (1.0 / np.sqrt(ha))
                s_opt = _score(opt_points)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = opt_points.copy()
            except Exception:
                pass
    
    # Final refinement: multi-start Nelder-Mead from best found
    if best_points is not None and time.time() < deadline - 10:
        try:
            result = scipy.optimize.minimize(
                _neg_score, best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            opt_points = result.x.reshape(-1, 2)
            ha = _hull_area(opt_points)
            if ha > 1e-10:
                opt_points = opt_points * (1.0 / np.sqrt(ha))
            s_opt = _score(opt_points)
            if s_opt > best_score:
                best_score = s_opt
                best_points = opt_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: random
        rng = np.random.RandomState(N)
        best_points = rng.random((N, 2))
    
    return best_points
