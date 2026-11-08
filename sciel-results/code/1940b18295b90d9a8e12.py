import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _generate_two_shell(k, m, r1, r2, phi1, phi2):
    """Generate points for a two-shell configuration."""
    pts = []
    # Inner shell
    if k == 1:
        pts.append([0.0, 0.0])
    elif k == 2:
        pts.append([r1 * np.cos(phi1), r1 * np.sin(phi1)])
        pts.append([r1 * np.cos(phi1 + np.pi), r1 * np.sin(phi1 + np.pi)])
    else:
        angles = np.linspace(0, 2*np.pi, k, endpoint=False) + phi1
        inner = np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles)))
        pts.extend(inner.tolist())
    
    # Outer shell
    angles = np.linspace(0, 2*np.pi, m, endpoint=False) + phi2
    outer = np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles)))
    pts.extend(outer.tolist())
    
    return np.array(pts)

def _generate_three_shell(k1, k2, k3, r1, r2, r3, phi1, phi2, phi3):
    """Generate points for a three-shell configuration."""
    pts = []
    # Shell 1
    if k1 == 1:
        pts.append([0.0, 0.0])
    elif k1 == 2:
        pts.append([r1 * np.cos(phi1), r1 * np.sin(phi1)])
        pts.append([r1 * np.cos(phi1 + np.pi), r1 * np.sin(phi1 + np.pi)])
    else:
        angles = np.linspace(0, 2*np.pi, k1, endpoint=False) + phi1
        inner = np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles)))
        pts.extend(inner.tolist())
        
    # Shell 2
    if k2 == 1:
        pts.append([r2 * np.cos(phi2), r2 * np.sin(phi2)]) # If k2=1, it's just 1 point on shell 2
        # Wait, if k2=1, it's a single point.
    elif k2 == 2:
        pts.append([r2 * np.cos(phi2), r2 * np.sin(phi2)])
        pts.append([r2 * np.cos(phi2 + np.pi), r2 * np.sin(phi2 + np.pi)])
    else:
        angles = np.linspace(0, 2*np.pi, k2, endpoint=False) + phi2
        mid = np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles)))
        pts.extend(mid.tolist())
        
    # Shell 3
    angles = np.linspace(0, 2*np.pi, k3, endpoint=False) + phi3
    outer = np.column_stack((r3 * np.cos(angles), r3 * np.sin(angles)))
    pts.extend(outer.tolist())
    
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    s = _score(pts)
    if s > best_score:
        best_score = s
        best_points = pts.copy()

    # 2. Two Shell Configurations
    partitions_2 = [
        (1, 12),
        (2, 11),
        (3, 10),
        (4, 9),
        (5, 8),
        (6, 7)
    ]
    
    for k, m in partitions_2:
        if time.time() > deadline - 15: break
        
        def objective_2(params):
            r1, r2, phi1, phi2 = params
            if r1 < 0: r1 = 0.0
            if r2 < 0: r2 = 0.0
            pts = _generate_two_shell(k, m, r1, r2, phi1, phi2)
            return -_score(pts)
        
        # Try a few initial guesses
        inits = [
            [0.3, 1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, np.pi/m],
            [0.7, 1.0, 0.0, 0.0],
            [0.1, 1.0, 0.0, 0.0]
        ]
        
        for x0 in inits:
            if time.time() > deadline - 10: break
            try:
                res = minimize(objective_2, x0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if time.time() < deadline - 5:
                    r1, r2, phi1, phi2 = res.x
                    pts = _generate_two_shell(k, m, r1, r2, phi1, phi2)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
            except Exception:
                pass

    # 3. Three Shell Configurations (1 + 5 + 7)
    if time.time() < deadline - 20:
        k1, k2, k3 = 1, 5, 7
        def objective_3(params):
            r1, r2, r3, phi1, phi2, phi3 = params
            if r1 < 0: r1 = 0.0
            if r2 < 0: r2 = 0.0
            if r3 < 0: r3 = 0.0
            pts = _generate_three_shell(k1, k2, k3, r1, r2, r3, phi1, phi2, phi3)
            return -_score(pts)
        
        x0 = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
        try:
            res = minimize(objective_3, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() < deadline - 5:
                r1, r2, r3, phi1, phi2, phi3 = res.x
                pts = _generate_three_shell(k1, k2, k3, r1, r2, r3, phi1, phi2, phi3)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # 4. Full 2D Optimization from Best Symmetric Solution
    if time.time() < deadline - 10 and best_points is not None:
        x0 = best_points.flatten()
        
        def objective_full(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        try:
            res = minimize(objective_full, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if time.time() < deadline - 2:
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    return best_points
