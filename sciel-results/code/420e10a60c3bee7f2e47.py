"""Heilbronn convex regions: 13 points, maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _min_triangle_area(pts):
    """Minimum area over all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                if a < min_area:
                    min_area = a
    return min_area

def _convex_hull_area(pts):
    """Area of convex hull using Andrew's monotone chain."""
    pts = np.array(pts, dtype=float)
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    n = len(pts)
    if n < 3:
        return 0.0
    lower = []
    for p in pts:
        while len(lower) >= 2:
            o, a = lower[-2], lower[-1]
            cross = (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2:
            o, a = upper[-2], upper[-1]
            cross = (a[0]-o[0])*(p[1]-o[1]) - (a[1]-o[1])*(p[0]-o[0])
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:,0]
    y = hull[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """Min triangle area / convex hull area."""
    h = _convex_hull_area(pts)
    if h < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / h

def _params_to_points(params):
    """
    params: [r1, r2, r3, phase1, phase2, phase3]
    Shell 1: 5 points at radius r1, phase phase1
    Shell 2: 5 points at radius r2, phase phase2
    Shell 3: 3 points at radius r3, phase phase3
    Center: 1 point at origin
    """
    r1, r2, r3 = params[0], params[1], params[2]
    p1, p2, p3 = params[3], params[4], params[5]
    pts = []
    # Shell 1: 5 points
    for i in range(5):
        a = p1 + 2*np.pi*i/5
        pts.append([r1*np.cos(a), r1*np.sin(a)])
    # Shell 2: 5 points
    for i in range(5):
        a = p2 + 2*np.pi*i/5
        pts.append([r2*np.cos(a), r2*np.sin(a)])
    # Shell 3: 3 points
    for i in range(3):
        a = p3 + 2*np.pi*i/3
        pts.append([r3*np.cos(a), r3*np.sin(a)])
    # Center
    pts.append([0.0, 0.0])
    return np.array(pts)

def _objective(params):
    pts = _params_to_points(params)
    return -_score(pts)

def construct():
    deadline = time.time() + 110  # safety margin before 150s kill

    # Initial guess: concentric shells with some spacing
    # Shell 1 (inner): 5 pts, Shell 2 (mid): 5 pts, Shell 3 (outer): 3 pts
    x0 = np.array([0.35, 0.7, 1.0, 0.0, np.pi/5, np.pi/3])

    best_score = 0.0
    best_params = x0.copy()

    # Nelder-Mead optimization
    options = {
        'maxiter': 5000,
        'maxfev': 5000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }

    try:
        result = minimize(_objective, x0, method='Nelder-Mead', options=options)
        if time.time() > deadline:
            # Return best found so far
            pass
        else:
            s = -result.fun
            if s > best_score:
                best_score = s
                best_params = result.x
    except Exception:
        pass

    # Try multiple restarts with different initial conditions if time permits
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            # Random perturbation of initial guess
            x_try = x0 + 0.2 * rng.randn(6)
            # Ensure radii are positive and ordered
            x_try[0] = abs(x_try[0]) + 0.1
            x_try[1] = abs(x_try[1]) + 0.3
            x_try[2] = abs(x_try[2]) + 0.5
            # Sort radii
            r_order = np.argsort(x_try[:3])
            x_try[:3] = x_try[r_order]
            # Ensure r1 < r2 < r3
            x_try[0] = max(0.1, x_try[0])
            x_try[1] = max(x_try[0] + 0.1, x_try[1])
            x_try[2] = max(x_try[1] + 0.1, x_try[2])

            try:
                res = minimize(_objective, x_try, method='Nelder-Mead',
                               options={'maxiter': 2000, 'maxfev': 2000,
                                        'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = res.x
            except Exception:
                continue

    pts = _params_to_points(best_params)
    return pts
# EVOLVE-BLOCK-END
