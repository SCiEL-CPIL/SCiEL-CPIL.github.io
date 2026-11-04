import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

def _min_triangle_area(points):
    """Compute minimum triangle area for N points (N<=14)."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            for k in range(j+1, n):
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute min triangle area / convex hull area."""
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _fourier_ring(params):
    """
    Generate 14 points as a distorted regular 14-gon.
    params: [a1, b1, a2, b2, a3, b3]
    where a_k, b_k are Fourier coefficients for radial perturbation.
    r(theta) = 1 + sum_k (a_k*cos(k*theta) + b_k*sin(k*theta))
    """
    a1, b1, a2, b2, a3, b3 = params
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Radial perturbation
    r = np.ones(N)
    r += a1 * np.cos(1*angles) + b1 * np.sin(1*angles)
    r += a2 * np.cos(2*angles) + b2 * np.sin(2*angles)
    r += a3 * np.cos(3*angles) + b3 * np.sin(3*angles)
    
    # Ensure r > 0
    r = np.maximum(r, 0.1)
    
    points = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    return points

def _optimize_fourier(time_budget):
    """Optimize the Fourier ring parameters."""
    def objective(params):
        pts = _fourier_ring(params)
        return _neg_score(pts)
    
    # Bounds: keep perturbations small to maintain convexity
    bounds = [(-0.15, 0.15), (-0.15, 0.15),
              (-0.1, 0.1), (-0.1, 0.1),
              (-0.05, 0.05), (-0.05, 0.05)]
    
    # Grid search for initial guess
    best_score = 0
    best_params = np.zeros(6)
    
    # Coarse grid
    ranges = [(-0.1, 0.1), (-0.1, 0.1), (-0.08, 0.08), (-0.08, 0.08), (-0.04, 0.04), (-0.04, 0.04)]
    steps = [5, 5, 4, 4, 3, 3]
    
    # Multi-dimensional grid search
    indices = [0] * 6
    while True:
        params = [ranges[i][0] + (ranges[i][1] - ranges[i][0]) * indices[i] / (steps[i] - 1) for i in range(6)]
        pts = _fourier_ring(params)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_params = np.array(params)
        
        # Increment indices
        for d in range(6):
            indices[d] += 1
            if indices[d] < steps[d]:
                break
            indices[d] = 0
        else:
            break
    
    # Local polish with Nelder-Mead
    try:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':500, 'maxfev':2000})
        if -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    pts = _fourier_ring(best_params)
    return best_score, pts

def _two_ring_config(n_out, r_in, phase_in):
    """Construct two-ring configuration."""
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    outer = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_out == N:
        return outer
    n_in = N - n_out
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phase_in
    inner = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
    return np.vstack((outer, inner))

def _optimize_two_ring(n_out, time_budget):
    """Optimize two-ring configuration."""
    n_in = N - n_out
    if n_in == 0:
        pts = _two_ring_config(N, 0, 0)
        return _score(pts), pts
    
    def objective(params):
        r_in, phase_in = params
        if r_in < 0.05 or r_in > 1.3:
            return 1e6
        pts = _two_ring_config(n_out, r_in, phase_in)
        return _neg_score(pts)
    
    # Grid search
    best_score = 0
    best_params = [0.5, 0.0]
    for r in np.linspace(0.1, 1.2, 15):
        for ph in np.linspace(0, 2*np.pi, 15, endpoint=False):
            s = -objective([r, ph])
            if s > best_score:
                best_score = s
                best_params = [r, ph]
    
    # Local polish
    try:
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':300})
        if -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
    
    pts = _two_ring_config(n_out, best_params[0], best_params[1])
    return best_score, pts

def _polish_full(points, time_budget):
    """Polish full 28D configuration."""
    def obj_flat(x):
        p = x.reshape(N, 2)
        return _neg_score(p)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(obj_flat, points.flatten(), method='Nelder-Mead',
                       options={'xatol':1e-8, 'fatol':1e-8, 'maxiter':500, 'maxfev':3000})
        pts = res.x.reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_score = 0
    best_points = None
    
    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # 1. Try Fourier ring optimization (low-dimensional, smooth)
    if time.time() < deadline - 30:
        score, pts = _optimize_fourier(deadline - time.time() - 30)
        try_update(score, pts)
    
    # 2. Try two-ring configurations
    splits = [(14,0), (12,2), (10,4), (8,6), (7,7), (11,3), (9,5), (13,1)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 15:
            break
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        score, pts = _optimize_two_ring(n_out, min(10.0, remaining / 2.0))
        try_update(score, pts)
    
    # 3. Final polish on best found
    if best_points is not None and time.time() < deadline - 5:
        s, pts = _polish_full(best_points, deadline - time.time() - 5)
        try_update(s, pts)
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
