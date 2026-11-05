import numpy as np
from scipy.optimize import minimize
import time

def score(points):
    # points: np.array of shape (8, 2)
    # 1. Center
    p = np.array(points)
    p -= np.mean(p, axis=0)
    
    # 2. Check for degeneracy
    if np.allclose(p, 0):
        return -np.inf
        
    # 3. Scale to diameter 2
    # Compute all pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    D = np.max(dists)
    if D < 1e-10:
        return -np.inf
    p = p * (2.0 / D)
    
    # Recompute distances after scaling
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return -np.inf
        
    # Product of ordered pairs
    # Sum of logs
    log_prod = np.sum(np.log(valid_dists))
    
    # Score = exp(log_prod) / 8^8
    # We want to maximize Score, so we can just return log_prod - 8*log(8)
    # But for optimization, returning the value is fine.
    # To avoid overflow/underflow issues in intermediate steps, return log score?
    # Minimize negative log score.
    return -(log_prod - 8 * np.log(8))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    # Strategy: Try symmetric configurations
    # 1. Regular octagon (baseline)
    # 2. Two concentric squares with 4-fold symmetry
    # 3. Two concentric rectangles
    # 4. Free optimization from good starting points
    
    best_points = None
    best_score = np.inf  # We minimize negative log score
    
    def evaluate(points):
        nonlocal best_points, best_score
        s = score(points)
        if s < best_score:
            best_score = s
            best_points = points.copy()
        return s
    
    # 1. Regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    evaluate(p_oct)
    
    # 2. Two concentric squares with 4-fold rotational symmetry
    # Points: (r1*cos(k*pi/4 + phi), r1*sin(k*pi/4 + phi)) for k=0..3
    #         (r2*cos(k*pi/4 + phi + pi/4), r2*sin(k*pi/4 + phi + pi/4)) for k=0..3
    # Parameters: r1, r2, phi
    def two_squares(params):
        r1, r2, phi = params
        if r1 < 0.01 or r2 < 0.01:
            return 1e10
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + phi
        a2 = a1 + np.pi/4
        pts1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        pts2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        return np.vstack([pts1, pts2])
    
    # Optimize two squares
    x0 = [1.0, 0.7, 0.0]
    try:
        res = minimize(lambda p: score(two_squares(p)), x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_points = two_squares(res.x).copy()
    except Exception:
        pass
    
    # Try different initializations for two squares
    for r1_init in [0.8, 1.0, 1.2]:
        for r2_init in [0.3, 0.5, 0.7, 0.9]:
            for phi_init in [0.0, np.pi/8, np.pi/4]:
                if time.time() > deadline:
                    break
                x0 = [r1_init, r2_init, phi_init]
                try:
                    res = minimize(lambda p: score(two_squares(p)), x0, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_points = two_squares(res.x).copy()
                except Exception:
                    pass
    
    # 3. Free optimization from the best symmetric configuration
    if best_points is not None and time.time() < deadline:
        try:
            res = minimize(score, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(8, 2).copy()
        except Exception:
            pass
    
    # 4. Try more diverse symmetric configurations
    # Two concentric rectangles (not squares)
    def two_rects(params):
        r1, r2, phi = params
        if r1 < 0.01 or r2 < 0.01:
            return 1e10
        # Rectangle 1: corners at (±a1, ±b1) rotated by phi
        a1 = r1 * 1.0  # semi-axis
        b1 = r1 * 0.6  # semi-axis
        # Rectangle 2: corners at (±a2, ±b2) rotated by phi + pi/4
        a2 = r2 * 1.0
        b2 = r2 * 0.6
        
        pts = []
        for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            x = dx * a1
            y = dy * b1
            xr = x * np.cos(phi) - y * np.sin(phi)
            yr = x * np.sin(phi) + y * np.cos(phi)
            pts.append([xr, yr])
        for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            x = dx * a2
            y = dy * b2
            xr = x * np.cos(phi + np.pi/4) - y * np.sin(phi + np.pi/4)
            yr = x * np.sin(phi + np.pi/4) + y * np.cos(phi + np.pi/4)
            pts.append([xr, yr])
        return np.array(pts)
    
    for r1_init in [0.8, 1.0, 1.2]:
        for r2_init in [0.3, 0.5, 0.7]:
            for phi_init in [0.0, np.pi/8, np.pi/6]:
                if time.time() > deadline:
                    break
                x0 = [r1_init, r2_init, phi_init]
                try:
                    res = minimize(lambda p: score(two_rects(p)), x0, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_points = two_rects(res.x).copy()
                except Exception:
                    pass
    
    # 5. Final free optimization from best found
    if best_points is not None and time.time() < deadline - 10:
        try:
            res = minimize(score, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(8, 2).copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return [[float(x), float(y)] for x, y in best_points]
