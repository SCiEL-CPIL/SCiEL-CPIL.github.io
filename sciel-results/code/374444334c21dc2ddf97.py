import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized computation of all triangle areas
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    # Filter for i < j < k
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """
    Compute the soft minimum of an array of areas.
    Uses log-sum-exp trick for stability.
    soft_min(x) = -1/tau * log(sum(exp(-tau * x)))
    """
    min_val = np.min(areas)
    if np.allclose(areas, min_val):
        return min_val
        
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
    
    # 1. Baseline: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    s = get_score(points)
    best_score = s
    best_points = points.copy()
    
    # 2. Baseline: Regular 12-gon + center
    # This topology often yields better ratios for small N
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points_12 = np.column_stack((np.cos(angles_12), np.sin(angles_12)))
    center = np.array([[0.0, 0.0]])
    points_12c = np.vstack([points_12, center])
    s_12c = get_score(points_12c)
    if s_12c > best_score:
        best_score = s_12c
        best_points = points_12c.copy()
    
    # Define the smooth objective function
    def neg_score_smooth(points, tau):
        """
        Minimize - (soft_min(areas) / hull_area)
        """
        try:
            n = len(points)
            idx = np.arange(n)
            i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
            mask = (i < j) & (j < k)
            i = i[mask]
            j = j[mask]
            k = k[mask]
            
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            # Soft min
            sm = soft_min(areas, tau)
            
            # Hull area
            try:
                hull = ConvexHull(points)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
                
            score = sm / hull_area
            return -score
        except:
            return 1e6

    # 3. Optimize angles on unit circle (13-gon topology)
    def neg_score_circle_angles(theta, tau):
        angles = np.concatenate(([0.0], theta))
        angles = np.sort(angles % (2 * np.pi))
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        return neg_score_smooth(pts, tau)

    x0_circle = np.array([2 * np.pi * i / N for i in range(1, N)])
    
    # Try a range of tau values
    taus = [50, 100, 200]
    
    for tau in taus:
        if time_left() < 10.0:
            break
        # Multiple restarts for circle angles
        n_restarts = 3
        for r in range(n_restarts):
            if time_left() < 5.0:
                break
            if r == 0:
                x_start = x0_circle.copy()
            else:
                step = 0.2
                x_start = x0_circle + np.random.normal(0, step, N - 1)
            
            try:
                res = minimize(neg_score_circle_angles, x_start, args=(tau,), method='Powell',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-9})
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    angles = np.concatenate(([0.0], res.x))
                    angles = np.sort(angles % (2 * np.pi))
                    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
            except:
                pass

    # 4. Optimize 12-gon + center topology
    # Parameters: 12 angles for the ring, and (x, y) for the center point
    # Fix first angle at 0 for rotational invariance
    def neg_score_12c(params, tau):
        # params: [x_c, y_c, theta_1, ..., theta_11] (12 angles, first fixed at 0)
        x_c = params[0]
        y_c = params[1]
        theta = np.concatenate(([0.0], params[2:]))
        theta = np.sort(theta % (2 * np.pi))
        ring_pts = np.column_stack((np.cos(theta), np.sin(theta)))
        center_pt = np.array([[x_c, y_c]])
        pts = np.vstack([ring_pts, center_pt])
        return neg_score_smooth(pts, tau)

    # Initial guess: regular 12-gon + center
    x0_12c = np.concatenate(([0.0, 0.0], np.array([2 * np.pi * i / 12 for i in range(1, 12)])))
    
    for tau in [50, 100]:
        if time_left() < 10.0:
            break
        n_restarts = 2
        for r in range(n_restarts):
            if time_left() < 5.0:
                break
            if r == 0:
                x_start = x0_12c.copy()
            else:
                x_start = x0_12c + np.random.normal(0, 0.1, len(x0_12c))
            
            try:
                res = minimize(neg_score_12c, x_start, args=(tau,), method='Powell',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-9})
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    x_c = res.x[0]
                    y_c = res.x[1]
                    theta = np.concatenate(([0.0], res.x[2:]))
                    theta = np.sort(theta % (2 * np.pi))
                    ring_pts = np.column_stack((np.cos(theta), np.sin(theta)))
                    center_pt = np.array([[x_c, y_c]])
                    best_points = np.vstack([ring_pts, center_pt])
            except:
                pass

    # 5. Full 2D Optimization
    # Use the best points found so far as initial guess
    if time_left() > 20.0 and best_points is not None:
        
        def neg_score_2d(pts_flat, tau):
            pts = pts_flat.reshape(N, 2)
            return neg_score_smooth(pts, tau)
        
        x0_2d = best_points.flatten()
        
        # Use higher tau for 2D to push towards true min
        taus_2d = [100, 200]
        
        for tau in taus_2d:
            if time_left() < 10.0:
                break
            # A few restarts with small perturbations
            n_restarts = 3
            for r in range(n_restarts):
                if time_left() < 5.0:
                    break
                if r == 0:
                    x_start = x0_2d.copy()
                else:
                    x_start = x0_2d + np.random.normal(0, 0.05, 2*N)
                
                try:
                    res2d = minimize(neg_score_2d, x_start, args=(tau,), method='Powell',
                                     options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-8})
                    current_score = -res2d.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = res2d.x.reshape(N, 2)
                        x0_2d = res2d.x
                except:
                    pass

    # Final check with exact min (not soft min) to ensure we report the true score
    if best_points is not None:
        final_score = get_score(best_points)
        if final_score > best_score:
            best_score = final_score
        
    return best_points
