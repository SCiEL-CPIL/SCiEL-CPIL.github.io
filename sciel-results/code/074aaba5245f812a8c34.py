import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def get_score(points):
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

    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = np.ones(N)
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    best_score = get_score(points)
    best_points = points.copy()
    
    # Objective function for joint optimization of angles and radii
    def objective(params):
        # params: [theta_1..theta_12, r_1..r_13]
        theta = np.concatenate(([0.0], params[:N-1]))
        r = params[N-1:]
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        score = get_score(pts)
        return -score

    # Initial parameters
    x0 = np.concatenate((angles[1:], radii))
    
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        if n_restarts == 1:
            x_start = x0
        else:
            # Perturb best solution
            # Extract angles and radii from best_points
            # Note: best_points might not be in angular order if radii changed significantly?
            # No, we maintain the order in the parameter vector.
            # We need to recover angles and radii from best_points to perturb.
            # However, best_points is just (x,y).
            # We can recover r and theta.
            # But if the optimizer changed the "order" implicitly? No, indices are fixed.
            
            # To be safe, we just perturb the last successful x_start or the regular polygon.
            # Better: Use the current best_points to get r, theta.
            r_best = np.linalg.norm(best_points, axis=1)
            theta_best = np.arctan2(best_points[:, 1], best_points[:, 0])
            # Sort angles to match the parameter order (0, ..., 2pi)
            # The parameter order is fixed by index 0..N-1.
            # We must ensure the recovered theta matches the index order.
            # Since we constructed points from sorted angles, the index order corresponds to angular order.
            # arctan2 returns -pi to pi. We need to unwrap or sort.
            # Actually, we can just use the x_start from the previous iteration if it was good.
            # Or simply perturb the regular polygon again with larger noise.
            
            # Let's try perturbing the regular polygon with varying noise
            noise_scale = 0.1 + 0.5 * np.random.rand()
            x_start = x0 + np.random.normal(0, noise_scale, N + N - 1)
            # Also try perturbing the best found parameters if we stored them
            # We didn't store x_best. Let's store it.
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-9, 'adaptive': True})
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                # Reconstruct points
                theta = np.concatenate(([0.0], res.x[:N-1]))
                r = np.maximum(res.x[N-1:], 1e-6)
                best_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
                # Store best x for future perturbations
                x_best = res.x
        except Exception:
            pass

    return best_points
