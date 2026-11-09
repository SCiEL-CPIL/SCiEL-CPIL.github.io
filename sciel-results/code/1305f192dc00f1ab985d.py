import numpy as np
import time
import itertools
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

    # Vectorized triangle area computation
    # Generate all C(N,3) index triples
    idx = np.array(list(itertools.combinations(range(N), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    min_area = np.min(areas)
    
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
    x_best = np.concatenate((angles[1:], radii))
    
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
            x_start = x0.copy()
        else:
            # Alternate between perturbing best solution and regular polygon
            if n_restarts % 2 == 0:
                # Perturb the best found parameters
                noise_scale = 0.05 + 0.15 * np.random.rand()
                x_start = x_best + np.random.normal(0, noise_scale, 2*N - 1)
            else:
                # Perturb the regular polygon with varying noise
                noise_scale = 0.1 + 0.4 * np.random.rand()
                x_start = x0 + np.random.normal(0, noise_scale, 2*N - 1)
        
        # Try Nelder-Mead first
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-10, 'adaptive': True})
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                theta = np.concatenate(([0.0], res.x[:N-1]))
                r = np.maximum(res.x[N-1:], 1e-6)
                best_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
                x_best = res.x.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B from the best result (or current x_start if NM failed)
        if time.time() < deadline - 10:
            try:
                x_start_lbfgs = x_best.copy()
                # Small perturbation for L-BFGS-B
                x_start_lbfgs += np.random.normal(0, 0.02, 2*N - 1)
                res2 = minimize(objective, x_start_lbfgs, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                current_score2 = -res2.fun
                if current_score2 > best_score:
                    best_score = current_score2
                    theta = np.concatenate(([0.0], res2.x[:N-1]))
                    r = np.maximum(res2.x[N-1:], 1e-6)
                    best_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
                    x_best = res2.x.copy()
            except Exception:
                pass

    return best_points
