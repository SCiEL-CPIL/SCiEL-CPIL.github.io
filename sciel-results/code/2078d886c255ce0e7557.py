import itertools
import numpy as np
import scipy.optimize
import time

N = 14
D = 3

def _true_score(points):
    # points shape (N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    d_min = np.min(dist_sq)
    d_max = np.max(dist_sq)
    if d_min < 1e-12:
        return 1e12
    return d_max / d_min

def _soft_max(v, beta):
    v_max = np.max(v)
    return (v_max + np.log(np.sum(np.exp(beta * (v - v_max)))) / beta)

def _soft_min(v, beta):
    return -_soft_max(-v, beta)

def _objective_soft(x):
    # x is shape (N-1, D)
    pts = np.zeros((N, D))
    pts[1:] = x
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    
    beta = 100.0
    d_max = _soft_max(dist_sq, beta)
    d_min = _soft_min(dist_sq, beta)
    
    if d_min < 1e-10:
        return 1e12
        
    return d_max / d_min

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_true_score = np.inf
    best_points = None
    
    inits = []
    for i in range(15):
        pts = np.random.randn(N, D)
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        # Translate so point 0 is at origin
        init_pts = pts - pts[0]
        inits.append(init_pts[1:].flatten())

    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            res = scipy.optimize.minimize(
                _objective_soft, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Evaluate true score
            candidate_pts = np.zeros((N, D))
            candidate_pts[1:] = res.x.reshape(N-1, D)
            
            true_s = _true_score(candidate_pts)
            
            if true_s < best_true_score:
                best_true_score = true_s
                best_points = candidate_pts
                
        except Exception:
            pass

    # If we have time, do a Nelder-Mead polish on the best point
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points[1:].flatten()
        
        def objective_true(x):
            pts = np.zeros((N, D))
            pts[1:] = x
            return _true_score(pts)
            
        try:
            res_nm = scipy.optimize.minimize(
                objective_true, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            candidate_pts = np.zeros((N, D))
            candidate_pts[1:] = res_nm.x.reshape(N-1, D)
            
            true_s = _true_score(candidate_pts)
            
            if true_s < best_true_score:
                best_true_score = true_s
                best_points = candidate_pts
                
        except Exception:
            pass

    if best_points is None:
        # Fallback to grid (seed behavior)
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_points
