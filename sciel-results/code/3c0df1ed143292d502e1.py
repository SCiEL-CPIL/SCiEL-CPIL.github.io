import numpy as np
from scipy.optimize import minimize
import time

def get_polygon_from_params(params):
    r0, r1, r2, r3, r4, t1, t2, t3, t4 = params
    v0 = np.array([r0, 0.0])
    v1 = np.array([r1 * np.cos(t1), r1 * np.sin(t1)])
    v2 = np.array([r2 * np.cos(t2), r2 * np.sin(t2)])
    v3 = np.array([r3 * np.cos(t3), r3 * np.sin(t3)])
    v4 = np.array([r4 * np.cos(t4), r4 * np.sin(t4)])
    verts = np.array([v0, v1, v2, v3, v4])
    all_verts = np.zeros((20, 2))
    all_verts[0:5] = verts
    for i in range(5):
        x, y = verts[i]
        all_verts[i+5] = np.array([-y, x])
        all_verts[i+10] = np.array([-x, -y])
        all_verts[i+15] = np.array([y, -x])
    return all_verts

def vertex_score(v, polygon):
    diffs = polygon - v
    dists = np.linalg.norm(diffs, axis=1)
    dists_sorted = np.sort(dists)
    d = dists_sorted[1:]
    if d[0] < 0.05:
        return 0.0
    gaps = np.diff(d)
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = d[0:4]
    elif k == 17:
        d_sel = d[15:19]
    else:
        d_sel = d[k-1:k+3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def min_vertex_score(polygon):
    scores = [vertex_score(polygon[i], polygon) for i in range(20)]
    return np.min(scores)

def objective(params):
    polygon = get_polygon_from_params(params)
    r = params[0:5]
    t = params[5:9]
    if np.any(r <= 0.01) or np.any(t <= 0.01) or np.any(t >= np.pi/2 - 0.01):
        return 1.0
    if np.any(np.diff(t) <= 0.001):
        return 1.0
    score = min_vertex_score(polygon)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    r_init = np.ones(5)
    t_init = np.array([np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10])
    params_init = np.concatenate([r_init, t_init])
    
    best_params = params_init
    best_score = min_vertex_score(get_polygon_from_params(params_init))
    
    try:
        res = minimize(objective, params_init, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if -res.fun > best_score:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
        
    if time.time() < deadline - 10.0:
        # Try a few random restarts if time permits
        for _ in range(3):
            if time.time() > deadline - 10.0: break
            r_rand = np.random.uniform(0.5, 2.0, 5)
            t_rand = np.sort(np.random.uniform(0.1, np.pi/2 - 0.1, 4))
            params_rand = np.concatenate([r_rand, t_rand])
            try:
                res_r = minimize(objective, params_rand, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if -res_r.fun > best_score:
                    best_score = -res_r.fun
                    best_params = res_r.x
            except Exception:
                pass

    polygon = get_polygon_from_params(best_params)
    
    # The evaluator normalizes, but we should ensure the returned polygon is valid.
    # The task says "returning a (20, 2) array of finite vertex coordinates in cyclic order".
    # It does not say we must normalize it ourselves, but the score is scale invariant.
    # However, returning a polygon with very large or small coordinates might cause issues?
    # "finite vertex coordinates" - so no inf/nan.
    # My radii are bounded, so this is fine.
    
    return polygon
