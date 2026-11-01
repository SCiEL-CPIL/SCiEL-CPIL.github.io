import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas(pts):
        areas = np.empty(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * abs(det)
        return areas
    
    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def smooth_obj(x, beta):
        pts = get_points(x)
        areas = compute_areas(pts)
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 1.0
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(smooth_min / hull_area)
    
    def raw_obj(x):
        pts = get_points(x)
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def init_config(offset=0.0, rx=1.0, ry=1.0, n_center=0):
        x = np.zeros(2 * (N - 2))
        if n_center > 0:
            for i in range(n_center):
                ang = 2 * np.pi * i / n_center + offset
                x[2*i] = 0.1 * np.cos(ang)
                x[2*i+1] = 0.1 * np.sin(ang)
            n_out = (N - 2) - n_center
            for i in range(n_out):
                ang = 2 * np.pi * i / n_out + offset
                x[2*(n_center+i)] = rx * np.cos(ang)
                x[2*(n_center+i)+1] = ry * np.sin(ang)
        else:
            for i in range(N - 2):
                ang = 2 * np.pi * i / (N - 2) + offset
                x[2*i] = rx * np.cos(ang)
                x[2*i+1] = ry * np.sin(ang)
        return x
    
    start_time = time.time()
    time_limit = 115.0
    best_score = -np.inf
    best_x = None
    dim = 2 * (N - 2)
    bounds = [(-5.0, 5.0)] * dim
    
    # Build diverse initial set
    starts = []
    for off in np.linspace(0, 2*np.pi, 10, endpoint=False):
        starts.append(init_config(offset=off))
    for ry in [0.2, 0.4, 0.6, 0.8, 0.9]:
        for off in [0.0, np.pi/5, 2*np.pi/5]:
            starts.append(init_config(offset=off, rx=1.0, ry=ry))
    for nc in [1, 2, 3]:
        for off in [0.0, np.pi/3, 2*np.pi/3]:
            starts.append(init_config(offset=off, n_center=nc))
    np.random.seed(7)
    base = init_config()
    for s in [0.05, 0.1, 0.2, 0.4]:
        for _ in range(4):
            starts.append(base + np.random.normal(0, s, dim))
    
    # Phase 1: DE global search
    try:
        de_res = differential_evolution(
            raw_obj, bounds, seed=99, maxiter=25, popsize=14,
            tol=1e-3, init='sobol', polish=False,
            mutation=(0.6, 1.4), recombination=0.8
        )
        if de_res.fun < -best_score:
            best_x = de_res.x.copy()
            best_score = -de_res.fun
    except Exception:
        pass
    
    if best_x is not None:
        starts.insert(0, best_x.copy())
    
    # Phase 2: Annealing with L-BFGS-B and SLSQP
    betas = [15.0, 40.0, 100.0, 250.0, 600.0, 1500.0, 4000.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 18.0:
            break
        
        scored = []
        for x0 in starts:
            try:
                v = smooth_obj(x0, beta)
                scored.append((v, x0))
            except Exception:
                pass
        scored.sort(key=lambda t: t[0])
        
        K = min(10, len(scored))
        for _, x0 in scored[:K]:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            try:
                r = minimize(smooth_obj, x0, args=(beta,), method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 1200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 12000})
                if r.fun < -best_score:
                    best_x = r.x.copy()
                    best_score = -r.fun
            except Exception:
                pass
            
            if time.time() - start_time < time_limit - 12.0:
                try:
                    r2 = minimize(smooth_obj, x0, args=(beta,), method='SLSQP',
                                  bounds=bounds, options={'maxiter': 400, 'ftol': 1e-12})
                    if r2.fun < -best_score:
                        best_x = r2.x.copy()
                        best_score = -r2.fun
                except Exception:
                    pass
        
        if best_x is not None:
            ns = [best_x.copy()]
            np.random.seed(int(time.time()*1000) % 99999)
            for s in [0.01, 0.03, 0.08, 0.15]:
                ns.append(best_x + np.random.normal(0, s, dim))
            starts = ns + scored[:4]
    
    # Phase 3: High-precision refinement
    if time.time() - start_time < time_limit - 12.0 and best_x is not None:
        for bf in [8000.0, 20000.0, 50000.0]:
            if time.time() - start_time > time_limit - 6.0:
                break
            try:
                rf = minimize(smooth_obj, best_x, args=(bf,), method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 2500, 'ftol': 1e-15, 'gtol': 1e-13})
                if rf.fun < -best_score:
                    best_x = rf.x.copy()
                    best_score = -rf.fun
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead on raw
    if time.time() - start_time < time_limit - 5.0 and best_x is not None:
        try:
            rn = minimize(raw_obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if rn.fun < -best_score:
                best_x = rn.x.copy()
                best_score = -rn.fun
        except Exception:
            pass
    
    # Phase 5: Final polish
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            rp = minimize(smooth_obj, best_x, args=(100000.0,), method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 4000, 'ftol': 1e-16, 'gtol': 1e-14})
            if rp.fun < -best_score:
                best_x = rp.x.copy()
                best_score = -rp.fun
        except Exception:
            pass
    
    if best_x is None:
        best_x = init_config()
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
