import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses log-sum-exp smoothing with multi-scale annealing and SLSQP for
    better handling of the non-smooth minimum function.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    
    def compute_areas(pts):
        pi = pts[tri_i]
        pj = pts[tri_j]
        pk = pts[tri_k]
        v1 = pj - pi
        v2 = pk - pi
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        return areas
    
    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def objective(x, beta):
        """Smoothed objective: -smooth_min_area / hull_area"""
        pts = x.reshape(N, 2)
        
        # Center the points to remove translation invariance
        pts = pts - np.mean(pts, axis=0)
        
        areas = compute_areas(pts)
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        # Numerically stable log-sum-exp for min
        a_ref = np.min(areas)
        exp_terms = np.exp(-beta * (areas - a_ref))
        log_sum = -beta * a_ref + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
    
    def objective_raw(x):
        """Raw objective: -min_area / hull_area"""
        pts = x.reshape(N, 2)
        pts = pts - np.mean(pts, axis=0)
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def reg_polygon(n_points, radius=1.0, angle_offset=0.0):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + angle_offset
        pts = np.zeros((n_points, 2))
        pts[:, 0] = radius * np.cos(angles)
        pts[:, 1] = radius * np.sin(angles)
        return pts
    
    def two_ring(n_points, r1=0.5, r2=1.0, n_inner=7, angle_offset=0.0):
        """Two concentric rings"""
        pts = np.zeros((n_points, 2))
        n_outer = n_points - n_inner
        # Inner ring
        angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + angle_offset
        pts[:n_inner, 0] = r1 * np.cos(angles_in)
        pts[:n_inner, 1] = r1 * np.sin(angles_in)
        # Outer ring
        angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + angle_offset
        pts[n_inner:, 0] = r2 * np.cos(angles_out)
        pts[n_inner:, 1] = r2 * np.sin(angles_out)
        return pts
    
    def grid_init(n_points, scale=1.0):
        """Grid-like initialization"""
        pts = np.zeros((n_points, 2))
        cols = 4
        for i in range(n_points):
            row = i // cols
            col = i % cols
            pts[i, 0] = (col - 1.5) * scale
            pts[i, 1] = (row - 1.5) * scale
        return pts
    
    start_time = time.time()
    time_limit = 115.0
    dim = 2 * N
    
    best_score = -np.inf
    best_x = None
    
    # Generate diverse initial configurations
    def generate_initials():
        initials = []
        
        # Regular polygons with various radii and rotations
        for radius in [0.5, 0.7, 1.0, 1.3, 1.5]:
            for offset in [0.0, np.pi/14, np.pi/7, 2*np.pi/14, np.pi/3]:
                pts = reg_polygon(N, radius=radius, angle_offset=offset)
                initials.append(pts.flatten())
        
        # Two-ring configurations
        for r1 in [0.3, 0.5, 0.7]:
            for r2 in [1.0, 1.2, 1.5]:
                for offset in [0.0, np.pi/7, np.pi/3]:
                    pts = two_ring(N, r1=r1, r2=r2, n_inner=7, angle_offset=offset)
                    initials.append(pts.flatten())
        
        # Grid configurations
        for scale in [0.5, 0.7, 1.0, 1.3]:
            pts = grid_init(N, scale=scale)
            initials.append(pts.flatten())
        
        # Random configurations
        for _ in range(10):
            pts = np.random.uniform(-1.5, 1.5, size=(N, 2))
            initials.append(pts.flatten())
        
        return initials
    
    # Phase 1: Coarse optimization with multiple starts
    betas_coarse = [50.0, 100.0, 200.0, 500.0]
    initials = generate_initials()
    
    for beta in betas_coarse:
        if time.time() - start_time > time_limit - 30.0:
            break
        
        # Sort initials by current score, try best ones first
        scored = []
        for x0 in initials:
            try:
                score = -objective(x0, beta)
                scored.append((score, x0))
            except:
                scored.append((-1e-10, x0))
        scored.sort(reverse=True)
        
        # Try top 8 starts
        for _, x0 in scored[:8]:
            if time.time() - start_time > time_limit - 30.0:
                break
            
            try:
                res = minimize(
                    objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-3.0, 3.0)] * dim,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
        
        # Update initials with perturbed best
        if best_x is not None:
            new_initials = [best_x.copy()]
            for sigma in [0.1, 0.2, 0.3, 0.5]:
                for _ in range(4):
                    jitter = np.random.normal(0, sigma, size=dim)
                    new_initials.append(best_x + jitter)
            new_initials.extend([x for _, x in scored[:5]])
            initials = new_initials
    
    # Phase 2: Fine-tuning
    betas_fine = [1000.0, 2000.0, 5000.0, 10000.0]
    
    if best_x is not None:
        for beta in betas_fine:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            candidates = [best_x.copy()]
            for sigma in [0.02, 0.05, 0.1]:
                for _ in range(6):
                    jitter = np.random.normal(0, sigma, size=dim)
                    candidates.append(best_x + jitter)
            
            for x0 in candidates:
                if time.time() - start_time > time_limit - 15.0:
                    break
                
                try:
                    res = minimize(
                        objective,
                        x0,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=[(-3.0, 3.0)] * dim,
                        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 20000}
                    )
                    
                    if res.fun < -best_score:
                        best_x = res.x.copy()
                        best_score = -res.fun
                except Exception:
                    pass
    
    # Phase 3: SLSQP refinement for better handling of the minimum
    if time.time() - start_time < time_limit - 10.0 and best_x is not None:
        for beta_final in [20000.0, 50000.0]:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            try:
                res = minimize(
                    objective,
                    best_x,
                    args=(beta_final,),
                    method='SLSQP',
                    bounds=[(-3.0, 3.0)] * dim,
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
    
    # Phase 4: Nelder-Mead polish on raw objective
    if time.time() - start_time < time_limit - 3.0 and best_x is not None:
        try:
            res_raw = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_raw.fun < -best_score:
                best_x = res_raw.x.copy()
                best_score = -res_raw.fun
        except Exception:
            pass
    
    # Fallback
    if best_x is None:
        pts = reg_polygon(N, radius=1.0)
        best_x = pts.flatten()
    
    # Final: center the result
    final_pts = best_x.reshape(N, 2)
    final_pts = final_pts - np.mean(final_pts, axis=0)
    
    return final_pts
