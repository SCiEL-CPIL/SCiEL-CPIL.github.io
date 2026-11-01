import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses log-sum-exp smoothing with SLSQP and aggressive annealing.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle indices for vectorized area computation
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    
    def get_points(x):
        """
        x has 2*(N-2) elements: coordinates of points 2..N-1
        Points 0 and 1 are fixed at (0,0) and (1,0)
        """
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas_vectorized(pts):
        """Compute all triangle areas using vectorized operations"""
        pi = pts[tri_i]
        pj = pts[tri_j]
        pk = pts[tri_k]
        
        v1 = pj - pi
        v2 = pk - pi
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        return areas
    
    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def objective_smooth(x, beta):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
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
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def initial_reg_polygon(n_points, center=(0.5, 0.5), radius=0.5, angle_offset=0.0):
        """Generate points on a regular polygon"""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + angle_offset
        pts = np.zeros((n_points, 2))
        for i in range(n_points):
            pts[i, 0] = center[0] + radius * np.cos(angles[i])
            pts[i, 1] = center[1] + radius * np.sin(angles[i])
        return pts
    
    def pts_to_x(pts):
        """Convert point array to parameter vector x"""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x
    
    def local_search(x, max_iter=200, tol=1e-12):
        """Coordinate descent local search on raw objective"""
        current = x.copy()
        current_score = -objective_raw(current)
        
        for _ in range(max_iter):
            improved = False
            for i in range(2 * (N - 2)):
                for delta in [0.001, -0.001, 0.01, -0.01, 0.0001, -0.0001]:
                    trial = current.copy()
                    trial[i] += delta
                    trial_score = -objective_raw(trial)
                    if trial_score > current_score + tol:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
            if not improved:
                break
        
        return current, current_score
    
    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * (N - 2)
    
    # Generate diverse initial configurations
    def generate_initials():
        initials = []
        
        # Regular polygon centered at (0.5, 0.5) with various radii
        for radius in [0.35, 0.45, 0.55, 0.65]:
            for angle_offset in [0.0, np.pi/14, 2*np.pi/14]:
                pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=radius, angle_offset=angle_offset)
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                initials.append(pts_to_x(pts))
        
        # Elliptical distributions
        for rx in [0.45, 0.55, 0.65]:
            for ry in [0.35, 0.45, 0.55]:
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                pts = np.zeros((N, 2))
                for i in range(N):
                    pts[i, 0] = 0.5 + rx * np.cos(angles[i])
                    pts[i, 1] = 0.5 + ry * np.sin(angles[i])
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                initials.append(pts_to_x(pts))
        
        # Random perturbations of regular polygon
        base_pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=0.5)
        base_pts[0] = [0.0, 0.0]
        base_pts[1] = [1.0, 0.0]
        base_x = pts_to_x(base_pts)
        for _ in range(5):
            jitter = np.random.normal(0, 0.1, size=dim)
            initials.append(base_x + jitter)
        
        return initials
    
    # Phase 1: Coarse optimization with SLSQP
    betas_coarse = [10.0, 30.0, 100.0, 300.0]
    initials = generate_initials()
    
    for beta in betas_coarse:
        if time.time() - start_time > time_limit - 30.0:
            break
        
        for x0 in initials:
            if time.time() - start_time > time_limit - 30.0:
                break
            
            try:
                res = minimize(
                    objective_smooth,
                    x0,
                    args=(beta,),
                    method='SLSQP',
                    bounds=[(-1.5, 1.5)] * dim,
                    options={'maxiter': 300, 'ftol': 1e-10}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
        
        # Update initials with best solution
        if best_x is not None:
            new_initials = [best_x.copy()]
            for sigma in [0.05, 0.1, 0.15, 0.2]:
                for _ in range(2):
                    jitter = np.random.normal(0, sigma, size=dim)
                    new_initials.append(best_x + jitter)
            new_initials.extend(initials[:3])
            initials = new_initials
    
    # Phase 2: Fine-tuning with higher beta
    betas_fine = [500.0, 1000.0, 2000.0, 4000.0]
    
    if best_x is not None:
        for beta in betas_fine:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            candidates = [best_x.copy()]
            for sigma in [0.02, 0.05, 0.1]:
                for _ in range(3):
                    jitter = np.random.normal(0, sigma, size=dim)
                    candidates.append(best_x + jitter)
            
            for x0 in candidates:
                if time.time() - start_time > time_limit - 15.0:
                    break
                
                try:
                    res = minimize(
                        objective_smooth,
                        x0,
                        args=(beta,),
                        method='SLSQP',
                        bounds=[(-1.5, 1.5)] * dim,
                        options={'maxiter': 500, 'ftol': 1e-12}
                    )
                    
                    if res.fun < -best_score:
                        best_x = res.x.copy()
                        best_score = -res.fun
                except Exception:
                    pass
    
    # Phase 3: Ultra-high precision with very high beta
    if time.time() - start_time < time_limit - 8.0 and best_x is not None:
        for beta_final in [8000.0, 15000.0, 30000.0]:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            try:
                res_final = minimize(
                    objective_smooth,
                    best_x,
                    args=(beta_final,),
                    method='SLSQP',
                    bounds=[(-1.5, 1.5)] * dim,
                    options={'maxiter': 1000, 'ftol': 1e-14}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 4: Local search refinement
    if time.time() - start_time < time_limit - 3.0 and best_x is not None:
        try:
            best_x, local_score = local_search(best_x, max_iter=100)
            if local_score > best_score:
                best_score = local_score
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        try:
            res_nm = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_x = res_nm.x.copy()
                best_score = -res_nm.fun
        except Exception:
            pass
    
    # Fallback
    if best_x is None:
        pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=0.5)
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        best_x = pts_to_x(pts)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
