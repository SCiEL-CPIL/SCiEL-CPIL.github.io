import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time
from scipy.stats import qmc

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    New design:
    1. Use Sobol sequences for high-quality initial points.
    2. Use trust-constr optimizer with numerical Hessian for robustness.
    3. Multi-scale annealing with dynamic restarts.
    """
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
    # Precompute triangle indices
    tri_i = np.array([t[0] for t in triples])
    tri_j = np.array([t[1] for t in triples])
    tri_k = np.array([t[2] for t in triples])
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas_vectorized(pts):
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
    
    def objective_smooth(x, beta):
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        # Log-sum-exp smoothing
        a_ref = np.min(areas)
        exp_terms = np.exp(-beta * (areas - a_ref))
        log_sum = -beta * a_ref + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
        
    def objective_raw(x):
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)

    def pts_to_x(pts):
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x

    # Generate initial points using Sobol sequence
    def generate_sobol_initials(num_samples=20):
        sampler = qmc.Sobol(d=N-2, scramble=True)
        samples = sampler.random(num_samples)
        # Map to [0, 1] or similar range
        # We want points roughly in [0, 1]x[0, 1]
        samples = samples * 1.0 
        initials = []
        for s in samples:
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            # Assign Sobol coordinates to remaining points
            # We have N-2 points, each needs 2 coords. Sobol gives N-2 coords?
            # No, Sobol d=N-2 gives 1 coord per point. We need 2.
            # Let's use 2*N-2 dimensions for Sobol.
            pass
        # Better: Generate 2*(N-2) dimensional Sobol samples
        sampler2 = qmc.Sobol(d=2*(N-2), scramble=True)
        samples2 = sampler2.random(num_samples)
        for s in samples2:
            x = s.copy()
            # Scale to a reasonable range, e.g., [-0.5, 1.5]
            x = x * 2.0 - 0.5
            initials.append(x)
        return initials

    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * (N - 2)
    bounds = [(-2.0, 2.0)] * dim
    
    # Phase 1: Sobol-based multi-start with coarse optimization
    initials = generate_sobol_initials(30)
    
    # Add a few regular polygon starts as fallback
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([0.5 + 0.5*np.cos(angles), 0.5 + 0.5*np.sin(angles)])
    reg_pts[0] = [0.0, 0.0]
    reg_pts[1] = [1.0, 0.0]
    initials.append(pts_to_x(reg_pts))
    
    betas = [10, 50, 200, 1000, 5000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 15.0:
            break
            
        # Sort initials by current score to prioritize good ones?
        # For now, just iterate.
        for x0 in initials:
            if time.time() - start_time > time_limit - 15.0:
                break
            
            try:
                # Use trust-constr for better handling of constraints/curvature
                res = minimize(
                    objective_smooth,
                    x0,
                    args=(beta,),
                    method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': 500, 'gtol': 1e-6, 'xtol': 1e-8, 'verbose': 0}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
        
        # If we found a good solution, add perturbations for next beta
        if best_x is not None:
            new_initials = [best_x.copy()]
            for sigma in [0.05, 0.1]:
                for _ in range(5):
                    jitter = np.random.normal(0, sigma, size=dim)
                    new_initials.append(best_x + jitter)
            # Keep some diversity
            new_initials.extend(initials[:10])
            initials = new_initials
    
    # Phase 2: High precision refinement
    if best_x is not None:
        for beta in [10000, 50000]:
            if time.time() - start_time > time_limit - 5.0:
                break
            try:
                res = minimize(
                    objective_smooth,
                    best_x,
                    args=(beta,),
                    method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': 1000, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0}
                )
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass

    # Phase 3: Raw objective polish with Nelder-Mead
    if best_x is not None and time.time() - start_time < time_limit - 2.0:
        try:
            res_raw = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_raw.fun < -best_score:
                best_x = res_raw.x.copy()
                best_score = -res_raw.fun
        except Exception:
            pass

    if best_x is None:
        pts = np.column_stack([0.5 + 0.5*np.cos(angles), 0.5 + 0.5*np.sin(angles)])
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        best_x = pts_to_x(pts)
        
    return get_points(best_x)
