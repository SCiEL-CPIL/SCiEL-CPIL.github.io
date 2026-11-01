import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    """
    Maximize minimum triangle area / convex hull area for 14 points.
    Uses log-sum-exp smoothing to create a differentiable objective,
    then applies multi-scale annealing with high-precision refinement.
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
        # Vectorized computation of all triangle areas
        pi = pts[tri_i]  # (n_triples, 2)
        pj = pts[tri_j]  # (n_triples, 2)
        pk = pts[tri_k]  # (n_triples, 2)
        
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
        min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        # Numerically stable log-sum-exp
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1.0
        
        # Use the smaller areas for numerical stability
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
        """Convert point array (with fixed pts[0], pts[1]) to parameter vector x"""
        x = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            x[2 * i] = pts[i + 2, 0]
            x[2 * i + 1] = pts[i + 2, 1]
        return x
    
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
        for radius in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for angle_offset in [0.0, np.pi/14, np.pi/7, 2*np.pi/14]:
                pts = initial_reg_polygon(N, center=(0.5, 0.5), radius=radius, angle_offset=angle_offset)
                # Fix first two points
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                initials.append(pts_to_x(pts))
        
        # Elliptical distributions
        for rx in [0.4, 0.5, 0.6, 0.7]:
            for ry in [0.3, 0.4, 0.5, 0.6]:
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                pts = np.zeros((N, 2))
                for i in range(N):
                    pts[i, 0] = 0.5 + rx * np.cos(angles[i])
                    pts[i, 1] = 0.5 + ry * np.sin(angles[i])
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                initials.append(pts_to_x(pts))
        
        # Grid-like distributions with perturbations
        for scale in [0.8, 1.0, 1.2]:
            base_x = np.linspace(0, 1, 4)
            base_y = np.linspace(0, 1, 4)
            grid_pts = []
            for gx in base_x:
                for gy in base_y:
                    if len(grid_pts) < N:
                        grid_pts.append([gx * scale, gy * scale])
            while len(grid_pts) < N:
                grid_pts.append([np.random.uniform(0, scale), np.random.uniform(0, scale)])
            pts = np.array(grid_pts[:N])
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            initials.append(pts_to_x(pts))
        
        return initials
    
    # Phase 1: Multi-start with coarse-to-fine beta annealing
    betas_coarse = [20.0, 50.0, 100.0, 200.0, 400.0]
    initials = generate_initials()
    
    for beta in betas_coarse:
        if time.time() - start_time > time_limit - 20.0:
            break
        
        for x0 in initials:
            if time.time() - start_time > time_limit - 20.0:
                break
            
            try:
                res = minimize(
                    objective_smooth,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * dim,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
        
        # Keep only the best few starting points for next iteration
        if best_x is not None:
            # Create perturbed versions of the best solution
            new_initials = [best_x.copy()]
            for sigma in [0.1, 0.2, 0.3]:
                for _ in range(3):
                    jitter = np.random.normal(0, sigma, size=dim)
                    new_initials.append(best_x + jitter)
            # Add some original initials for diversity
            new_initials.extend(initials[:5])
            initials = new_initials
    
    # Phase 2: Fine-tuning with higher beta values
    betas_fine = [800.0, 1500.0, 3000.0, 5000.0, 8000.0]
    
    if best_x is not None:
        for beta in betas_fine:
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Try multiple perturbations around the current best
            candidates = [best_x.copy()]
            for sigma in [0.05, 0.1, 0.15]:
                for _ in range(5):
                    jitter = np.random.normal(0, sigma, size=dim)
                    candidates.append(best_x + jitter)
            
            for x0 in candidates:
                if time.time() - start_time > time_limit - 10.0:
                    break
                
                try:
                    res = minimize(
                        objective_smooth,
                        x0,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=[(-2.0, 2.0)] * dim,
                        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
                    )
                    
                    if res.fun < -best_score:
                        best_x = res.x.copy()
                        best_score = -res.fun
                except Exception:
                    pass
    
    # Phase 3: Ultra-high precision refinement
    if time.time() - start_time < time_limit - 5.0 and best_x is not None:
        for beta_final in [10000.0, 20000.0, 50000.0]:
            if time.time() - start_time > time_limit - 3.0:
                break
            
            try:
                res_final = minimize(
                    objective_smooth,
                    best_x,
                    args=(beta_final,),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * dim,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 4: Direct optimization on raw objective for final polish
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            res_raw = minimize(
                objective_raw,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res_raw.fun < -best_score:
                best_x = res_raw.x.copy()
                best_score = -res_raw.fun
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
