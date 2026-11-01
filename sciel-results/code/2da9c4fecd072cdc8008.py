import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    # Precompute all triangle index triples
    triples = list(itertools.combinations(range(N), 3))
    n_triples = len(triples)
    
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
        """Compute all triangle areas efficiently using vectorization"""
        areas = np.zeros(n_triples)
        for idx, (i, j, k) in enumerate(triples):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * abs(det)
        return areas
    
    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1.0
    
    def compute_min_area(pts):
        """Compute minimum triangle area and hull area"""
        areas = compute_areas_vectorized(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0, 1.0
        return min_area, hull_area
    
    def logsumexp_min_objective(x, beta=500.0):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        """
        pts = get_points(x)
        areas = compute_areas_vectorized(pts)
        
        # Numerically stable log-sum-exp
        a_max = np.max(areas)
        if a_max < 1e-15:
            return 1.0
        
        # log(sum(exp(-beta * a))) = -beta * a_max + log(sum(exp(-beta * (a - a_max))))
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        
        return -(smooth_min / hull_area)
    
    def raw_objective(x):
        """Raw objective: -min_area / hull_area"""
        pts = get_points(x)
        min_area, hull_area = compute_min_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    def gradient_smoothed(x, beta=500.0):
        """
        Analytical gradient of the smoothed objective.
        f(x) = -(1/beta) * log(sum(exp(-beta * A_i(x)))) / H(x)
        
        where A_i are triangle areas and H is hull area.
        For simplicity, we approximate gradient numerically with high precision.
        """
        # Use central differences with small step
        eps = 1e-7
        grad = np.zeros(len(x))
        f0 = logsumexp_min_objective(x, beta)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            f_plus = logsumexp_min_objective(x_plus, beta)
            f_minus = logsumexp_min_objective(x_minus, beta)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    def initial_circle(seed_offset=0.0):
        """Generate initial points on a circle"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = np.cos(angles[idx])
            x[2 * i + 1] = np.sin(angles[idx])
        return x
    
    def initial_elliptic(ratio=0.7, seed_offset=0.0):
        """Generate initial points on an ellipse"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = np.cos(angles[idx])
            x[2 * i + 1] = ratio * np.sin(angles[idx])
        return x
    
    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    # Phase 1: Global search with differential evolution
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    try:
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-5,
            init='sobol',
            polish=True
        )
        if result_de.fun < -best_score:
            best_x = result_de.x.copy()
            best_score = -result_de.fun
    except Exception:
        pass
    
    # Phase 2: Multi-start local optimization with increasing beta (annealing)
    # Start with lower beta (smoother landscape) and increase to sharpen
    
    betas_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    # Multiple diverse starting points
    start_points = []
    for angle_offset in [0.0, 0.3, 0.7, 1.2, 1.8]:
        start_points.append(initial_circle(angle_offset))
    for ratio in [0.5, 0.7, 0.85]:
        for angle_offset in [0.0, 0.5]:
            start_points.append(initial_elliptic(ratio, angle_offset))
    
    # Add the DE result if available
    if best_x is not None:
        start_points.append(best_x.copy())
    
    for beta in betas_schedule:
        if time.time() - start_time > time_limit:
            break
        
        for x0 in start_points:
            if time.time() - start_time > time_limit:
                break
            
            # Try L-BFGS-B with numerical gradient
            try:
                res = minimize(
                    logsumexp_min_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (N - 2),
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < -best_score:
                    best_x = res.x.copy()
                    best_score = -res.fun
            except Exception:
                pass
            
            # Try BFGS without bounds
            try:
                res2 = minimize(
                    logsumexp_min_objective,
                    x0,
                    args=(beta,),
                    method='BFGS',
                    options={'maxiter': 500, 'gtol': 1e-10}
                )
                
                if res2.fun < -best_score:
                    best_x = res2.x.copy()
                    best_score = -res2.fun
            except Exception:
                pass
        
        # Update start points with best found so far
        if best_x is not None:
            start_points = [best_x.copy()] + start_points[:3]
    
    # Phase 3: Final high-precision refinement with very high beta
    if time.time() - start_time < time_limit - 5.0 and best_x is not None:
        for beta_final in [1000.0, 2000.0]:
            if time.time() - start_time > time_limit - 3.0:
                break
            
            try:
                res_final = minimize(
                    logsumexp_min_objective,
                    best_x,
                    args=(beta_final,),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (N - 2),
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    # Phase 4: Final Nelder-Mead on raw objective for last-mile improvement
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < -best_score:
                best_x = res_nm.x.copy()
                best_score = -res_nm.fun
        except Exception:
            pass
    
    # Phase 5: One more polish with very tight tolerance
    if time.time() - start_time < time_limit - 1.0 and best_x is not None:
        try:
            res_polish = minimize(
                logsumexp_min_objective,
                best_x,
                args=(5000.0,),
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (N - 2),
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_polish.fun < -best_score:
                best_x = res_polish.x.copy()
                best_score = -res_polish.fun
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return circle configuration
        best_x = initial_circle(0.0)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
