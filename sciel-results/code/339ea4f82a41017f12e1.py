import numpy as np
from scipy.optimize import minimize
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
    
    def compute_all_areas(pts):
        """Compute all triangle areas vectorized"""
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
    
    def logsumexp_min_objective(x, beta):
        """
        Smoothed objective: -min_area / hull_area using log-sum-exp.
        Returns (value, gradient)
        """
        pts = get_points(x)
        areas = compute_all_areas(pts)
        
        if np.max(areas) < 1e-15:
            return 1.0, np.zeros_like(x)
        
        # Numerically stable log-sum-exp for min
        a_max = np.max(areas)
        exp_terms = np.exp(-beta * (areas - a_max))
        log_sum = -beta * a_max + np.log(np.sum(exp_terms))
        smooth_min = -log_sum / beta
        
        # Gradient weights
        weights = exp_terms / np.sum(exp_terms)
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0, np.zeros_like(x)
        
        value = -(smooth_min / hull_area)
        
        # Gradient of smooth_min w.r.t. x
        grad_smooth = np.zeros_like(x)
        
        for idx, (i, j, k) in enumerate(triples):
            w = weights[idx]
            if w < 1e-15:
                continue
            
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(det) < 1e-15:
                continue
            sign = np.sign(det)
            
            d_det_di = np.array([pts[j][1] - pts[k][1], pts[k][0] - pts[j][0]])
            d_det_dj = np.array([pts[k][1] - pts[i][1], pts[i][0] - pts[k][0]])
            d_det_dk = np.array([pts[i][1] - pts[j][1], pts[j][0] - pts[i][0]])
            
            da_di = 0.5 * sign * d_det_di
            da_dj = 0.5 * sign * d_det_dj
            da_dk = 0.5 * sign * d_det_dk
            
            if i >= 2:
                xi = i - 2
                grad_smooth[2*xi:2*xi+2] += w * da_di
            if j >= 2:
                xj = j - 2
                grad_smooth[2*xj:2*xj+2] += w * da_dj
            if k >= 2:
                xk = k - 2
                grad_smooth[2*xk:2*xk+2] += w * da_dk
        
        # Numerical gradient for hull_area
        eps = 1e-7
        hull_grad = np.zeros_like(x)
        for dim_i in range(len(x)):
            x_plus = x.copy()
            x_plus[dim_i] += eps
            x_minus = x.copy()
            x_minus[dim_i] -= eps
            ha_plus = compute_hull_area(get_points(x_plus))
            ha_minus = compute_hull_area(get_points(x_minus))
            hull_grad[dim_i] = (ha_plus - ha_minus) / (2 * eps)
        
        grad = -grad_smooth / hull_area + smooth_min * hull_grad / (hull_area ** 2)
        
        return value, grad
    
    def initial_regular_polygon(radius=0.5):
        """Generate initial points on a regular polygon"""
        x = np.zeros(2 * (N - 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = 0.5 + radius * np.cos(angles[idx])
            x[2 * i + 1] = radius * np.sin(angles[idx])
        return x
    
    # Time management
    start_time = time.time()
    time_limit = 115.0
    
    best_score = -np.inf
    best_x = None
    
    dim = 2 * (N - 2)
    bounds = [(-2.0, 2.0)] * dim
    
    # Multi-scale optimization with gradient-based methods
    beta_schedule = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    x0 = initial_regular_polygon(radius=0.5)
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # SLSQP with gradient
        try:
            res = minimize(
                logsumexp_min_objective,
                x0,
                args=(beta,),
                method='SLSQP',
                bounds=bounds,
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.fun < -best_score:
                best_x = res.x.copy()
                best_score = -res.fun
                x0 = res.x.copy()
        except Exception:
            pass
        
        # L-BFGS-B with gradient
        try:
            res2 = minimize(
                logsumexp_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res2.fun < -best_score:
                best_x = res2.x.copy()
                best_score = -res2.fun
                x0 = res2.x.copy()
        except Exception:
            pass
        
        # Multi-start with perturbations
        if time.time() - start_time < time_limit - 10.0:
            for n_pert in range(3):
                if time.time() - start_time > time_limit - 5.0:
                    break
                perturbation = np.random.normal(0, 0.05, size=dim)
                x_pert = (best_x + perturbation) if best_x is not None else (x0 + perturbation)
                
                try:
                    res_pert = minimize(
                        logsumexp_min_objective,
                        x_pert,
                        args=(beta,),
                        method='SLSQP',
                        bounds=bounds,
                        jac=True,
                        options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                    )
                    
                    if res_pert.fun < -best_score:
                        best_x = res_pert.x.copy()
                        best_score = -res_pert.fun
                        x0 = res_pert.x.copy()
                except Exception:
                    pass
    
    # Final refinement
    if time.time() - start_time < time_limit - 2.0 and best_x is not None:
        for beta_final in [10000.0, 20000.0]:
            if time.time() - start_time > time_limit - 1.0:
                break
            
            try:
                res_final = minimize(
                    logsumexp_min_objective,
                    best_x,
                    args=(beta_final,),
                    method='SLSQP',
                    bounds=bounds,
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
                )
                
                if res_final.fun < -best_score:
                    best_x = res_final.x.copy()
                    best_score = -res_final.fun
            except Exception:
                pass
    
    if best_x is None:
        best_x = initial_regular_polygon(radius=0.5)
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
