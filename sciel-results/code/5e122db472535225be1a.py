import scipy
import numpy as np
import time
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the minimum triangle area divided by the convex hull area.
    Uses vectorized scoring and smooth surrogate optimization with epigraph formulation.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Precompute triangle index pairs
    tri_idx = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_idx.append((i, j, k))
    tri_idx = np.array(tri_idx)  # shape (286, 3)
    
    def _triangle_areas(points):
        """Vectorized computation of all triangle areas."""
        p1 = points[tri_idx[:, 0]]
        p2 = points[tri_idx[:, 1]]
        p3 = points[tri_idx[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        return areas
    
    def _hull_area(points):
        """Compute convex hull area."""
        try:
            hull = scipy.spatial.ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0
    
    def _exact_score(points):
        """Exact score: min triangle area / hull area."""
        ha = _hull_area(points)
        if ha < 1e-9:
            return 0.0
        areas = _triangle_areas(points)
        return float(np.min(areas) / ha)
    
    def _soft_min_score(points, temperature=0.01):
        """Smooth approximation of min triangle area / hull area using log-sum-exp."""
        ha = _hull_area(points)
        if ha < 1e-9:
            return 0.0
        areas = _triangle_areas(points)
        ratio_areas = areas / ha
        scaled = -ratio_areas / temperature
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -temperature * log_sum
        return float(soft_min)
    
    def _epigraph_objective(x, temperature=0.01):
        """
        Epigraph formulation: maximize t subject to t <= soft_min(areas/ha).
        We minimize -t + penalty for violations.
        x = [points (N*2), t (1)]
        """
        points = x[:N*2].reshape((N, 2))
        t = x[N*2]
        
        ha = _hull_area(points)
        if ha < 1e-9:
            return 1e6  # Large penalty for degenerate hull
        
        areas = _triangle_areas(points)
        ratio_areas = areas / ha
        
        # Soft constraint: t should be <= min(ratio_areas)
        # Use smooth hinge: max(0, t - soft_min(ratio_areas))^2
        scaled = -ratio_areas / temperature
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -temperature * log_sum
        
        # Penalty for t exceeding soft_min
        violation = max(0, t - soft_min_val)
        penalty = violation * violation / (2 * temperature)
        
        # Objective: minimize -t + penalty
        return -t + penalty
    
    # Initial configuration: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = initial_points.copy()
    best_score = _exact_score(initial_points)
    
    if time.time() > deadline:
        return best_points
    
    # Phase 1: Optimize soft-min surrogate with L-BFGS-B
    # Start with moderate temperature, then decrease
    
    def soft_callback(xk):
        if time.time() > deadline - 5.0:
            raise TimeoutError()
    
    # Temperature schedule: start at 0.03, go down to 0.003
    temperatures = [0.03, 0.02, 0.01, 0.005, 0.003]
    
    for temp in temperatures:
        if time.time() > deadline - 5.0:
            break
        
        def obj_temp(x, t=temp):
            points = x.reshape((N, 2))
            s = _soft_min_score(points, temperature=t)
            return -s
        
        x0 = best_points.flatten()
        
        try:
            res = minimize(obj_temp, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
                           callback=soft_callback)
            
            current_points = res.x.reshape((N, 2))
            current_score = _exact_score(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 2: Epigraph optimization with auxiliary variable t
    # This gives a smoother optimization landscape
    
    def epigraph_callback(xk):
        if time.time() > deadline - 5.0:
            raise TimeoutError()
    
    # Initialize t with current best score
    t_init = best_score
    
    epigraph_temps = [0.01, 0.005, 0.003, 0.002]
    
    for temp in epigraph_temps:
        if time.time() > deadline - 5.0:
            break
        
        def obj_epigraph(x, t=temp):
            return _epigraph_objective(x, temperature=t)
        
        # x = [points, t]
        x0 = np.concatenate([best_points.flatten(), [t_init]])
        
        try:
            res = minimize(obj_epigraph, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
                           callback=epigraph_callback)
            
            current_points = res.x[:N*2].reshape((N, 2))
            current_score = _exact_score(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                t_init = current_score
                
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 3: High-precision polish with very low temperature soft-min
    def polish_callback(xk):
        if time.time() > deadline - 3.0:
            raise TimeoutError()
    
    polish_temps = [0.002, 0.001, 0.0005]
    
    for temp in polish_temps:
        if time.time() > deadline - 3.0:
            break
        
        def obj_polish(x, t=temp):
            points = x.reshape((N, 2))
            s = _soft_min_score(points, temperature=t)
            return -s
        
        x0 = best_points.flatten()
        
        try:
            res = minimize(obj_polish, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
                           callback=polish_callback)
            
            current_points = res.x.reshape((N, 2))
            current_score = _exact_score(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 4: Final local search with small perturbations
    def final_callback(xk):
        if time.time() > deadline - 2.0:
            raise TimeoutError()
    
    np.random.seed(42)
    for r in range(5):
        if time.time() > deadline - 2.0:
            break
        
        noise_scale = 0.01 * (r + 1)
        noise = np.random.normal(0, noise_scale, (N, 2))
        x0 = (best_points + noise).flatten()
        
        def obj_final(x):
            points = x.reshape((N, 2))
            s = _soft_min_score(points, temperature=0.001)
            return -s
        
        try:
            res = minimize(obj_final, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
                           callback=final_callback)
            
            current_points = res.x.reshape((N, 2))
            current_score = _exact_score(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        except TimeoutError:
            pass
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
