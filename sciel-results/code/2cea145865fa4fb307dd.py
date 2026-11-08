import scipy
import numpy as np
import time
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 13 points to maximize the minimum triangle area divided by the convex hull area.
    Uses vectorized scoring and smooth surrogate optimization starting from a regular 13-gon.
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
        # Use soft min: -log(sum(exp(-a/T)))/T approximates min(a)
        # Normalize by hull area
        ratio_areas = areas / ha
        scaled = -ratio_areas / temperature
        # Numerically stable log-sum-exp
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -temperature * log_sum
        return float(soft_min)
    
    # Initial configuration: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = initial_points.copy()
    best_score = _exact_score(initial_points)
    
    if time.time() > deadline:
        return best_points
    
    # Phase 1: Optimize soft-min surrogate with L-BFGS-B for fast convergence
    # The soft-min is smooth and differentiable, allowing gradient-based methods
    
    def soft_objective(x):
        points = x.reshape((N, 2))
        s = _soft_min_score(points, temperature=0.01)
        return -s
    
    def soft_callback(xk):
        if time.time() > deadline - 5.0:
            raise TimeoutError()
    
    # Try several temperatures for the soft-min
    temperatures = [0.05, 0.02, 0.01, 0.005]
    
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
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
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
    
    # Phase 2: Fine-tune with Nelder-Mead on exact score
    def exact_objective(x):
        points = x.reshape((N, 2))
        s = _exact_score(points)
        return -s
    
    def exact_callback(xk):
        if time.time() > deadline - 3.0:
            raise TimeoutError()
    
    # A few Nelder-Mead restarts from best point
    n_nm_restarts = 3
    for r in range(n_nm_restarts):
        if time.time() > deadline - 3.0:
            break
        
        if r == 0:
            x0 = best_points.flatten()
        else:
            noise = np.random.normal(0, 0.02, (N, 2))
            x0 = (best_points + noise).flatten()
        
        try:
            res = minimize(exact_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
                           callback=exact_callback)
            
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
