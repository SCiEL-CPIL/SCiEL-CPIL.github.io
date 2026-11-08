import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 140  # Safety margin for 150s limit
    
    # 1. Initialization: Regular 14-gon
    # Center at origin, radius 1
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Function to calculate the raw score
    def raw_score(points):
        # points shape (N, 2)
        # Calculate areas of all triangles
        # Using vectorized cross product for area
        # Area = 0.5 * |cross(P2-P1, P3-P1)|
        
        # To avoid O(N^3) python loops, we can use broadcasting
        # But N=14 is small, 364 triangles. Loop is fine and safer for clarity.
        areas = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Cross product magnitude
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas.append(area)
        
        min_area = min(areas)
        
        # Calculate Convex Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area
        except Exception:
            # If points are collinear or degenerate, return 0 or small penalty
            return 0.0
        
        if hull_area < 1e-10:
            return 0.0
            
        return min_area / hull_area

    # Function to calculate soft score for optimization
    # We want to maximize min(areas)/hull_area
    # Equivalent to minimizing - (sum (area/hull_area)^-alpha)^(-1/alpha)
    def soft_objective(points, alpha):
        # Check timeout
        if time.time() - start_time > timeout:
            return 0.0 # Just a dummy value, optimizer should be interrupted by wrapper if possible, 
                       # but scipy doesn't easily support timeouts. We rely on short steps.
        
        areas = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    p1, p2, p3 = points[i], points[j], points[k]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas.append(area)
        
        min_area = min(areas)
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 1e10 # Large penalty
            
        if hull_area < 1e-10:
            return 1e10
            
        ratios = np.array(areas) / hull_area
        # Avoid division by zero or log issues
        ratios = np.maximum(ratios, 1e-12)
        
        # Soft min approximation
        # sum(r^(-alpha))^(-1/alpha)
        inv_alpha = -1.0 / alpha
        val = np.sum(ratios ** (-alpha)) ** inv_alpha
        return -val # We minimize, so negative

    # Optimization Strategy
    # Anneal alpha from low to high
    alphas = [1.0, 2.0, 5.0, 10.0, 20.0]
    current_points = x0.copy()
    
    for alpha in alphas:
        if time.time() - start_time > timeout:
            break
            
        # Define objective for this alpha
        def obj_func(pts):
            pts = pts.reshape(N, 2)
            return soft_objective(pts, alpha)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                obj_func, 
                current_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success or res.fun < 1e9: # Check if it didn't crash
                current_points = res.x.reshape(N, 2)
        except Exception:
            pass

    # Final Polish with Nelder-Mead on the raw objective
    # Nelder-Mead handles non-smooth functions well
    if time.time() - start_time < timeout - 10:
        def raw_obj_neg(pts):
            pts = pts.reshape(N, 2)
            s = raw_score(pts)
            return -s # Minimize negative score
        
        try:
            res_nm = minimize(
                raw_obj_neg, 
                current_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.fun < 0: # Ensure valid score
                current_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    return current_points
# EVOLVE-BLOCK-END
