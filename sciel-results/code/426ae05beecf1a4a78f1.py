import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Conversion functions
    def bary_to_cart(v, w):
        # P = v*B + w*C (since A is origin)
        x = v * B[0] + w * C[0]
        y = v * B[1] + w * C[1]
        return np.array([x, y])
        
    def cart_from_bary_params(bary_params):
        # bary_params: (11, 2)
        v = bary_params[:, 0]
        w = bary_params[:, 1]
        x = v * 1.0 + w * 0.5
        y = v * 0.0 + w * (np.sqrt(3)/2)
        return np.column_stack((x, y))
        
    def is_valid_bary(bary_params):
        v = bary_params[:, 0]
        w = bary_params[:, 1]
        return np.all(v >= 0) and np.all(w >= 0) and np.all(v + w <= 1)
        
    def min_triangle_area(pts):
        n = len(pts)
        min_a = float('inf')
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                    if area < min_a:
                        min_a = area
        return min_a
        
    def objective(bary_params):
        if not is_valid_bary(bary_params):
            return 1e6
        pts = cart_from_bary_params(bary_params)
        area = min_triangle_area(pts)
        return -area
        
    best_score = -1e6
    best_points = None
    
    # Run multiple restarts
    max_restarts = 5
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Differential Evolution
        bounds = [(0, 1)] * (2 * N)
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=20, 
                tol=1e-5, 
                seed=42 + i,
                polish=False # We will do our own polish
            )
            
            if time.time() > deadline:
                break
                
            # Local Refinement with Nelder-Mead
            # Flatten the best result
            x0 = res.x
            res2 = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res2.success or res2.fun < res.fun:
                final_bary = res2.x
                if is_valid_bary(final_bary):
                    final_pts = cart_from_bary_params(final_bary)
                    score = -res2.fun
                    if score > best_score:
                        best_score = score
                        best_points = final_pts
        except Exception:
            pass

    if best_points is None:
        # Fallback to random
        pts = []
        for _ in range(N):
            while True:
                v = np.random.random()
                w = np.random.random()
                if v + w <= 1:
                    break
            pts.append([v + 0.5*w, (np.sqrt(3)/2)*w])
        best_points = np.array(pts)

    return best_points
