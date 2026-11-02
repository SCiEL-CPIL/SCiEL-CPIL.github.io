import numpy as np
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 110  # Leave 10s for safety

def construct():
    start_time = time.time()
    
    # Vertices of the unit-area equilateral triangle
    # Area = 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4
    # Wait, the task says "unit-area triangle".
    # The seed program uses (0,0), (1,0), (0.5, sqrt(3)/2).
    # Area of this triangle is sqrt(3)/4.
    # The evaluator measures "raw value". 
    # "maximize the smallest area of a triangle formed by three points."
    # If the triangle area is sqrt(3)/4, the max min area is some fraction of that.
    # The prompt says "Place 11 points on or inside a unit-area triangle".
    # But the interface says: "construct() -> array of shape (n, 2): n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
    # This specific triangle has area sqrt(3)/4.
    # If the task implies the triangle itself has area 1, then the coordinates should be scaled.
    # However, the interface explicitly gives the coordinates (0,0), (1,0), (0.5, sqrt(3)/2).
    # So we must place points in THIS triangle.
    # The "unit-area" in the text might be loose language or referring to the normalized area.
    # We will maximize the actual geometric area of the sub-triangles.
    
    def bary_to_cart(v, w):
        # v, w are barycentric coords for vertices (1,0) and (0.5, sqrt(3)/2)
        # u = 1 - v - w for vertex (0,0)
        # x = v * 1 + w * 0.5 + u * 0
        # y = v * 0 + w * sqrt(3)/2 + u * 0
        x = v + 0.5 * w
        y = (3 ** 0.5) / 2 * w
        return x, y

    def project_to_simplex(v, w):
        # Ensure v >= 0, w >= 0, v + w <= 1
        v = np.maximum(v, 0)
        w = np.maximum(w, 0)
        # If v + w > 1, we need to scale down or adjust.
        # Simplest projection: if v+w > 1, set v = v/(v+w), w = w/(v+w)
        # This projects radially to the hypotenuse.
        sum_vw = v + w
        mask = sum_vw > 1
        v[mask] = v[mask] / sum_vw[mask]
        w[mask] = w[mask] / sum_vw[mask]
        return v, w

    def compute_min_area(points):
        # points: (N, 2) array
        n = len(points)
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |det(p2-p1, p3-p1)|
                    d1 = p2 - p1
                    d2 = p3 - p1
                    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(z):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            return 0  # Or a large number, but we should stop
        
        # Parse z into v, w arrays
        v = z[0:N]
        w = z[N:2*N]
        
        # Project to simplex
        v, w = project_to_simplex(v, w)
        
        # Convert to Cartesian
        points = np.array([bary_to_cart(vi, wi) for vi, wi in zip(v, w)])
        
        # Compute min area
        min_area = compute_min_area(points)
        
        # We want to maximize min_area, so minimize -min_area
        return -min_area

    # Initial guesses
    best_score = -float('inf')
    best_points = None
    
    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    
    # 1. Random points inside
    # 2. Grid-like points
    # 3. Centroid + corners?
    
    for restart in range(20):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Random initialization in barycentric coords
        # Sample v, w such that v+w <= 1
        v_init = rng.random(N)
        w_init = rng.random(N)
        # Scale to fit in simplex
        # Actually, just sample u,v,w from Dirichlet(1,1,1)
        u_init = rng.random(N)
        v_init = rng.random(N)
        w_init = rng.random(N)
        sum_uvw = u_init + v_init + w_init
        v_init = v_init / sum_uvw
        w_init = w_init / sum_uvw
        
        z_init = np.concatenate([v_init, w_init])
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, z_init, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < -best_score:
                v_opt, w_opt = project_to_simplex(res.x[0:N], res.x[N:2*N])
                points_opt = np.array([bary_to_cart(vi, wi) for vi, wi in zip(v_opt, w_opt)])
                score = compute_min_area(points_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt
                    
        except Exception as e:
            continue

    # If best_points is None, return random
    if best_points is None:
        v_init = np.random.random(N)
        w_init = np.random.random(N)
        sum_uvw = 1 - v_init - w_init
        # Fallback: just use random Cartesian inside
        points = []
        for _ in range(N):
            u, v = np.random.random(), np.random.random()
            if u + v < 1:
                points.append([u + 0.5*v, (3**0.5)/2 * v])
            else:
                points.append([0.5, 0]) # Centroid
        best_points = np.array(points)

    return best_points
