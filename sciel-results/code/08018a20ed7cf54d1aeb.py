import numpy as np
import scipy.optimize
import time

N = 11
H = np.sqrt(3) / 2

def min_triangle_area(points):
    # points shape (11, 2)
    # Calculate areas of all C(11,3) = 165 triangles
    # Vectorized calculation for speed
    n = len(points)
    min_area = np.inf
    # Precompute cross products? No, simple loop is fine for 165 triangles
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p0 = points[i]
                p1 = points[j]
                p2 = points[k]
                area = 0.5 * np.abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]))
                if area < min_area:
                    min_area = area
    return min_area

def is_inside_triangle(p):
    # Barycentric coordinates
    l2 = p[1] / H
    l1 = p[0] - 0.5 * l2
    l0 = 1 - l1 - l2
    return min(l0, l1, l2)

def penalty(points):
    pen = 0.0
    for p in points:
        min_l = is_inside_triangle(p)
        if min_l < 0:
            pen += -min_l
    return pen

def objective(params):
    points = params.reshape(-1, 2)
    pen = penalty(points)
    if pen > 1e-8:
        return 1000 * pen
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Bounds for x and y of each point
    # x in [0, 1], y in [0, H]
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))
        bounds.append((0, H))
        
    best_points = None
    best_val = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Try multiple starts
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        if i == 0:
            # Initial guess: Centroid + random points
            points = np.zeros((N, 2))
            points[0] = [0.5, H/3]
            for j in range(1, N):
                # Sample in triangle
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v <= 1:
                        x = v * 1 + (1-u-v)*0.5
                        y = (1-u-v)*H
                        points[j] = [x, y]
                        break
        else:
            # Random points in triangle
            points = np.zeros((N, 2))
            for j in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v <= 1:
                        x = v * 1 + (1-u-v)*0.5
                        y = (1-u-v)*H
                        points[j] = [x, y]
                        break
        
        params = points.flatten()
        
        # Optimize
        try:
            res = scipy.optimize.minimize(
                objective, params, method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_points = res.x.reshape(-1, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback: Grid points
        best_points = np.zeros((N, 2))
        for i in range(N):
            u = (i+1) / (N+1)
            v = 0.5
            # Map to triangle? Just use barycentric
            # This is a rough fallback
            x = 0.5
            y = H/3
            best_points[i] = [x + 0.1*np.sin(i), y + 0.1*np.cos(i)]
            # Ensure inside
            # Simple projection
            for p in best_points:
                l2 = p[1] / H
                l1 = p[0] - 0.5 * l2
                l0 = 1 - l1 - l2
                if l0 < 0 or l1 < 0 or l2 < 0:
                     # Move to center
                     p[0] = 0.5
                     p[1] = H/3

    # Final check: ensure points are inside
    for i in range(N):
        p = best_points[i]
        l2 = p[1] / H
        l1 = p[0] - 0.5 * l2
        l0 = 1 - l1 - l2
        if l0 < 0 or l1 < 0 or l2 < 0:
             # Project to nearest point? Just set to center for safety
             best_points[i] = [0.5, H/3]
             
    return best_points
