import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 115.0  # 120s limit, leave 5s margin

def construct():
    start_time = time.time()
    
    # Parameters:
    # r1: radius of inner ring (5 hexagons)
    # r2: radius of outer ring (6 hexagons)
    # theta1: angular offset of inner ring
    # theta2: angular offset of outer ring
    # alpha: orientation of all inner hexagons
    # S: side length of outer hexagon
    # beta: orientation of outer hexagon
    
    # Initial guess
    # Center at 0
    # Inner 5 at radius ~1.7
    # Outer 6 at radius ~3.0
    # S ~ 4.0
    x0 = [1.7, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0]
    
    def objective(params):
        r1, r2, theta1, theta2, alpha, S, beta = params
        
        # Ensure positive radii and side length
        if r1 < 0.1 or r2 < 0.1 or S < 1.0:
            return 1e6
            
        # Generate centers
        centers = np.zeros((N, 2))
        
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # Inner ring (5 hexagons)
        for i in range(5):
            angle = i * 2 * np.pi / 5 + theta1
            centers[i + 1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
            
        # Outer ring (6 hexagons)
        for j in range(6):
            angle = j * 2 * np.pi / 6 + theta2
            centers[j + 6] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
        penalty = 0.0
        
        # 1. Check Overlaps between inner hexagons
        # Since all inner hexagons have the same orientation 'alpha',
        # the minimum distance for non-overlap is sqrt(3).
        min_dist_sq = 3.0 # (sqrt(3))^2
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    # Strong penalty
                    penalty += (min_dist_sq - dist_sq) * 1000.0
        
        # 2. Check Containment
        # Outer hexagon center at (0,0)
        # Normals for outer hexagon
        angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + beta
        normals = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
        
        # Distance from center to side of outer hexagon (apothem)
        d_side = S * np.sqrt(3) / 2
        
        # Inner hexagon vertices relative to its center
        # Orientation is 'alpha'
        angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + alpha
        verts_in = np.column_stack((np.cos(angles_in), np.sin(angles_in)))
        
        for i in range(N):
            cx, cy = centers[i]
            # Vertices of this inner hexagon in global coords
            verts = verts_in + np.array([cx, cy])
            
            # For each vertex v, check if it is inside the outer hexagon.
            # Inside condition: dot(v, n) <= d_side for all normals n.
            dots = verts @ normals.T
            max_dots = np.max(dots, axis=1) # Max projection for each vertex
            
            viol = np.maximum(0, max_dots - d_side)
            penalty += np.sum(viol**2) * 100.0
            
        return S + penalty

    best_params = x0
    best_score = objective(x0)
    
    # Optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # Try a second run with slightly different initial conditions if time allows
    if time.time() - start_time < DEADLINE - 20:
        x0_2 = [1.8, 3.2, np.pi/5, np.pi/6, 0.0, 4.2, 0.0]
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    r1, r2, theta1, theta2, alpha, S, beta = best_params
    
    # Construct inner list
    inner = []
    
    # Center
    inner.append([0.0, 0.0, float(np.degrees(alpha))])
    
    # Inner ring
    for i in range(5):
        angle = i * 2 * np.pi / 5 + theta1
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        inner.append([float(x), float(y), float(np.degrees(alpha))])
        
    # Outer ring
    for j in range(6):
        angle = j * 2 * np.pi / 6 + theta2
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        inner.append([float(x), float(y), float(np.degrees(alpha))])
        
    # Outer hexagon center is (0,0)
    return inner, (0.0, 0.0), float(S), float(np.degrees(beta))
