import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    eps = 1e-7
    
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append((x, y))
        
    r2 = 2.4 + eps
    # Angles 30, 90, 150, 210
    angles2 = [30, 90, 150, 210]
    for ang in angles2:
        angle = np.deg2rad(ang)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append((x, y))

    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])

    # Generate all vertices of inner hexagons
    # Pointy-top hexagon vertices relative to center:
    # (sqrt(3)/2, 0.5), (0, 1), (-sqrt(3)/2, 0.5), (-sqrt(3)/2, -0.5), (0, -1), (sqrt(3)/2, -0.5)
    # Wait, earlier I said 30, 90...
    # 30 deg: (sqrt(3)/2, 0.5)
    # 90 deg: (0, 1)
    # 150 deg: (-sqrt(3)/2, 0.5)
    # 210 deg: (-sqrt(3)/2, -0.5)
    # 270 deg: (0, -1)
    # 330 deg: (sqrt(3)/2, -0.5)
    # This matches.
    
    verts_rel = np.array([
        [np.sqrt(3)/2, 0.5],
        [0, 1],
        [-np.sqrt(3)/2, 0.5],
        [-np.sqrt(3)/2, -0.5],
        [0, -1],
        [np.sqrt(3)/2, -0.5]
    ])
    
    all_verts = []
    for cx, cy, _ in inner:
        for vx, vy in verts_rel:
            all_verts.append([cx + vx, cy + vy])
            
    all_verts = np.array(all_verts)
    
    # Initial guess for outer center: centroid of vertices
    cx0, cy0 = np.mean(all_verts, axis=0)
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e9
        cx, cy, S = params
        if S < 0:
            return 1e9
            
        # Check if all vertices are inside the outer hexagon centered at (cx, cy) with side S
        # Outer hexagon is pointy-top (rot 30)
        # Inequalities for point (x, y) to be in hex centered at (0,0) side S:
        # 1. y <= S
        # 2. -y <= S
        # 3. (2/sqrt(3))*x - y <= S
        # 4. -(2/sqrt(3))*x - y <= S
        # 5. (2/sqrt(3))*x + y <= S
        # 6. -(2/sqrt(3))*x + y <= S
        
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        
        # Coefficients
        c = 2.0 / np.sqrt(3)
        
        v1 = dy - S
        v2 = -dy - S
        v3 = c * dx - dy - S
        v4 = -c * dx - dy - S
        v5 = c * dx + dy - S
        v6 = -c * dx + dy - S
        
        violations = np.maximum(0, np.maximum(np.maximum(v1, v2), np.maximum(v3, np.maximum(v4, np.maximum(v5, v6)))))
        
        # Return S + penalty
        return S + 100 * np.sum(violations**2)

    # Optimize
    x0 = np.array([cx0, cy0, 3.4])
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
    
    if time.time() - start_time > DEADLINE:
        # Fallback to analytic solution
        side = 3.4
        return inner, (0.0, 0.6), side, 30.0
        
    cx_opt, cy_opt, S_opt = res.x
    S_opt += 1e-6 # Safety margin
    
    return inner, (cx_opt, cy_opt), S_opt, 30.0
