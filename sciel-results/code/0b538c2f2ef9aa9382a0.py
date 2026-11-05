import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Generate candidate layouts
    layouts = []
    
    # Layout 1: 3x4 Parallelogram
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    pts = []
    for i in range(3):
        for j in range(4):
            pts.append(i * u + j * v)
    layouts.append(np.array(pts))
    
    # Layout 2: 4x3 Parallelogram
    pts = []
    for i in range(4):
        for j in range(3):
            pts.append(i * u + j * v)
    layouts.append(np.array(pts))
    
    # Layout 3: 2x6 Parallelogram
    pts = []
    for i in range(2):
        for j in range(6):
            pts.append(i * u + j * v)
    layouts.append(np.array(pts))
    
    # Layout 4: 6x2 Parallelogram
    pts = []
    for i in range(6):
        for j in range(2):
            pts.append(i * u + j * v)
    layouts.append(np.array(pts))
    
    # Layout 5: "Hexagon" cluster (1+6+5)
    # Center at 0,0
    # Ring 1: 6 neighbors
    # Ring 2: 5 neighbors of ring 1 (leaving one gap to keep it compact? or just add 5)
    # Let's just use a BFS to get 12 closest points in hex grid
    visited = set()
    queue = [(0,0)]
    visited.add((0,0))
    pts = []
    hex_neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    while len(pts) < 12 and queue:
        q, r = queue.pop(0)
        # Convert axial to cartesian
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        pts.append([x, y])
        for dq, dr in hex_neighbors:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                queue.append((nq, nr))
    layouts.append(np.array(pts))
    
    best_score = float('inf')
    best_inner = None
    best_center = None
    best_side = None
    best_angle = None
    
    for pts in layouts:
        if time.time() > deadline - 10:
            break
            
        # Optimize outer hexagon for this layout
        # Inner angles are 0
        # We need to find min enclosing regular hexagon
        
        def cost(params):
            cx, cy, angle = params
            normals = []
            for i in range(6):
                theta = np.deg2rad(30 + i * 60) + angle
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            diffs = pts - np.array([cx, cy])
            projs = diffs @ normals.T
            widths = np.max(projs, axis=0) - np.min(projs, axis=0)
            return np.max(widths)

        # Initial guess
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        
        best_local_S = float('inf')
        best_local_params = None
        
        # Try multiple angles to avoid local minima
        for angle in np.linspace(0, np.pi/3, 31):
            if time.time() > deadline - 5:
                break
            res = minimize(cost, [cx0, cy0, angle], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
            if res.fun < best_local_S:
                best_local_S = res.fun
                best_local_params = res.x
        
        if best_local_params is not None:
            cx, cy, angle = best_local_params
            S = best_local_S / np.sqrt(3)
            
            if S < best_score:
                best_score = S
                best_center = (cx, cy)
                best_side = S
                best_angle = np.rad2deg(angle)
                # Construct inner list
                best_inner = [[p[0], p[1], 0.0] for p in pts]
                
    if best_inner is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle
