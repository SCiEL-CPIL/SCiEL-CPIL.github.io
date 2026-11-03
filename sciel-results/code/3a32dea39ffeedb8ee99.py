import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, angle_deg):
    angle = np.deg2rad(angle_deg)
    # Unit hexagon vertices relative to center
    # Flat top orientation: vertices at 0, 60, 120...
    # Side length 1
    # Distance from center to vertex is 1 (circumradius = side length for regular hexagon)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def poly_overlap(v1, v2):
    # Separating Axis Theorem for two convex polygons
    # v1, v2 are arrays of shape (6, 2)
    def project(verts, axis):
        proj = np.dot(verts, axis)
        return proj.min(), proj.max()
    
    axes = []
    for i in range(6):
        # Edge vector
        edge = v1[i] - v1[(i+1)%6]
        # Normal
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        edge = v2[i] - v2[(i+1)%6]
        normal = np.array([-edge[1], edge[0]])
        axes.append(normal)
        
    for axis in axes:
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def overlap_penalty(v1, v2):
    # If overlap, return a penalty.
    # For smooth optimization, we can return the minimum distance if not overlapping,
    # but SAT doesn't give distance easily.
    # We can use a simple heuristic: if bounding boxes overlap, and SAT says overlap, return 1.0.
    # To help gradient-free methods, we can return a value based on how "deep" the overlap is.
    # Approximation: calculate the minimum distance between any vertex of v1 and any edge of v2, and vice versa.
    # If they overlap, this distance is negative (or we define it as such).
    
    # Bounding box check
    min1 = v1.min(axis=0)
    max1 = v1.max(axis=0)
    min2 = v2.min(axis=0)
    max2 = v2.max(axis=0)
    
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return 0.0
        
    if not poly_overlap(v1, v2):
        return 0.0
        
    # If overlapping, estimate penetration depth
    # A rough estimate is the minimum distance from any vertex of one to the other polygon
    # If a vertex is inside, distance is 0? No.
    # Let's just return a constant penalty for now, or a small function of the overlap area.
    # For Nelder-Mead, a step function might work but is hard.
    # Let's try to return the minimum distance between the two sets of vertices if they don't overlap,
    # and if they do, return a large number.
    # To make it continuous, we can use the minimum distance between edges.
    
    # Simplified: return 1.0 if overlap.
    return 1.0

def outside_penalty(vertices, L):
    # vertices: (6, 2)
    # L: outer side length
    # Check if all vertices are inside the outer hexagon
    # Outer hexagon center (0,0), rotation 0
    # Inequalities:
    # x <= L
    # x >= -L
    # sqrt(3)*x + y <= L*sqrt(3)
    # sqrt(3)*x - y <= L*sqrt(3)
    # -sqrt(3)*x + y <= L*sqrt(3)
    # -sqrt(3)*x - y <= L*sqrt(3)
    
    x = vertices[:, 0]
    y = vertices[:, 1]
    sqrt3 = np.sqrt(3)
    
    pen = 0.0
    # Squared violations
    pen += np.sum(np.maximum(0, x - L)**2)
    pen += np.sum(np.maximum(0, -x - L)**2)
    pen += np.sum(np.maximum(0, sqrt3*x + y - L*sqrt3)**2)
    pen += np.sum(np.maximum(0, sqrt3*x - y - L*sqrt3)**2)
    pen += np.sum(np.maximum(0, -sqrt3*x + y - L*sqrt3)**2)
    pen += np.sum(np.maximum(0, -sqrt3*x - y - L*sqrt3)**2)
    
    return pen

def objective(params):
    # params: [x1, y1, th1, ..., x11, y11, th11, L]
    L = params[-1]
    if L < 0:
        L = 0.1 # Prevent negative L
    
    centers = params[:-1].reshape(N, 3)
    
    total_pen = 0.0
    
    # Calculate vertices for all hexagons
    verts_list = []
    for i in range(N):
        cx, cy, th = centers[i]
        v = hex_vertices(cx, cy, th)
        verts_list.append(v)
        
        # Outside penalty
        total_pen += outside_penalty(v, L)
        
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            total_pen += overlap_penalty(verts_list[i], verts_list[j])
            
    # Objective: Minimize L + Penalty
    return L + 10.0 * total_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess
    # Place hexagons in a hexagonal lattice
    # Center at 0,0
    # Neighbors at distance sqrt(3)
    sqrt3 = np.sqrt(3)
    
    # Generate initial positions
    # 1 center, 6 ring, 4 more
    init_params = []
    
    # Center
    init_params.extend([0.0, 0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        init_params.extend([x, y, 0.0])
        
    # Ring 2 (partial)
    # We need 4 more.
    # Place them at distance 2*sqrt3? Or fill gaps.
    # Let's just place them in a grid.
    # Grid points:
    # (0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2), (3,0), (0,3)
    # Convert lattice (u,v) to Cartesian:
    # x = sqrt(3) * (u + v/2)
    # y = 1.5 * v
    
    lattice_points = [
        (0,0),
        (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2), (3,0), (0,3)
    ]
    
    # Use the first 11 points from a larger set
    # Generate a grid
    points = []
    for u in range(-3, 4):
        for v in range(-3, 4):
            points.append((u, v))
            
    # Sort by distance from origin to pick the most compact 11
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[0]*p[1])
    
    selected = points[:N]
    
    init_params = []
    for u, v in selected:
        x = sqrt3 * (u + v/2)
        y = 1.5 * v
        init_params.extend([x, y, 0.0])
        
    # Initial L
    # Estimate bounding box
    xs = [p[0] for p in init_params[::3]]
    ys = [p[1] for p in init_params[1::3]]
    max_x = max(abs(min(xs)), abs(max(xs)))
    max_y = max(abs(min(ys)), abs(max(ys)))
    L_init = max(max_x, max_y / np.sqrt(3)) + 1.0
    
    init_params.append(L_init)
    
    best_result = None
    best_score = float('inf')
    
    # Run Nelder-Mead
    # Since the penalty function is non-smooth (step-like for overlap),
    # we might need multiple restarts or a robust method.
    # Let's try Nelder-Mead with a fixed number of iterations.
    
    x0 = np.array(init_params)
    
    # We can try to run the optimization
    # To avoid getting stuck, we can do a few random restarts
    
    num_restarts = 3
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            x0_curr = x0
        else:
            # Add some noise
            x0_curr = x0 + np.random.normal(0, 0.5, x0.shape)
            # Ensure L is positive
            x0_curr[-1] = max(3.0, x0_curr[-1])
            
        try:
            res = minimize(objective, x0_curr, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback to initial
        best_result = x0
        best_score = objective(x0)
        
    # Extract final parameters
    final_params = best_result
    L_final = final_params[-1]
    centers = final_params[:-1].reshape(N, 3)
    
    inner = []
    for i in range(N):
        cx, cy, th = centers[i]
        inner.append([cx, cy, np.rad2deg(th)])
        
    # Return interface: inner, (center_x, center_y), side, angle
    # Outer hexagon center (0,0), side L_final, angle 0
    return inner, (0.0, 0.0), L_final, 0.0

# EVOLVE-BLOCK-END
