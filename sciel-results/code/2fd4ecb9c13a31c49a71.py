import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate compact cluster of 11 hexagons
    # Axial coordinates (q, r)
    points_axial = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            d = (abs(q) + abs(r) + abs(q+r)) / 2
            points_axial.append((d, q, r))
            
    points_axial.sort(key=lambda x: x[0])
    selected_axial = [ (q, r) for _, q, r in points_axial[:N] ]
    
    # Convert to Cartesian (flat-top, side 1)
    pts = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        pts.append(np.array([x, y]))
    pts = np.array(pts)
    
    # Center the cluster at origin
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    
    # Inner hexagon apothem
    a_in = np.sqrt(3) / 2.0
    
    # 2. Find minimal enclosing hexagon
    # We assume the outer hexagon is aligned with the inner ones (angle 0).
    # Normals for angle 0 (flat-top hexagon has normals at 0, 60, 120, 180, 240, 300 degrees?
    # Flat top: sides are horizontal. Normals are vertical?
    # Standard flat-top hexagon:
    # Vertices at 0, 60, ...
    # Sides are between vertices.
    # Normal to side between 0 and 60 is at 30 degrees.
    # Let's use the standard "pointy top" or "flat top" convention carefully.
    # The code above used:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # This corresponds to pointy-top?
    # If q=1, r=0: x=1.5, y=0.866.
    # If q=0, r=1: x=0, y=1.732.
    # Distance (1,0) to (0,1) is sqrt(2.25 + (0.866-1.732)^2) = sqrt(2.25 + 0.75) = sqrt(3).
    # This is a standard hex grid.
    # The hexagons are "pointy top" if vertices are at top/bottom?
    # Vertex at (0, 1) is top. So yes, pointy top.
    # Normals to sides of a pointy-top hexagon are at 0, 60, 120, 180, 240, 300?
    # Side between (0.5, 0.866) and (0, 1.732)?
    # Let's just use the normals that define the hexagon.
    # For a pointy-top hexagon centered at origin, the 6 supporting lines are:
    # y = +/- h (horizontal)? No, pointy top has flat left/right?
    # Pointy top: Vertices at 90, 150, 210, 270, 330, 30.
    # Sides are vertical? No.
    # Let's stick to the definition:
    # A regular hexagon can be defined by 6 inequalities:
    # n_k . x <= A
    # where n_k are unit vectors at angles theta + k*pi/3.
    
    # If we align inner and outer, we can choose theta=0.
    # Let's define normals for theta=0.
    angles = np.arange(6) * np.pi/3
    V = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Function to compute required A for a given center C
    def get_A(C):
        # C is 2D vector
        # proj[i, k] = n_k . (P_i - C)
        # We need max_{i,k} proj[i,k] + a_in
        # Actually, the condition is n_k . (P_i - C) <= A - a_in
        # So A >= max_{i,k} (n_k . P_i - n_k . C) + a_in
        # Note: The max over k of (max_i (n_k . P_i) - n_k . C)
        
        # P_i is pts_centered
        # n_k . P_i -> proj_centers (6, N)
        proj_centers = V @ pts_centered.T  # (6, N)
        max_proj_per_normal = np.max(proj_centers, axis=1) # (6,)
        
        # n_k . C
        nC = V @ C  # (6,)
        
        # Required A for each normal
        A_req = max_proj_per_normal - nC + a_in
        
        return np.max(A_req)
    
    # Optimize C
    # Initial guess: 0,0 (centroid)
    C0 = np.array([0.0, 0.0])
    
    # Use Nelder-Mead or similar for this small convex problem
    # Or just a grid search since it's small
    # Let's try a small grid search around 0,0
    best_C = C0
    best_A = get_A(C0)
    
    # Grid search
    step = 0.1
    range_val = 1.0
    for dx in np.arange(-range_val, range_val + step, step):
        for dy in np.arange(-range_val, range_val + step, step):
            C_test = np.array([dx, dy])
            A_test = get_A(C_test)
            if A_test < best_A:
                best_A = A_test
                best_C = C_test
                
    # Refine with scipy
    res = minimize(lambda C: get_A(C), best_C, method='Nelder-Mead')
    if res.fun < best_A:
        best_A = res.fun
        best_C = res.x
        
    # Calculate L
    L = 2 * best_A / np.sqrt(3)
    
    # Construct inner list
    # The inner points are pts (original coords)
    # The outer center is centroid + best_C
    outer_center = centroid + best_C
    
    inner = []
    for p in pts:
        inner.append([p[0], p[1], 0.0])
        
    # Check time
    if time.time() > deadline:
        # Return best so far
        pass

    return inner, (outer_center[0], outer_center[1]), L, 0.0
