import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation theta
    def get_vertices(cx, cy, theta):
        # Unit hexagon vertices relative to center
        # Angles: 0, 60, 120, 180, 240, 300
        angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + theta
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack((xs, ys))
    
    # Objective function
    # params: x0, y0, th0, x1, y1, th1, ..., x10, y10, th10, phi_outer
    def objective(params):
        if time.time() > deadline:
            return 1e10  # Timeout penalty
            
        # Unpack
        inner_params = params[:3*N]
        phi = params[3*N]
        
        # Calculate all vertices of all inner hexagons
        all_vertices = []
        for i in range(N):
            x = inner_params[3*i]
            y = inner_params[3*i+1]
            th = inner_params[3*i+2]
            verts = get_vertices(x, y, th)
            all_vertices.append(verts)
        
        # Concatenate all vertices (6*N points)
        verts = np.vstack(all_vertices) # Shape (6N, 2)
        
        # Outer hexagon normals
        # Outer hexagon centered at (0,0), side S, rotation phi
        # Normals are at angles phi + k*pi/3
        normal_angles = np.array([k*np.pi/3 for k in range(6)]) + phi
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # Shape (6, 2)
        
        # Project all vertices onto all normals
        # proj[i, j] = dot(verts[i], normals[j])
        proj = verts @ normals.T # Shape (6N, 6)
        
        # Max projection
        max_proj = np.max(proj)
        
        # S = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
        S = 2.0 * max_proj / np.sqrt(3)
        
        return S

    # Initial Layout
    # 7 hexagons in a honeycomb
    # 4 more added to the outer ring
    # Honeycomb centers:
    # 0: (0,0)
    # 1: (sqrt(3), 0)
    # 2: (sqrt(3)/2, 1.5)
    # 3: (-sqrt(3)/2, 1.5)
    # 4: (-sqrt(3), 0)
    # 5: (-sqrt(3)/2, -1.5)
    # 6: (sqrt(3)/2, -1.5)
    
    # Add 4 more:
    # Place them at the "corners" of the honeycomb, extending the shape.
    # E.g., at (1.5*sqrt(3), 0), (0, 3), (-1.5*sqrt(3), 0), (0, -3)
    # Or better, attach to the sides of the outer hexagons.
    # Let's just place them in a rough 3x4 grid and let optimizer fix it.
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    th0 = np.zeros(N)
    
    # Grid initialization
    # 3 columns, 4 rows?
    # Hex grid spacing: dx = sqrt(3), dy = 1.5
    # Let's place them in a compact cluster
    coords = [
        (0, 0),
        (np.sqrt(3), 0),
        (-np.sqrt(3), 0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (np.sqrt(3)/2, -1.5),
        (-np.sqrt(3)/2, -1.5),
        # Extra 4
        (1.5*np.sqrt(3), 0),
        (0, 3.0),
        (-1.5*np.sqrt(3), 0),
        (0, -3.0)
    ]
    
    for i in range(N):
        x0[i] = coords[i][0]
        y0[i] = coords[i][1]
        th0[i] = 0.0
        
    phi0 = 0.0
    
    init_params = np.concatenate([np.concatenate([x0, y0, th0]), [phi0]])
    
    # Optimize
    result = None
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() < deadline:
            result = res
    except Exception as e:
        pass
    
    if result is None or result.fun > 1e9:
        # Fallback to initial guess
        best_params = init_params
    else:
        best_params = result.x
        
    # Extract best solution
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        # Convert radians to degrees
        th_deg = np.degrees(th)
        inner.append([x, y, th_deg])
        
    phi = best_params[3*N]
    
    # Calculate final S
    # Re-evaluate to be sure
    # ... (using objective logic)
    # For return, we just need the side length.
    # We can compute it from best_params
    
    # Recompute S for return
    inner_params = best_params[:3*N]
    phi = best_params[3*N]
    
    all_vertices = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        th = inner_params[3*i+2]
        verts = get_vertices(x, y, th)
        all_vertices.append(verts)
        
    verts = np.vstack(all_vertices)
    normal_angles = np.array([k*np.pi/3 for k in range(6)]) + phi
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    proj = verts @ normals.T
    max_proj = np.max(proj)
    S = 2.0 * max_proj / np.sqrt(3)
    
    return inner, (0.0, 0.0), S, np.degrees(phi)
# EVOLVE-BLOCK-END
