import numpy as np
from scipy.optimize import linprog

def construct():
    # Define the 11 centers of the unit hexagons
    # Using a compact hexagonal cluster configuration
    sqrt3 = np.sqrt(3)
    
    # Axial coordinates (q, r) for the 11 hexagons
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Spikes: (2,0), (0,2), (-2,0), (0,-2)
    centers_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    # Convert axial to Cartesian coordinates
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in centers_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Vertices of a unit regular hexagon (flat-top) relative to its center
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    rel_vertices = [(np.cos(np.pi * k / 3), np.sin(np.pi * k / 3)) for k in range(6)]
    
    # Calculate absolute coordinates of all 66 vertices (11 hexagons * 6 vertices)
    all_vertices = []
    for cx, cy in centers:
        for vx, vy in rel_vertices:
            all_vertices.append((cx + vx, cy + vy))
            
    all_vertices = np.array(all_vertices)
    
    best_S = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_theta = 0.0
    
    # Sweep the rotation of the outer container hexagon
    # The container has 60-degree rotational symmetry.
    # We sweep from 0 to 60 degrees to cover all relative orientations.
    # Using a fine step (0.5 degrees) for precision.
    for theta_deg in np.linspace(0, 60, 121):
        theta = np.radians(theta_deg)
        
        # Normals to the 6 sides of the outer hexagon
        normals = []
        for k in range(6):
            angle = theta + k * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        normals = np.array(normals)
        
        # Linear Programming setup
        # Variables: cx, cy, t
        # t is the apothem (distance from center to side) of the outer hexagon.
        # Side length S = t * 2 / sqrt(3). Minimizing t minimizes S.
        c = np.array([0.0, 0.0, 1.0])
        
        A_ub = []
        b_ub = []
        
        # Constraints: For each vertex p and each normal n,
        # p . n - c . n <= t
        # -nx*cx - ny*cy - t <= -p . n
        for px, py in all_vertices:
            for nx, ny in normals:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: cx, cy are free; t >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                t_val = res.x[2]
                S_val = t_val * 2.0 / sqrt3
                if S_val < best_S:
                    best_S = S_val
                    best_cx = res.x[0]
                    best_cy = res.x[1]
                    best_theta = theta
        except Exception:
            pass
            
    # Construct the return values
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (best_cx, best_cy), best_S, np.degrees(best_theta)
