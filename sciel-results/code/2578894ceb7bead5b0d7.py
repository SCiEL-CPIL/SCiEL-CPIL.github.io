import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    # Initial guess: 7 hexes in a flower, 4 more added
    # Hex grid coords to Cartesian (x, y) for side 1
    # Axial (q, r) to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # Wait, standard hex grid with flat top?
    # Prompt: vertices (cx + s*cos(a + 60i), ...).
    # If a=0, vertex at (s, 0). This is pointy right? No, flat top/bottom?
    # Vertex at 0 deg is (s, 0). Vertex at 60 is (s/2, s*sqrt(3)/2).
    # This is a "pointy-top" hexagon? No, vertex at top?
    # 0 deg is East. 90 deg is North.
    # Vertices at 0, 60, 120, 180, 240, 300.
    # This shape has vertices at E, NE, NW, W, SW, SE.
    # This is a "pointy-east/west" hexagon? Or flat north/south?
    # Edge from 0 to 60. Midpoint 30.
    # Edge from 60 to 120. Midpoint 90 (North).
    # So it has a flat edge at North.
    
    # Let's just place them in a cluster.
    # Center at (0,0).
    # Ring 1: distance sqrt(3) from center?
    # For two unit hexagons touching, center distance is sqrt(3) if flat-to-flat.
    # Let's use the axial grid coordinates for a flat-top hex grid.
    # q, r -> x, y
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # This generates a grid of touching hexagons.
    
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), # Ring 1 (6 hexes)
        # Add 4 more.
        # Let's try (2, -1), (2, -2), (-2, 2), (-2, 1) ?
        # Or just extend the lattice.
        (2, -1),
        (2, -2),
        (-2, 2),
        (-2, 1)
    ]
    
    pts = np.zeros((11, 2))
    for i, (q, r) in enumerate(coords):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        pts[i] = [x, y]
        
    # Center the points
    pts -= np.mean(pts, axis=0)
    
    # Variables: 22 coordinates (11 x, 11 y)
    x0 = pts.flatten()
    
    def get_hex_vertices(cx, cy):
        # Angle 0
        angles = np.deg2rad(np.arange(6) * 60)
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        return np.column_stack((vx, vy))
        
    def cost(params):
        pts = params.reshape(11, 2)
        
        # 1. Compute S (bounding hex side)
        # Normals for outer hex (angle 0):
        # Edges midpoints at 30, 90, ...
        # Normals at 30, 90, ...
        normal_angles = np.deg2rad(np.arange(6) * 60 + 30)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Get all vertices of all 11 hexagons
        all_verts = []
        for i in range(11):
            v = get_hex_vertices(pts[i, 0], pts[i, 1])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Center the vertices to find minimal S
        c = np.mean(all_verts, axis=0)
        rel_verts = all_verts - c
        
        # Project onto normals
        # proj = rel_verts @ normals.T
        # max_proj = np.max(np.abs(proj), axis=0)
        # S_min = max_proj * 2 / np.sqrt(3)
        
        proj = rel_verts @ normals.T
        max_abs_proj = np.max(np.abs(proj))
        S = max_abs_proj * 2 / np.sqrt(3)
        
        # 2. Separation Penalty
        # Check all pairs
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                # Calculate min distance between two hexagons
                # Approximation: distance between centers - sqrt(3)
                # But they might be rotated? No, fixed angle.
                # For fixed angle 0, min center dist for no overlap is sqrt(3).
                # If dist < sqrt(3) - 1e-6, penalty.
                d = np.linalg.norm(pts[i] - pts[j])
                # The "edge normal" check in the prompt is specific.
                # If d < sqrt(3), they might overlap.
                # To be safe, use a soft penalty if d < sqrt(3) - 1e-5.
                if d < np.sqrt(3) - 1e-5:
                    penalty += (np.sqrt(3) - 1e-5 - d) ** 2 * 1000
        return S + penalty
    
    # Optimize
    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_params = res.x
    # If Nelder-Mead fails, try BFGS
    if res.fun > 10: # Arbitrary threshold
         res = minimize(cost, x0, method='BFGS', options={'maxiter': 1000})
         best_params = res.x
         
    pts = best_params.reshape(11, 2)
    
    # Calculate final S and center
    all_verts = []
    for i in range(11):
        v = get_hex_vertices(pts[i, 0], pts[i, 1])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    c = np.mean(all_verts, axis=0)
    
    normal_angles = np.deg2rad(np.arange(6) * 60 + 30)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    rel_verts = all_verts - c
    proj = rel_verts @ normals.T
    max_abs_proj = np.max(np.abs(proj))
    S = max_abs_proj * 2 / np.sqrt(3)
    
    # Construct return values
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = pts[i, 0] - c[0]
        inner[i, 1] = pts[i, 1] - c[1]
        inner[i, 2] = 0.0
        
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, S

