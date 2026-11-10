import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    # ... setup ...
    
    # Initial guess: Lattice
    # 1 center, 6 ring 1, 4 ring 2 (at 30, 90, 150, 210)
    # Ring 1 dist sqrt(3), angles 0, 60, 120, 180, 240, 300
    # Ring 2 dist 3, angles 30, 90, 150, 210
    
    def get_positions(x):
        # x is flat array of 22 coords (11 hexagons)
        # We fix angles to 0
        coords = x.reshape(11, 2)
        return coords

    def min_side(coords):
        # Compute vertices for each hexagon (angle 0)
        # Hex vertices relative to center:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Note: This is "pointy-top" if angle 0 is vertex.
        # Task says: vertices (cx + s*cos(a + 60i), cy + s*sin(a + 60i)).
        # If a=0, vertex at (1,0). This is "flat-top"? No, vertex at 0 deg.
        # Standard hexagon with vertex at 0.
        
        verts_rel = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Outer hexagon edge normals
        # Vertices of outer hex (side S) at angles 0, 60, ...
        # Edges are between vertices.
        # Normal to edge between 0 and 60 is at 30 deg.
        # Normals at 30, 90, 150, 210, 270, 330.
        normals = np.array([
            [np.cos(np.radians(30)), np.sin(np.radians(30))],
            [np.cos(np.radians(90)), np.sin(np.radians(90))],
            [np.cos(np.radians(150)), np.sin(np.radians(150))],
            [np.cos(np.radians(210)), np.sin(np.radians(210))],
            [np.cos(np.radians(270)), np.sin(np.radians(270))],
            [np.cos(np.radians(330)), np.sin(np.radians(330))]
        ])
        
        max_proj = 0
        for c in coords:
            verts = c + verts_rel
            # Project all vertices on all normals
            # proj = verts @ normals.T
            # max_proj = max(max_proj, max(proj))
            proj = verts @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        # Side S such that S * sqrt(3)/2 = max_proj
        S = max_proj * 2 / np.sqrt(3)
        return S

    def constraints(x):
        coords = get_positions(x)
        cons = []
        # Separation
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(coords[i] - coords[j])
                # Constraint: d >= sqrt(3) + 1e-6
                cons.append(d - (np.sqrt(3) + 1e-6))
        return cons

    # Initial x
    x0 = np.zeros(22)
    # Center
    x0[0:2] = [0, 0]
    # Ring 1
    r1 = np.sqrt(3)
    angles1 = [0, 60, 120, 180, 240, 300]
    for k, ang in enumerate(angles1):
        idx = (k+1)*2
        x0[idx] = r1 * np.cos(np.radians(ang))
        x0[idx+1] = r1 * np.sin(np.radians(ang))
    # Ring 2 (4 hexagons)
    r2 = 3.0
    angles2 = [30, 90, 150, 210]
    for k, ang in enumerate(angles2):
        idx = (k+7)*2
        x0[idx] = r2 * np.cos(np.radians(ang))
        x0[idx+1] = r2 * np.sin(np.radians(ang))
        
    # Optimization
    # Use SLSQP
    # Objective: min_side
    # Constraints: >= 0
    
    # We need to handle the fact that min_side is not differentiable everywhere (max of linear functions).
    # But SLSQP can handle it.
    
    res = minimize(min_side, x0, method='SLSQP', 
                   constraints=[{'type': 'ineq', 'fun': lambda x: constraints(x)}],
                   options={'maxiter': 500, 'ftol': 1e-9})
    
    # Check result
    if res.success:
        coords = get_positions(res.x)
        S = min_side(coords)
        # Construct return values
        inner = np.zeros((11, 3))
        inner[:, :2] = coords
        inner[:, 2] = 0
        outer = np.array([0, 0, 0])
        return inner, outer, float(S)
    else:
        # Fallback to initial
        coords = get_positions(x0)
        S = min_side(coords)
        inner = np.zeros((11, 3))
        inner[:, :2] = coords
        inner[:, 2] = 0
        outer = np.array([0, 0, 0])
        return inner, outer, float(S)
