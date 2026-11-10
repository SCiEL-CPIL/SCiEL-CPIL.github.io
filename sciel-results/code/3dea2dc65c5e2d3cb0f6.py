import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    sqrt3 = np.sqrt(3)
    
    # Honeycomb lattice
    v1 = np.array([sqrt3, 0])
    v2 = np.array([sqrt3/2, 3/2])
    
    # 12 positions: center + ring1 (6) + 5 from ring2
    positions_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    
    centers = np.array([q*v1 + r*v2 for q, r in positions_axial])
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Precompute all 72 vertices of the 12 inner hexagons
    all_vertices = []
    for i in range(12):
        cx_i, cy_i, _ = inner_hex_data[i]
        angles = np.deg2rad(60 * np.arange(6))
        verts = np.column_stack([cx_i + np.cos(angles), cy_i + np.sin(angles)])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    def objective(params):
        cx, cy, s, theta = params
        if s < np.sqrt(12):
            return 1e10
        
        normals_angles = theta + np.deg2rad(60 * np.arange(6))
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        apothem = s * np.sqrt(3) / 2
        
        diff = all_vertices - np.array([cx, cy])
        dots = diff @ normals.T  # (72, 6)
        violations = np.maximum(dots - apothem, 0)
        if np.any(violations > 0):
            return s + 1000 * np.sum(violations)
        return s
    
    # Try multiple initial guesses
    best_result = None
    best_obj = np.inf
    
    for theta0 in [0, 30, 60, 90, 120, 150]:
        for s0 in [3, 4, 5, 6]:
            cx0, cy0 = 0, 0
            result = minimize(objective, [cx0, cy0, s0, np.deg2rad(theta0)],
                            method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
    
    cx, cy, s, theta = best_result.x
    theta_deg = np.rad2deg(theta)
    
    outer_hex_data = np.array([cx, cy, theta_deg])
    
    return inner_hex_data, outer_hex_data, s
