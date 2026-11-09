import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # 1. Define the positions of the 12 inner hexagons in a compact lattice
    # Using a 3-4-3-2 arrangement
    s = np.sqrt(3)
    h = 1.5
    
    positions = []
    # Row 0 (y=0): 3 hexagons
    for x in [-s, 0, s]:
        positions.append([x, 0.0])
        
    # Row 1 (y=h): 4 hexagons
    for x in [-1.5*s, -0.5*s, 0.5*s, 1.5*s]:
        positions.append([x, h])
        
    # Row 2 (y=2h): 3 hexagons
    for x in [-s, 0, s]:
        positions.append([x, 2*h])
        
    # Row 3 (y=3h): 2 hexagons
    for x in [-0.5*s, 0.5*s]:
        positions.append([x, 3*h])
        
    # Ensure we have exactly 12
    assert len(positions) == 12
    
    # 2. Generate vertices for all inner hexagons
    # Each hexagon has side length 1.
    # Vertices are at radius 1 from center.
    all_verts = []
    base_angles = np.arange(6) * np.pi / 3
    for cx, cy in positions:
        verts = np.column_stack([
            cx + np.cos(base_angles),
            cy + np.sin(base_angles)
        ])
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # 3. Find the minimum enclosing regular hexagon
    # Variables: cx, cy, theta (angle of the outer hexagon)
    # The outer hexagon is defined by 6 half-planes.
    # A point p is inside if for all 6 normals n_k, |(p-c).n_k| <= S * sqrt(3)/2
    # So S >= (2/sqrt(3)) * max_k |(p-c).n_k|
    
    def objective(params):
        cx, cy, theta = params
        # Calculate normals for the 6 sides of the outer hexagon
        # The normals are perpendicular to the sides.
        # If the hexagon has a vertex at angle theta, the side normal is at theta + 30 deg?
        # Let's define the hexagon by its "flat" direction or "pointy" direction.
        # Standard: Vertices at angles theta, theta+60, ...
        # Then the side between theta and theta+60 has normal at theta+30.
        # So normals are at theta + 30 + k*60.
        
        normal_angles = theta + np.pi/6 + np.arange(6) * np.pi / 3
        normals = np.column_stack([
            np.cos(normal_angles),
            np.sin(normal_angles)
        ])
        
        # Vectorized calculation of max projection
        # all_verts: (72, 2)
        # normals: (6, 2)
        # proj[i, k] = (all_verts[i] - c) . normals[k]
        
        # Subtract center
        shifted_verts = all_verts - np.array([cx, cy])
        
        # Dot products
        # (72, 1, 2) dot (1, 6, 2) -> (72, 6)
        proj = np.einsum('ij,kj->ik', shifted_verts, normals)
        
        # Max absolute projection
        max_proj = np.max(np.abs(proj))
        
        # Required side length
        return max_proj * 2 / np.sqrt(3)
        
    # Initial guess
    min_x, min_y = np.min(all_verts, axis=0)
    max_x, max_y = np.max(all_verts, axis=0)
    init_c = [(min_x + max_x) / 2, (min_y + max_y) / 2]
    init_theta = 0.0
    
    # Optimize
    # Nelder-Mead is robust for this small problem
    res = minimize(objective, init_c + [init_theta], method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    cx, cy, theta = res.x
    side = objective(res.x)
    
    # Convert theta to degrees
    theta_deg = np.rad2deg(theta)
    
    # Prepare inner hexagons list
    # Format: (x, y, angle_degrees)
    # All inner hexagons are aligned with angle 0 in this lattice
    inner = [[x, y, 0.0] for x, y in positions]
    
    return inner, (cx, cy), side, theta_deg
