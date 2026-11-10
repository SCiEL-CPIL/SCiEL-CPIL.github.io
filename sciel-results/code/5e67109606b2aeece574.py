import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    # Normals for the outer hexagon (fixed at angle 0, center 0,0)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Normals (perpendicular to edges) point to midpoints of edges
    # Edge 0 is between v0 and v1. Midpoint angle is 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    def get_vertices(center, angle_deg):
        """
        Returns (6, 2) array of vertices for a unit hexagon
        centered at (cx, cy) with rotation angle_deg.
        """
        cx, cy = center
        a = np.deg2rad(angle_deg)
        # Base vertices for angle 0: (1,0), (0.5, sqrt(3)/2), ...
        base_angles = np.deg2rad(np.arange(0, 360, 60))
        verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
        # Rotate
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        verts = verts @ R.T
        # Translate
        verts += center
        return verts

    def calc_s(positions, angles_deg):
        """
        Calculate the minimum side length s of the outer hexagon
        required to contain the given hexagons.
        positions: (N, 2) array of centers
        angles_deg: (N,) array of angles
        """
        max_proj = 0
        for i in range(len(positions)):
            verts = get_vertices(positions[i], angles_deg[i])
            # Project all vertices onto all 6 normals
            # verts: (6, 2), normals: (6, 2)
            # We need max |v . n|
            # v . n for all v and n
            # This is (6, 2) @ (2, 6) -> (6, 6)
            projections = verts @ normals.T
            max_proj = max(max_proj, np.max(np.abs(projections)))
        
        s = (2.0 / np.sqrt(3.0)) * max_proj
        return s

    def generate_topology(type_id):
        """
        Generate initial positions and angles for 12 hexagons.
        Returns (positions, angles)
        """
        positions = []
        angles = [0.0] * 12
        
        if type_id == 0:
            # 3x4 grid
            # Horizontal spacing: sqrt(3)
            # Vertical spacing: 1.5
            # Offset for odd rows: sqrt(3)/2
            dx = np.sqrt(3)
            dy = 1.5
            for r in range(3):
                for c in range(4):
                    x = c * dx
                    y = r * dy
                    if r % 2 == 1:
                        x += dx / 2.0
                    positions.append([x, y])
        elif type_id == 1:
            # 4x3 grid (transposed)
            dx = np.sqrt(3)
            dy = 1.5
            for r in range(4):
                for c in range(3):
                    x = c * dx
                    y = r * dy
                    if r % 2 == 1:
                        x += dx / 2.0
                    positions.append([x, y])
        elif type_id == 2:
            # 1 center, 6 around, 5 more (cluster)
            # Center
            positions.append([0, 0])
            # 6 neighbors
            for i in range(6):
                ang = np.deg2rad(i * 60)
                x = np.sqrt(3) * np.cos(ang)
                y = np.sqrt(3) * np.sin(ang)
                positions.append([x, y])
            # 5 more: place in the "gaps" or extend one side
            # Let's just extend the 3x4 grid but make it more compact?
            # Actually, a "circular" packing of 12 is hard to define analytically.
            # Let's try a 3x4 grid rotated by 30 degrees? No, that's handled by the optimizer.
            # Let's try a different topology: 2 rows of 6?
            # Or a "flower" with 12 petals? No, 1 center + 6 + 5.
            # The 5th ring in honeycomb has 6, 12, 18...
            # 1 + 6 = 7.
            # Next 5 from the 2nd ring.
            # The 2nd ring positions are at distance 2*sqrt(3) or sqrt(3+3+3)?
            # Distance to 2nd ring neighbors is 2*sqrt(3) if straight, or sqrt(3) if diagonal?
            # In honeycomb, distance between centers is sqrt(3).
            # 1st ring: 6 at dist sqrt(3).
            # 2nd ring: 12 at dist 2*sqrt(3) (straight) or sqrt(3)*sqrt(3) = 3?
            # Let's just use the 3x4 grid for now. It's robust.
            pass
            
        # Center the positions at origin for optimization
        positions = np.array(positions)
        centroid = np.mean(positions, axis=0)
        positions -= centroid
        
        return positions, np.array(angles)

    def objective(params, base_pos, base_ang):
        """
        params: [phi, x0, y0]
        base_pos: (12, 2)
        base_ang: (12,)
        """
        phi = params[0]
        x0, y0 = params[1], params[2]
        
        # Rotate base positions
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        rotated_pos = base_pos @ R.T
        rotated_pos += [x0, y0]
        
        # Rotate base angles
        rotated_ang = base_ang + np.degrees(phi)
        
        return calc_s(rotated_pos, rotated_ang)

    best_s = float('inf')
    best_params = None
    best_type = -1
    
    # Try a few topologies
    topologies = [0, 1]
    
    for t in topologies:
        if time.time() > deadline - 10:
            break
            
        base_pos, base_ang = generate_topology(t)
        
        # Initial guess: phi=0, x0=0, y0=0
        x0 = [0, 0, 0]
        
        # Optimize
        try:
            res = minimize(objective, x0, args=(base_pos, base_ang), method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
            
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x
                best_type = t
        except Exception:
            continue
            
        # Also try some random initial rotations to escape local minima
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            x0_rand = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
            try:
                res = minimize(objective, x0_rand, args=(base_pos, base_ang), method='Nelder-Mead', 
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
                if res.fun < best_s:
                    best_s = res.fun
                    best_params = res.x
                    best_type = t
            except Exception:
                continue

    if best_params is None:
        # Fallback to seed
        return np.array([[0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0], [-1.25, 2.17, 0], [1.25, 2.17, 0], 
                         [-1.25, -2.17, 0], [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0], 
                         [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]]), np.array([0, 0, 0]), 8.0

    # Reconstruct the best solution
    base_pos, base_ang = generate_topology(best_type)
    phi, x0, y0 = best_params
    
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    final_pos = base_pos @ R.T
    final_pos += [x0, y0]
    final_ang = base_ang + np.degrees(phi)
    
    # Format output
    inner_data = np.column_stack((final_pos, final_ang))
    outer_data = np.array([0, 0, 0])
    s_val = best_s
    
    return inner_data, outer_data, s_val
