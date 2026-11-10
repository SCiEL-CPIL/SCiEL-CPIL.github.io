import numpy as np
import time
from scipy.optimize import linprog

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0 # 20s safety margin
    
    # Helper: Get vertices of a unit hexagon at (cx, cy) with angle 0
    def get_hex_vertices(cx, cy, angle_deg=0):
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + np.deg2rad(60 * i)
            x = cx + np.cos(theta)
            y = cy + np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    # Helper: Compute minimal enclosing regular hexagon side s for a set of points
    # points: (N, 2) array
    def min_enclosing_hex_side(points):
        min_s = float('inf')
        best_center = None
        best_angle = 0
        
        # We optimize over angle 0 to 60 degrees
        # Step size 0.1 deg is usually sufficient for high precision
        angles = np.arange(0, 60.01, 0.1)
        
        for theta_deg in angles:
            if time.time() > deadline:
                break
            theta_rad = np.deg2rad(theta_deg)
            
            # Normals of the 6 sides of the outer hexagon
            # The sides are perpendicular to the normals.
            # Normal i points to the middle of side i.
            # For a hexagon with angle 0 (flat top?), let's define angle as rotation of the hexagon.
            # Vertices are at angle + 60i.
            # Side i is between vertex i and i+1.
            # Normal to side i is at angle + 60i + 30?
            # Let's check: Vertex 0 at 0 deg. Vertex 1 at 60 deg.
            # Side 0 is between (1,0) and (0.5, 0.866). Midpoint angle is 30 deg.
            # So normal 0 is at 30 deg.
            # Normal i is at theta + 30 + 60i.
            
            normals = []
            for i in range(6):
                ang = theta_rad + np.deg2rad(30 + 60 * i)
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)
            
            # Project points onto normals
            # projections: (6, N)
            projections = points @ normals.T
            
            L = np.min(projections, axis=1)
            R = np.max(projections, axis=1)
            
            # Constraints for LP:
            # Minimize a
            # Variables: Cx, Cy, a
            # c = [0, 0, 1]
            c = np.array([0, 0, 1])
            
            A_ub = []
            b_ub = []
            
            for i in range(6):
                n_i = normals[i]
                # C dot n_i - a <= L_i
                # Cx * n_i[0] + Cy * n_i[1] - 1 * a <= L_i
                A_ub.append([n_i[0], n_i[1], -1])
                b_ub.append(L[i])
                
                # C dot n_i + a >= R_i  => -C dot n_i - a <= -R_i
                A_ub.append([-n_i[0], -n_i[1], -1])
                b_ub.append(-R[i])
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # a >= 0 is handled by bounds
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    a_opt = res.x[2]
                    s_opt = 2 * a_opt / np.sqrt(3)
                    if s_opt < min_s:
                        min_s = s_opt
                        best_center = [res.x[0], res.x[1]]
                        best_angle = theta_deg
            except Exception:
                pass
                
        return min_s, best_center, best_angle

    # Generate candidate inner configurations
    # Axial to Cartesian: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return [x, y]
    
    configs = []
    
    # Config 1: 7-hex center + 5 outer
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extra 5: (2,0), (1,1), (0,2), (-1,2), (2,-1) ? Let's pick a symmetric set.
    # Let's use a 3x4 block instead, it's more compact in some orientations.
    
    # Config 2: 3 rows of 4
    # Row 0 (r=0): q = 0, 1, 2, 3
    # Row 1 (r=1): q = 0, 1, 2, 3
    # Row 2 (r=2): q = 0, 1, 2, 3
    pts_3x4 = []
    for r in range(3):
        for q in range(4):
            pts_3x4.append(axial_to_cartesian(q, r))
    configs.append(np.array(pts_3x4))
    
    # Config 3: 2 rows of 6
    pts_2x6 = []
    for r in range(2):
        for q in range(6):
            pts_2x6.append(axial_to_cartesian(q, r))
    configs.append(np.array(pts_2x6))
    
    # Config 4: Hexagonal cluster (1+6+5)
    # Center
    # 6 neighbors
    # 5 more attached to the "equator"
    pts_cluster = []
    pts_cluster.append([0,0])
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for n in neighbors:
        pts_cluster.append(axial_to_cartesian(n[0], n[1]))
    # Add 5 more to make it 12.
    # Attach to (1,0), (0,1), (-1,1), (-1,0), (0,-1)
    extras = [(2,0), (1,1), (-2,1), (-2,0), (1,-1)]
    for e in extras:
        pts_cluster.append(axial_to_cartesian(e[0], e[1]))
    configs.append(np.array(pts_cluster))
    
    # Config 5: Diamond shape
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1)
    # (1,1), (-1,1), (1,-1), (-1,-1)
    # (2,0), (-2,0), (0,2), (0,-2)
    # Total 1+4+4+4 = 13. Remove one?
    # Let's just use the 3x4 and 2x6 as they are very standard.
    
    best_result = None
    best_s = float('inf')
    
    for centers in configs:
        if time.time() > deadline:
            break
        
        # Get all vertices of all inner hexagons
        all_vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, 0)
            all_vertices.extend(verts)
        all_vertices = np.array(all_vertices)
        
        s, center, angle = min_enclosing_hex_side(all_vertices)
        
        if s < best_s:
            best_s = s
            best_result = {
                'centers': centers,
                'outer_center': center,
                'outer_angle': angle,
                's': s
            }
            
    if best_result is None:
        # Fallback to seed
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, 8.0

    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(best_result['centers']):
        inner_hex_data[i] = [cx, cy, 0]
        
    outer_hex_data = np.array([best_result['outer_center'][0], best_result['outer_center'][1], best_result['outer_angle']])
    
    return inner_hex_data, outer_hex_data, best_result['s']

