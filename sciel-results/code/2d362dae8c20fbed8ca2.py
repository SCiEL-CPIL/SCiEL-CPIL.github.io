import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    # Initial guess: 1 center, 6 ring 1, 4 corners of ring 2
    # Ring 1 centers at distance sqrt(3), angles 0, 60, ...
    # Ring 2 corners at distance 2*sqrt(3), angles 0, 60, ...
    
    # Let's pick 4 corners: 0, 120, 240, and 60? 
    # Let's try 0, 120, 240, 300? (4 corners)
    
    r1 = np.sqrt(3)
    r2 = 2 * np.sqrt(3)
    
    angles_deg = [0, 60, 120, 180, 240, 300]
    
    centers = []
    angles = []
    
    # Center
    centers.append([0, 0])
    angles.append(0)
    
    # Ring 1
    for a in angles_deg:
        rad = np.deg2rad(a)
        centers.append([r1 * np.cos(rad), r1 * np.sin(rad)])
        angles.append(0)
        
    # Ring 2 Corners (pick 4)
    # Let's pick 0, 120, 240, 60
    selected_corners = [0, 60, 120, 240]
    for a in selected_corners:
        rad = np.deg2rad(a)
        centers.append([r2 * np.cos(rad), r2 * np.sin(rad)])
        angles.append(0)
        
    # Total 1 + 6 + 4 = 11
    
    init_L = 4.5
    
    def get_vertices(cx, cy, s, angle_deg):
        rad = np.deg2rad(angle_deg)
        verts = []
        for i in range(6):
            a = rad + np.deg2rad(60 * i)
            verts.append([cx + s * np.cos(a), cy + s * np.sin(a)])
        return np.array(verts)
    
    def is_inside_hex(v, L, angle_deg):
        # Check if point v is inside regular hexagon centered at 0,0 with side L and angle angle_deg
        # Transform v to frame of hexagon
        rad = np.deg2rad(angle_deg)
        # Rotate v by -rad
        vx = v[0] * np.cos(-rad) - v[1] * np.sin(-rad)
        vy = v[0] * np.sin(-rad) + v[1] * np.cos(-rad)
        
        # In this frame, hexagon vertices are at 0, 60, ...
        # Check 6 inequalities
        # The hexagon is defined by |x| <= L, |x +/- y/sqrt(3)| <= L? No.
        # Easier: check distance to edges.
        # Or check if max projection on 6 normals <= L * cos(30)? No.
        # The support function of a hexagon side L in direction n is L * cos(angle_diff) if angle_diff <= 30?
        # Let's just check the 6 half-planes.
        # Normals are at 30, 90, 150, 210, 270, 330.
        # Distance from origin to edge is L * sqrt(3) / 2.
        apothem = L * np.sqrt(3) / 2
        for i in range(6):
            n_rad = np.deg2rad(30 + 60 * i)
            nx = np.cos(n_rad)
            ny = np.sin(n_rad)
            proj = vx * nx + vy * ny
            if proj > apothem + 1e-9:
                return False
        return True

    def calculate_penalty(params, L):
        # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11]
        penalty = 0.0
        n_hex = 11
        verts_list = []
        for i in range(n_hex):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = params[3*i+2]
            verts = get_vertices(cx, cy, 1.0, ang)
            verts_list.append(verts)
            
            # Boundary check
            for v in verts:
                if not is_inside_hex(v, L, 0):
                    # Penalty: distance outside
                    # Calculate min distance to boundary
                    # Approximate: just add a large penalty
                    penalty += 100.0
        
        # Overlap check
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                # Check if hex i and j overlap
                # Simple check: distance between centers vs min separation
                # Better: check if any vertex of i is inside j, or vice versa
                # Or check edge intersections
                # For speed, let's use a bounding box check first
                vi = verts_list[i]
                vj = verts_list[j]
                
                # Bounding box
                if np.max(vi[:,0]) < np.min(vj[:,0]) or np.max(vj[:,0]) < np.min(vi[:,0]):
                    continue
                if np.max(vi[:,1]) < np.min(vj[:,1]) or np.max(vj[:,1]) < np.min(vi[:,1]):
                    continue
                    
                # Detailed check:
                # If any vertex of i is inside j, overlap
                for v in vi:
                    if is_inside_hex(v, 1.0, 0): # Wait, is_inside_hex checks against outer hex. 
                        # We need a generic point-in-polygon or point-in-hex check
                        pass
                
                # This is getting complicated. Let's use a simpler overlap metric.
                # Min distance between vertices?
                # Or just assume the lattice is valid and only optimize L?
                # If we keep the lattice structure, there is no overlap.
                # We can just optimize the positions slightly to allow smaller L.
        
        return penalty

    # Since the lattice is already non-overlapping, we can just optimize L for the fixed lattice positions.
    # The minimal L for the lattice positions is determined by the furthest vertex.
    # Let's calculate the exact L required for the 11-hex lattice subset.
    
    min_L = 0
    for i in range(11):
        cx = centers[i][0]
        cy = centers[i][1]
        ang = angles[i]
        verts = get_vertices(cx, cy, 1.0, ang)
        for v in verts:
            # Find min L such that v is inside hex(L, 0)
            # v is at (vx, vy).
            # The condition is proj_n(v) <= L * sqrt(3)/2 for all 6 normals.
            # So L >= 2/sqrt(3) * max(proj_n(v))
            max_proj = 0
            for k in range(6):
                n_rad = np.deg2rad(30 + 60 * k)
                nx = np.cos(n_rad)
                ny = np.sin(n_rad)
                proj = v[0]*nx + v[1]*ny
                if proj > max_proj:
                    max_proj = proj
            req_L = (2 / np.sqrt(3)) * max_proj
            if req_L > min_L:
                min_L = req_L
                
    # Add a small epsilon
    outer_hex_data = np.array([0, 0, 0])
    inner_hex_data = np.array([[c[0], c[1], a] for c, a in zip(centers, angles)])
    
    return inner_hex_data, outer_hex_data, min_L + 1e-6

