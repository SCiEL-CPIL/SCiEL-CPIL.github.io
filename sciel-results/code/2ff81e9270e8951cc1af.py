import numpy as np
from scipy.optimize import minimize_scalar

def evaluate_topo(axial_coords):
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    
    base_angle = np.pi / 6.0
    hex_angles = base_angle + np.deg2rad(np.arange(6) * 60)
    unit_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    all_vertices = []
    for c in centers:
        verts = unit_vertices + c
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    n_angles = np.deg2rad([30, 90, 150])
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    def calc_s(theta):
        c_t = np.cos(theta)
        s_t = np.sin(theta)
        rot_mat = np.array([[c_t, -s_t], [s_t, c_t]])
        rotated_verts = all_vertices @ rot_mat.T
        
        widths = []
        for n in normals:
            proj = rotated_verts @ n
            w = np.max(proj) - np.min(proj)
            widths.append(w)
        
        max_w = np.max(widths)
        s = max_w / np.sqrt(3)
        return s

    res = minimize_scalar(calc_s, bounds=(0, np.pi/3), method='bounded')
    theta_opt = res.x
    s_opt = res.fun
    
    # Calculate final positions
    c_t = np.cos(theta_opt)
    s_t = np.sin(theta_opt)
    rot_mat = np.array([[c_t, -s_t], [s_t, c_t]])
    rotated_verts = all_vertices @ rot_mat.T
    
    mids = []
    for n in normals:
        proj = rotated_verts @ n
        mid = (np.max(proj) + np.min(proj)) / 2.0
        mids.append(mid)
    mids = np.array(mids)
    
    A = normals[:2].T
    b = mids[:2]
    C_opt = np.linalg.solve(A, b)
    
    centers_translated = centers @ rot_mat.T - C_opt
    
    angle_deg = np.rad2deg(base_angle + theta_opt)
    inner_data = np.zeros((len(axial_coords), 3))
    for i in range(len(axial_coords)):
        inner_data[i, 0] = centers_translated[i, 0]
        inner_data[i, 1] = centers_translated[i, 1]
        inner_data[i, 2] = angle_deg
        
    outer_data = np.array([0, 0, 0])
    
    return inner_data, outer_data, s_opt

def hexagon_packing_12():
    # Topology 1: 3x4 Rhombus
    topo1 = [(q, r) for q in range(3) for r in range(4)]
    
    # Topology 2: 3-6-3
    topo2 = [
        (0,0), (1,0), (2,0),
        (-1,1), (0,1), (1,1), (2,1), (3,1), (-2,1), # Wait, 6 in middle?
        (0,2), (1,2), (2,2)
    ]
    # Let's check if topo2 is valid (no overlaps).
    # Axial distance check.
    
    # Topology 3: "Hexagon of side 2" minus 7
    # Side 2 hexagon coords:
    # q from -2 to 2, r from -2 to 2, with |q+r| <= 2? No.
    # Standard hex radius 2:
    # q in -2..2
    # r in -2..2
    # |q| <= 2, |r| <= 2, |q+r| <= 2
    # Total 19.
    # Remove 7.
    # Remove corners?
    # Corners: (2,0), (1,1), (0,2), (-2,0), (-1,-1), (0,-2)
    # That's 6. Remove 1 more, say (2,-1)?
    
    best_s = float('inf')
    best_res = None
    
    for topo in [topo1, topo2]:
        # Verify no overlaps (distance >= sqrt(3))
        valid = True
        centers_axial = topo
        for i in range(len(centers_axial)):
            for j in range(i+1, len(centers_axial)):
                q1, r1 = centers_axial[i]
                q2, r2 = centers_axial[j]
                # Axial distance squared
                dq = q1 - q2
                dr = r1 - r2
                # dist^2 = 3 * (dq^2 + dq*dr + dr^2) / 3?
                # Cartesian dist^2 = 3 * (dq^2 + dq*dr + dr^2)
                # We need dist >= sqrt(3), so dist^2 >= 3
                # 3 * (dq^2 + dq*dr + dr^2) >= 3
                # dq^2 + dq*dr + dr^2 >= 1
                val = dq*dq + dq*dr + dr*dr
                if val < 1 - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            res = evaluate_topo(topo)
            if res[2] < best_s:
                best_s = res[2]
                best_res = res
                
    return best_res
