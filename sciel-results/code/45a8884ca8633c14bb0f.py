import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate configurations (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (0, -1), (3, 1)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_36)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_hex_vertices(cx, cy, theta):
        # Unit hexagon centered at (cx, cy) with rotation theta
        # Vertices at angles theta + k*pi/3
        angles = theta + np.arange(6) * math.pi / 3.0
        verts = np.column_stack([np.cos(angles), np.sin(angles)])
        return verts + np.array([cx, cy])

    def objective(params):
        # params: [x1, y1, th1, ..., x11, y11, th11, cx, cy, phi]
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        n = 11
        # Extract positions and angles
        pos = params[:3*n].reshape(n, 3)
        cx, cy, phi = params[3*n:3*n+3]
        
        # Calculate Outer Hexagon Side Length S
        # Normals of outer hexagon
        angles_outer = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        
        max_proj = 0.0
        for i in range(n):
            xi, yi, thi = pos[i]
            verts = get_hex_vertices(xi, yi, thi)
            # Project vertices onto normals
            projs = verts @ normals.T # (6, 6)
            # Max absolute projection for each normal
            abs_projs = np.abs(projs)
            max_for_normal = np.max(abs_projs, axis=0) # (6,)
            # The distance from center (cx, cy) to the side is max_proj
            # Wait, the projection is relative to origin. We need relative to (cx, cy).
            # verts are absolute.
            # Let's recompute relative to center
            verts_rel = verts - np.array([cx, cy])
            projs_rel = verts_rel @ normals.T
            abs_projs_rel = np.abs(projs_rel)
            max_for_normal_rel = np.max(abs_projs_rel, axis=0)
            max_proj = max(max_proj, np.max(max_for_normal_rel))
            
        S = (max_proj + APOTHEM) / APOTHEM
        
        # Calculate Overlap Penalty
        penalty = 0.0
        min_dist = math.sqrt(3)
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, _ = pos[i]
                xj, yj, _ = pos[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
                    
        return S + 1000.0 * penalty

    best_global_S = float('inf')
    best_global_config = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Initial guess
        # Positions
        x0 = np.zeros(33)
        for i in range(11):
            x0[3*i] = pts[i, 0]
            x0[3*i+1] = pts[i, 1]
            x0[3*i+2] = 0.0 # Angle
        # Center
        x0[33] = np.mean(pts[:, 0])
        x0[34] = np.mean(pts[:, 1])
        x0[35] = 0.0 # Phi
        
        # Optimize
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 1e10:
                # Check if penalty is small
                # Re-evaluate components
                n = 11
                pos = res.x[:3*n].reshape(n, 3)
                cx, cy, phi = res.x[3*n:3*n+3]
                
                # Recalculate penalty to be sure
                penalty = 0.0
                min_dist = math.sqrt(3)
                for i in range(n):
                    for j in range(i+1, n):
                        xi, yi, _ = pos[i]
                        xj, yj, _ = pos[j]
                        dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                        if dist < min_dist:
                            penalty += (min_dist - dist)**2
                            
                if penalty < 1e-6:
                    # Calculate S
                    angles_outer = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
                    max_proj = 0.0
                    for i in range(n):
                        xi, yi, thi = pos[i]
                        verts = get_hex_vertices(xi, yi, thi)
                        verts_rel = verts - np.array([cx, cy])
                        projs_rel = verts_rel @ normals.T
                        abs_projs_rel = np.abs(projs_rel)
                        max_for_normal_rel = np.max(abs_projs_rel, axis=0)
                        max_proj = max(max_proj, np.max(max_for_normal_rel))
                        
                    S = (max_proj + APOTHEM) / APOTHEM
                    
                    if S < best_global_S:
                        best_global_S = S
                        best_global_config = (pos, cx, cy, phi)
        except Exception:
            pass

    if best_global_config is None:
        # Fallback to first candidate
        pts = axial_to_cartesian(candidates[0])
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        phi = 0.0
        angles_outer = phi + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        max_proj = 0.0
        for i in range(11):
            verts = get_hex_vertices(pts[i, 0], pts[i, 1], 0.0)
            verts_rel = verts - np.array([cx, cy])
            projs_rel = verts_rel @ normals.T
            abs_projs_rel = np.abs(projs_rel)
            max_for_normal_rel = np.max(abs_projs_rel, axis=0)
            max_proj = max(max_proj, np.max(max_for_normal_rel))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        pos = np.zeros((11, 3))
        for i in range(11):
            pos[i, 0] = pts[i, 0]
            pos[i, 1] = pts[i, 1]
        best_global_config = (pos, cx, cy, phi)

    pos, cx, cy, phi = best_global_config
    inner = []
    for i in range(11):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(math.degrees(pos[i, 2]))])
    
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(math.degrees(phi))

    return inner, center, side, rotation
