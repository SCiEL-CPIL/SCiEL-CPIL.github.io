import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = time.time() + 120.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    a = np.radians(angle_deg)
    base = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return cx + s * R.T @ base.T

def _seg_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-30:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / denom
    t = max(0.0, min(1.0, t))
    return np.linalg.norm(p - (a + t * ab))

def _min_dist_between_hex(cx1, cy1, ang1, cx2, cy2, ang2):
    v1 = _hex_vertices(cx1, cy1, ang1)
    v2 = _hex_vertices(cx2, cy2, ang2)
    d = 1e18
    # Check vertex-segment distances
    for i in range(6):
        for j in range(6):
            d = min(d, _seg_dist(v1[i], v2[j], v2[(j+1)%6]), _seg_dist(v2[j], v1[i], v1[(i+1)%6]))
    # Check if vertices are inside
    # Simple point in polygon for convex hex
    def point_in_hex(px, py, verts):
        inside = True
        for i in range(6):
            a = verts[i]; b = verts[(i+1)%6]
            cross = (b[0]-a[0])*(py-a[1]) - (b[1]-a[1])*(px-a[0])
            if cross < -1e-12:
                inside = False
                break
        return inside
        
    for v in v1:
        if point_in_hex(v[0], v[1], v2):
            return 0.0
    for v in v2:
        if point_in_hex(v[0], v[1], v1):
            return 0.0
    return d

def construct():
    start_time = time.time()
    
    # Parameters: R1, R2, alpha_inner, alpha_outer, th1, th2, th3, th4
    # Initial guess
    R1_init = 1.8
    R2_init = 3.2
    alpha_inner_init = 0.0
    alpha_outer_init = 0.0
    # Angles for the 4 hexes in ring 2. 
    # Symmetric positions for 4 in a 6-fold symmetry context? 
    # Maybe 30, 90, 210, 270? (Midpoints of 4 of the 6 sectors)
    th_init = np.array([30, 90, 210, 270])
    
    x0 = np.array([R1_init, R2_init, alpha_inner_init, alpha_outer_init] + th_init.tolist())
    
    def objective(params):
        if time.time() > DEADLINE - 5:
            return 1e10
        R1, R2, a_in, a_out = params[0], params[1], params[2], params[3]
        ths = params[4:8]
        
        # Generate centers and angles
        centers = []
        angles = []
        
        # Center hex
        centers.append((0.0, 0.0))
        angles.append(a_in)
        
        # Ring 1
        for k in range(6):
            ang = np.radians(60 * k)
            cx = R1 * np.cos(ang)
            cy = R1 * np.sin(ang)
            centers.append((cx, cy))
            angles.append(a_in)
            
        # Ring 2 (4 hexes)
        for i in range(4):
            ang = np.radians(ths[i])
            cx = R2 * np.cos(ang)
            cy = R2 * np.sin(ang)
            centers.append((cx, cy))
            angles.append(a_in)
            
        # Calculate required outer side S
        # Outer hexagon normals
        a_out_rad = np.radians(a_out)
        normals = []
        for k in range(6):
            ang = a_out_rad + np.radians(60 * k)
            # Normal to the face. Face is perpendicular to the radius to the vertex? 
            # For a regular hexagon with vertices at angles 0, 60..., the face between 0 and 60 has normal at 30.
            # Let's define vertices of outer hex to be consistent.
            # If we want to minimize S, we can just project all inner vertices onto the 6 directions.
            # The distance to the side is the projection onto the normal.
            # Normals are at 30, 90, 150, 210, 270, 330 degrees relative to the hex orientation.
            normals.append((np.cos(ang + np.radians(30)), np.sin(ang + np.radians(30))))
            
        max_proj = 0.0
        # Get all inner vertices
        for i in range(11):
            verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
            for v in verts:
                for n in normals:
                    proj = v[0]*n[0] + v[1]*n[1]
                    if proj > max_proj:
                        max_proj = proj
                        
        S = max_proj * 2.0 / np.sqrt(3)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                d = _min_dist_between_hex(centers[i][0], centers[i][1], angles[i], 
                                          centers[j][0], centers[j][1], angles[j])
                if d < 0:
                    penalty += d * d * 1000
                    if penalty > 1e6:
                        return 1e10 # Early exit if too bad
                        
        return S + penalty

    best_val = 1e18
    best_params = None
    
    # Multi-start
    starts = [
        [1.8, 3.2, 0.0, 0.0, 30, 90, 210, 270],
        [1.8, 3.2, 30.0, 0.0, 30, 90, 210, 270],
        [1.7, 3.0, 0.0, 30.0, 30, 90, 210, 270],
        [1.8, 3.5, 0.0, 0.0, 0, 60, 180, 240]
    ]
    
    for x0 in starts:
        if time.time() > DEADLINE - 10:
            break
        try:
            res = minimize(objective, np.array(x0), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback
        return [[0,0,0]] * 11, (0,0), 10.0, 0.0

    R1, R2, a_in, a_out = best_params[0], best_params[1], best_params[2], best_params[3]
    ths = best_params[4:8]
    
    inner = []
    # Center
    inner.append([0.0, 0.0, a_in])
    # Ring 1
    for k in range(6):
        ang = np.radians(60 * k)
        cx = R1 * np.cos(ang)
        cy = R1 * np.sin(ang)
        inner.append([float(cx), float(cy), float(a_in)])
    # Ring 2
    for i in range(4):
        ang = np.radians(ths[i])
        cx = R2 * np.cos(ang)
        cy = R2 * np.sin(ang)
        inner.append([float(cx), float(cy), float(a_in)])
        
    # Calculate final S
    # Re-use objective logic to get clean S without penalty if valid
    # But objective returns S + penalty. If penalty is 0, it's S.
    # We assume best_val is close to S if valid.
    # Let's recompute S exactly for the returned config to be safe.
    
    # Recompute S
    a_out_rad = np.radians(a_out)
    normals = []
    for k in range(6):
        ang = a_out_rad + np.radians(60 * k)
        normals.append((np.cos(ang + np.radians(30)), np.sin(ang + np.radians(30))))
        
    max_proj = 0.0
    for i in range(11):
        verts = _hex_vertices(inner[i][0], inner[i][1], inner[i][2])
        for v in verts:
            for n in normals:
                proj = v[0]*n[0] + v[1]*n[1]
                if proj > max_proj:
                    max_proj = proj
    final_S = max_proj * 2.0 / np.sqrt(3)
    
    return inner, (0.0, 0.0), float(final_S), float(a_out)
