# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _check_separation(v1, v2, tol=1e-6):
    for verts in [v1, v2]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                continue
            n = n / nn
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
    return False

def _check_inside(v, outer_verts, tol=-1e-6):
    for i in range(len(outer_verts)):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % len(outer_verts)]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < tol:
            return False
    return True

def _is_valid(inner_data, outer_data, s, tol=1e-8):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            if abs(np.linalg.norm(verts[(j+1)%6] - verts[j]) - 1.0) > tol:
                return False
        inner_verts_list.append(verts)
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _check_inside(v, outer_verts):
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    sqrt3 = np.sqrt(3)
    
    best_value = 0
    best_result = None
    
    # Generate honeycomb lattice positions
    # Pointy-top hexagons: center spacing sqrt(3) horizontal, 1.5 vertical
    def gen_hex_grid(rows, cols, offset=0):
        """Generate hex grid positions, rows x cols with optional offset"""
        positions = []
        for r in range(rows):
            for c in range(cols):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                if offset and r % 2 == 0:
                    x += sqrt3 / 2
                positions.append((x, y))
        return positions
    
    # Try multiple arrangements
    arrangements = []
    
    # Arrangement A: 1 center + 6 ring + 5 second ring (hexagonal cluster)
    posA = [(0, 0)]
    for k in range(6):
        ang = np.radians(60 * k)
        posA.append((sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)))
    # Second ring: 12 positions at distance 2*sqrt(3)
    for k in range(6):
        ang = np.radians(60 * k)
        posA.append((2*sqrt3 * np.cos(ang), 2*sqrt3 * np.sin(ang)))
    # We have 13, remove one to get 12
    posA = posA[:12]
    arrangements.append(posA)
    
    # Arrangement B: 3-4-3-2
    posB = []
    for i in range(3):
        posB.append((i * sqrt3, 0))
    for i in range(4):
        posB.append((i * sqrt3 - sqrt3/2, 1.5))
    for i in range(3):
        posB.append((i * sqrt3, 3.0))
    for i in range(2):
        posB.append((i * sqrt3 + sqrt3/2, 4.5))
    arrangements.append(posB)
    
    # Arrangement C: 2-3-4-3
    posC = []
    for i in range(2):
        posC.append((i * sqrt3, 0))
    for i in range(3):
        posC.append((i * sqrt3 - sqrt3/2, 1.5))
    for i in range(4):
        posC.append((i * sqrt3, 3.0))
    for i in range(3):
        posC.append((i * sqrt3 - sqrt3/2, 4.5))
    arrangements.append(posC)
    
    # Arrangement D: 4-4-4
    posD = []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3/2
            posD.append((x, y))
    arrangements.append(posD)
    
    # Arrangement E: 3-3-3-3
    posE = []
    for r in range(4):
        for c in range(3):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3/2
            posE.append((x, y))
    arrangements.append(posE)
    
    # Arrangement F: 2-4-4-2
    posF = []
    for i in range(2):
        posF.append((i * sqrt3, 0))
    for i in range(4):
        posF.append((i * sqrt3 - sqrt3/2, 1.5))
    for i in range(4):
        posF.append((i * sqrt3, 3.0))
    for i in range(2):
        posF.append((i * sqrt3 + sqrt3/2, 4.5))
    arrangements.append(posF)
    
    for arr in arrangements:
        if len(arr) != 12:
            continue
        if time.time() > deadline - 60:
            break
        
        # Center the arrangement
        arr = np.array(arr, dtype=float)
        arr[:, 0] -= arr[:, 0].mean()
        arr[:, 1] -= arr[:, 1].mean()
        
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i] = [arr[i, 0], arr[i, 1], 0]
        
        # Find min enclosing regular hexagon
        # For outer hex with center (ox,oy), angle a, side s:
        # inradius = s * sqrt(3)/2
        # Point (x,y) inside iff for all 6 normals: dot((x,y)-(ox,oy), n_k) <= inradius
        # So s >= max over all inner vertices and all 6 normals of dot(v - c, n_k) / (sqrt(3)/2)
        
        def find_min_s(ox, oy, oang):
            inner_verts = []
            for i in range(12):
                cx, cy, ang = inner[i]
                inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))
            all_verts = np.vstack(inner_verts)
            rel = all_verts - np.array([ox, oy])
            
            oang_rad = np.radians(oang)
            max_proj = 0
            for k in range(6):
                n_ang = oang_rad + np.radians(60 * k + 30)
                n = np.array([np.cos(n_ang), np.sin(n_ang)])
                proj = rel @ n
                max_proj = max(max_proj, proj.max())
            
            return max_proj / (sqrt3 / 2)
        
        # Optimize center and angle
        def objective(params):
            ox, oy, oang = params
            return find_min_s(ox, oy, oang)
        
        x0 = [0.0, 0.0, 0.0]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7})
            ox, oy, oang = res.x
            s = find_min_s(ox, oy, oang)
        except:
            ox, oy, oang = 0.0, 0.0, 0.0
            s = find_min_s(ox, oy, oang)
        
        outer = np.array([ox, oy, oang])
        
        # Validate
        if _is_valid(inner, outer, s):
            value = 1.0 / s
            if value > best_value:
                best_value = value
                best_result = (inner.copy(), outer.copy(), s)
    
    if best_result is not None:
        return best_result
    
    # Fallback
    inner_hex_data = np.array(
        [
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
            [0, -4, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
