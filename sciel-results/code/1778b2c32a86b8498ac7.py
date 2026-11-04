import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
    min_overlap = float('inf')
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return -min_overlap

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, eps=1e-9):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_fast(verts[i], verts[j], norms[i], norms[j])
            if d < -eps:
                return False
    return True

# --- New Hierarchical Approach ---

# Define a rigid cluster of 4 hexagons.
# Configuration: One center, 3 surrounding at 120 degrees.
# Distance between centers of touching hexagons is sqrt(3).
def _get_cluster_geometry():
    # Hexagon 0 at origin
    # Hexagons 1, 2, 3 at distance sqrt(3)
    d = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (d, 0.0),
        (d * np.cos(np.radians(120)), d * np.sin(np.radians(120))),
        (d * np.cos(np.radians(240)), d * np.sin(np.radians(240)))
    ]
    # Center of mass
    com_x = np.mean([c[0] for c in centers])
    com_y = np.mean([c[1] for c in centers])
    
    # Shift to center of mass
    rel_centers = [(c[0] - com_x, c[1] - com_y) for c in centers]
    
    # Pre-compute vertices and normals for canonical cluster (rotation 0)
    # We will rotate the whole cluster later.
    # For SAT, we need vertices in local coords.
    local_verts = []
    for cx, cy in rel_centers:
        v = _hex_vertices(cx, cy, 1.0, 0.0)
        local_verts.append(v)
        
    # Normals are the same for all hexagons in the cluster if they have same rotation
    local_norms = _hex_edge_normals(0.0)
    
    return local_verts, local_norms, com_x, com_y

# Pre-compute cluster data
CLUSTER_VERTS, CLUSTER_NORMS, CLUSTER_COM_X, CLUSTER_COM_Y = _get_cluster_geometry()

def _objective_hier(p, t0, deadline):
    if time.time() - t0 > deadline:
        return 1e6
    
    # Parameters: 
    # 0-2: Cluster 1 (x, y, rot_deg)
    # 3-5: Cluster 2 (x, y, rot_deg)
    # 6-8: Cluster 3 (x, y, rot_deg)
    # 9-11: Outer Hex (x, y, rot_deg)
    
    c1 = p[0:3]
    c2 = p[3:6]
    c3 = p[6:9]
    oh = p[9:12]
    
    clusters = [c1, c2, c3]
    
    # Calculate global vertices for all 12 hexagons
    all_verts = []
    all_norms = []
    
    for c in clusters:
        cx, cy, rot_deg = c
        rot_rad = np.radians(rot_deg)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        # Rotate local normals
        local_norms_rot = CLUSTER_NORMS @ rot_mat.T
        
        for i in range(4):
            # Rotate local vertices
            v_local = CLUSTER_VERTS[i]
            v_rot = v_local @ rot_mat.T
            # Translate
            v_global = v_rot + np.array([cx, cy])
            
            all_verts.append(v_global)
            all_norms.append(local_norms_rot)
    
    # Check collisions between hexagons from DIFFERENT clusters
    # Indices: 0-3 (C1), 4-7 (C2), 8-11 (C3)
    penalty = 0.0
    # Pairs: (C1, C2), (C1, C3), (C2, C3)
    # C1 vs C2: 4x4 = 16 pairs
    # C1 vs C3: 4x4 = 16 pairs
    # C2 vs C3: 4x4 = 16 pairs
    # Total 48 pairs.
    
    for i in range(4):
        for j in range(4):
            # C1[i] vs C2[j]
            d = _sat_overlap_fast(all_verts[i], all_verts[4+j], all_norms[i], all_norms[4+j])
            if d < 0:
                penalty += (-d) ** 2 * 100.0
            
            # C1[i] vs C3[j]
            d = _sat_overlap_fast(all_verts[i], all_verts[8+j], all_norms[i], all_norms[8+j])
            if d < 0:
                penalty += (-d) ** 2 * 100.0
            
            # C2[i] vs C3[j]
            d = _sat_overlap_fast(all_verts[4+i], all_verts[8+j], all_norms[4+i], all_norms[8+j])
            if d < 0:
                penalty += (-d) ** 2 * 100.0

    # Calculate outer hex side length
    # Flatten all vertices
    pts = np.vstack(all_verts)
    ox, oy, oang = oh
    pts[:, 0] -= ox
    pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(pts, oang)
    
    return L + penalty

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    np.random.seed(42)
    
    best_L = float('inf')
    best_params = None
    
    # Initial configurations for the 3 clusters
    # We want to start with clusters far apart to avoid initial overlap penalties
    # and let the optimizer bring them together.
    
    # Strategy 1: Linear arrangement
    init_configs = []
    
    # Config 1: Clusters spread out
    # Cluster 1 at (-3, 0), Cluster 2 at (3, 0), Cluster 3 at (0, 3)
    init_configs.append(np.array([
        -3, 0, 0,
        3, 0, 0,
        0, 3, 0,
        0, 0, 0
    ]))
    
    # Config 2: Triangular arrangement
    init_configs.append(np.array([
        -3, -2, 0,
        3, -2, 0,
        0, 3, 0,
        0, 0, 0
    ]))
    
    # Config 3: Random
    for _ in range(5):
        p = np.random.randn(12) * 3.0
        p[9] = 0 # Outer center x
        p[10] = 0 # Outer center y
        p[11] = np.random.uniform(0, 60)
        init_configs.append(p)
        
    for cfg in init_configs:
        if time_left() < 5:
            break
            
        try:
            res = minimize(
                _objective_hier,
                cfg,
                args=(t0, DEADLINE - 2),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if res.fun < best_L:
                # Verify disjointness
                # Reconstruct inner hexagons
                inner = []
                for c_idx in range(3):
                    cx, cy, rot_deg = res.x[c_idx*3 : c_idx*3+3]
                    rot_rad = np.radians(rot_deg)
                    cos_r = np.cos(rot_rad)
                    sin_r = np.sin(rot_rad)
                    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                    
                    for i in range(4):
                        v_local = CLUSTER_VERTS[i]
                        # We need the center of the hexagon, not just vertices.
                        # The vertices are relative to the cluster COM.
                        # We need to extract the hexagon center from the local geometry.
                        # In _get_cluster_geometry, we shifted centers to COM.
                        # Let's re-calculate centers for extraction.
                        pass
                        
                # Actually, we can just extract the centers from the construction logic.
                # Let's create a helper to extract inner hexes from params.
                
                def extract_inner(params):
                    inner = []
                    for c_idx in range(3):
                        cx, cy, rot_deg = params[c_idx*3 : c_idx*3+3]
                        rot_rad = np.radians(rot_deg)
                        cos_r = np.cos(rot_rad)
                        sin_r = np.sin(rot_rad)
                        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                        
                        # Recreate relative centers
                        d = np.sqrt(3)
                        centers = [
                            (0.0, 0.0),
                            (d, 0.0),
                            (d * np.cos(np.radians(120)), d * np.sin(np.radians(120))),
                            (d * np.cos(np.radians(240)), d * np.sin(np.radians(240)))
                        ]
                        com_x = np.mean([c[0] for c in centers])
                        com_y = np.mean([c[1] for c in centers])
                        
                        for k in range(4):
                            rx, ry = centers[k][0] - com_x, centers[k][1] - com_y
                            # Rotate
                            nx = rx * cos_r - ry * sin_r
                            ny = rx * sin_r + ry * cos_r
                            # Translate
                            gx = nx + cx
                            gy = ny + cy
                            # Rotation of the hexagon itself is the cluster rotation
                            inner.append([gx, gy, rot_deg])
                    return inner

                inner = extract_inner(res.x)
                if _check_disjoint(inner):
                    ox = res.x[9]
                    oy = res.x[10]
                    oang = res.x[11]
                    
                    all_pts = np.vstack([_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(12)])
                    all_pts[:, 0] -= ox
                    all_pts[:, 1] -= oy
                    L = _outer_hex_side_fast(all_pts, oang)
                    
                    if L < best_L:
                        best_L = L
                        best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback to original grid
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    inner = []
    for i in range(12):
        # Map 12 hexes from 3 clusters
        c_idx = i // 4
        h_idx = i % 4
        cx, cy, rot_deg = best_params[c_idx*3 : c_idx*3+3]
        
        d = np.sqrt(3)
        centers = [
            (0.0, 0.0),
            (d, 0.0),
            (d * np.cos(np.radians(120)), d * np.sin(np.radians(120))),
            (d * np.cos(np.radians(240)), d * np.sin(np.radians(240)))
        ]
        com_x = np.mean([c[0] for c in centers])
        com_y = np.mean([c[1] for c in centers])
        
        rx, ry = centers[h_idx][0] - com_x, centers[h_idx][1] - com_y
        rot_rad = np.radians(rot_deg)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        nx = rx * cos_r - ry * sin_r
        ny = rx * sin_r + ry * cos_r
        gx = nx + cx
        gy = ny + cy
        
        inner.append([float(gx), float(gy), float(rot_deg)])
    
    outer_center = (float(best_params[9]), float(best_params[10]))
    outer_angle = float(best_params[11])
    
    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = 2.0 + gap
    spacing_y = sqrt3 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > 12:
        pts = pts[:12]
    elif len(pts) < 12:
        while len(pts) < 12:
            pts.append([0, 0, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts
