# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _sat_overlap(verts1, normals1, verts2, normals2):
    all_normals = np.vstack([normals1, normals2])
    min_overlap = float('inf')
    best_axis = None
    best_sign = 0
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-6:
            return 0.0, None, 0
        elif max2 < min1 - 1e-6:
            return 0.0, None, 0
        
        if max1 < min2:
            overlap = min2 - max1
            sign = 1
        elif max2 < min1:
            overlap = min1 - max2
            sign = -1
        else:
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0, None, 0
            c1 = (min1 + max1) / 2
            c2 = (min2 + max2) / 2
            sign = 1 if c1 > c2 else -1
        
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = normal
            best_sign = sign
    
    return min_overlap, best_axis, best_sign

def _compute_min_s_for_cluster(cluster_verts_flat, outer_angle):
    """
    Given pre-computed cluster vertices (centered at origin), compute minimum outer hexagon side.
    The outer hexagon has support function h(n) = s * cos(30°) = s * sqrt(3)/2
    for each edge normal n. So s = max_proj / (sqrt(3)/2) = max_proj * 2/sqrt(3).
    """
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(cluster_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    s = max_proj * 2.0 / np.sqrt(3)
    return max(s, np.sqrt(12))

def _generate_cluster_patterns():
    """
    Generate several compact cluster patterns of 12 unit hexagons.
    Each pattern is a (12, 2) array of center positions relative to cluster centroid.
    Hexagons are assumed to have angle 0 (flat-top orientation).
    Returns list of (12, 2) arrays.
    """
    patterns = []
    
    # Pattern 1: Hexagonal close packing - center + 6 ring + 5 more
    # In hex grid, adjacent centers are distance 2*sqrt(3) apart (for unit hexagons touching)
    L = 2.0 * np.sqrt(3)
    
    # Hexagonal lattice basis
    # e1 = (L, 0), e2 = (L/2, L*sqrt(3)/2)
    # Axial coordinates (q, r) -> position = q*e1 + r*e2
    
    def axial_to_xy(q, r):
        return np.array([L * (q + r/2.0), L * r * np.sqrt(3)/2.0])
    
    # Pattern 1: Compact hexagonal shape (radius 2 in hex grid, but only 12 cells)
    # Use a 3-4-3-2 arrangement or similar
    # Let's try: row 0: 4, row 1: 4, row 2: 3, row 3: 1
    # Actually let's use known compact shapes
    
    # Pattern A: 4-4-4 grid (triangular lattice)
    pts = []
    for row in range(3):
        for col in range(4):
            pos = axial_to_xy(col, row)
            pts.append(pos)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern B: Hexagonal cluster - center + 6 neighbors + 5 more from second ring
    # First ring: 6 hexagons around center
    # Second ring: pick 5 that make it compact
    pts = [axial_to_xy(0, 0)]  # center
    # First ring
    for k in range(6):
        angle = np.radians(60 * k)
        q, r = int(round(L * np.cos(angle) / L)), 0
        # Use axial coords for first ring
        first_ring_axial = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        pts.append(axial_to_xy(*first_ring_axial[k]))
    # Second ring - pick 5 to make compact shape
    second_ring_candidates = [
        (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    ]
    # Pick 5 that are "closest" to complete a hexagon
    # For a full hex of radius 2, we need all 12 in second ring. Pick 5 that minimize bounding box
    # Let's pick: (2,0), (1,1), (0,2), (-1,1), (-2,0) - top arc
    selected_second = [(2,0), (1,1), (0,2), (-1,1), (-2,0)]
    for q, r in selected_second:
        pts.append(axial_to_xy(q, r))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern C: Different 5 from second ring - bottom arc
    pts = [axial_to_xy(0, 0)]
    for k in range(6):
        first_ring_axial = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        pts.append(axial_to_xy(*first_ring_axial[k]))
    selected_second = [(2,-1), (1,-2), (0,-2), (-1,-1), (-2,0)]
    for q, r in selected_second:
        pts.append(axial_to_xy(q, r))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern D: 6-6 two rows
    pts = []
    for col in range(6):
        pts.append(axial_to_xy(col, 0))
    for col in range(6):
        pts.append(axial_to_xy(col, 1))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern E: 5-4-3 triangular
    pts = []
    for row in range(3):
        for col in range(5 - row):
            pts.append(axial_to_xy(col, row))
    # Need 12, so add one more
    pts.append(axial_to_xy(5, 0))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern F: 4-5-3 arrangement
    pts = []
    for col in range(4):
        pts.append(axial_to_xy(col, 0))
    for col in range(5):
        pts.append(axial_to_xy(col, 1))
    for col in range(3):
        pts.append(axial_to_xy(col, 2))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern G: 3-4-4-1 
    pts = []
    for col in range(3):
        pts.append(axial_to_xy(col, 0))
    for col in range(4):
        pts.append(axial_to_xy(col, 1))
    for col in range(4):
        pts.append(axial_to_xy(col, 2))
    pts.append(axial_to_xy(1, 3))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern H: Diamond shape
    pts = []
    # Row 0: 1
    pts.append(axial_to_xy(0, 0))
    # Row 1: 3
    for col in range(-1, 2):
        pts.append(axial_to_xy(col, 1))
    # Row 2: 5
    for col in range(-2, 3):
        pts.append(axial_to_xy(col, 2))
    # Row 3: 3
    for col in range(-1, 2):
        pts.append(axial_to_xy(col, 3))
    # Row 4: 1
    pts.append(axial_to_xy(0, 4))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern I: 2-4-4-2
    pts = []
    for col in range(2):
        pts.append(axial_to_xy(col, 0))
    for col in range(4):
        pts.append(axial_to_xy(col, 1))
    for col in range(4):
        pts.append(axial_to_xy(col, 2))
    for col in range(2):
        pts.append(axial_to_xy(col, 3))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    # Pattern J: 1-3-5-3 (incomplete diamond)
    pts = []
    pts.append(axial_to_xy(0, 0))
    for col in range(-1, 2):
        pts.append(axial_to_xy(col, 1))
    for col in range(-2, 3):
        pts.append(axial_to_xy(col, 2))
    for col in range(-1, 2):
        pts.append(axial_to_xy(col, 3))
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    patterns.append(pts)
    
    return patterns

def _check_cluster_validity(cluster_centers):
    """
    Check if a cluster of unit hexagons (all with angle 0) has no overlaps.
    Two unit hexagons with centers distance d apart don't overlap if d >= 2*sqrt(3).
    """
    n = len(cluster_centers)
    min_dist = 2.0 * np.sqrt(3) - 1e-9
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            if dist < min_dist:
                return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 615.0
    
    # Generate cluster patterns
    patterns = _generate_cluster_patterns()
    
    # Filter valid patterns (no internal overlaps)
    valid_patterns = []
    for p in patterns:
        if _check_cluster_validity(p):
            valid_patterns.append(p)
    
    if not valid_patterns:
        # Fallback: simple grid
        L = 2.0 * np.sqrt(3)
        pts = []
        for row in range(3):
            for col in range(4):
                pts.append([L * (col + row/2.0), L * row * np.sqrt(3)/2.0])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        valid_patterns = [pts]
    
    best_s = float('inf')
    best_cluster = None
    best_rotation = 0.0
    best_outer_angle = 0.0
    
    # For each valid pattern, optimize rotation and outer angle
    for pattern in valid_patterns:
        # Precompute: for a given rotation theta, the cluster vertices are rotated
        # We need to find theta and outer_angle that minimize s
        
        def objective(params):
            theta = params[0]
            outer_angle = params[1]
            
            # Rotate cluster centers
            c, s_ = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s_], [s_, c]])
            rotated_centers = pattern @ R.T
            
            # Compute all vertices (hexagons with angle 0, rotated by theta)
            # Each hexagon has 6 vertices. With rotation theta, vertex angles are theta + 60*k
            # For efficiency, compute relative to each center
            hex_angles = theta + np.radians(60) * np.arange(6)
            cos_ha = np.cos(hex_angles)
            sin_ha = np.sin(hex_angles)
            
            # All 72 vertices
            all_verts = np.zeros((72, 2))
            for i in range(12):
                cx, cy = rotated_centers[i]
                all_verts[6*i:6*i+6] = np.column_stack([
                    cx + cos_ha,
                    cy + sin_ha
                ])
            
            s_val = _compute_min_s_for_cluster(all_verts, outer_angle)
            return s_val
        
        # Try multiple initial angles
        best_for_pattern = float('inf')
        best_params_for_pattern = None
        
        for init_theta in np.linspace(0, np.pi/3, 7):  # 0 to 60 degrees
            for init_outer in [0, 30]:
                x0 = np.array([init_theta, init_outer * np.pi / 180.0])
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                    if result.fun < best_for_pattern:
                        best_for_pattern = result.fun
                        best_params_for_pattern = result.x.copy()
                except Exception:
                    pass
                
                if time.time() > deadline - 120.0:
                    break
            if time.time() > deadline - 120.0:
                break
        
        if best_params_for_pattern is not None and best_for_pattern < best_s:
            best_s = best_for_pattern
            best_cluster = pattern
            best_rotation = best_params_for_pattern[0]
            best_outer_angle = best_params_for_pattern[1] * 180.0 / np.pi
    
    if best_cluster is None:
        # Fallback
        L = 2.0 * np.sqrt(3)
        pts = []
        for row in range(3):
            for col in range(4):
                pts.append([L * (col + row/2.0), L * row * np.sqrt(3)/2.0])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        best_cluster = pts
        best_rotation = 0.0
        best_outer_angle = 0.0
        # Compute s for fallback
        all_verts = np.zeros((72, 2))
        hex_angles = np.radians(60) * np.arange(6)
        for i in range(12):
            cx, cy = best_cluster[i]
            all_verts[6*i:6*i+6] = np.column_stack([
                cx + np.cos(hex_angles),
                cy + np.sin(hex_angles)
            ])
        best_s = _compute_min_s_for_cluster(all_verts, 0)
    
    # Now fine-tune: allow individual hexagons to have slightly different angles
    # and positions to further optimize
    if time.time() < deadline - 60.0:
        # Build initial inner array from best cluster
        c, s_ = np.cos(best_rotation), np.sin(best_rotation)
        R = np.array([[c, -s_], [s_, c]])
        rotated_centers = best_cluster @ R.T
        
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = rotated_centers[i, 0]
            inner[i, 1] = rotated_centers[i, 1]
            inner[i, 2] = best_rotation * 180.0 / np.pi  # all same angle
        
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        # Simulated annealing for fine-tuning
        np.random.seed(42)
        T_initial = 0.02
        T_min = 0.00001
        n_sa = 10000
        
        for it in range(n_sa):
            if time.time() > deadline - 60.0:
                break
            
            T = T_initial * (T_min / T_initial) ** (it / n_sa)
            new_inner = current_inner.copy() if 'current_inner' in dir() else inner.copy()
            new_outer_angle = current_outer_angle
            
            # Perturb 1-2 hexagons
            n_perturb = np.random.choice([1, 2], p=[0.7, 0.3])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.03
                new_inner[idx, 1] += np.random.randn() * 0.03
                new_inner[idx, 2] += np.random.randn() * 0.02
            if np.random.rand() < 0.2:
                new_outer_angle += np.random.randn() * 0.02
            
            # Check validity and compute s
            centers = new_inner[:, :2]
            all_verts = np.zeros((72, 2))
            for i in range(12):
                verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, new_inner[i, 2])
                all_verts[6*i:6*i+6] = verts
            
            new_s = _compute_min_s_for_cluster(all_verts, new_outer_angle)
            
            # Check overlaps
            is_valid = True
            all_normals_list = []
            for i in range(12):
                verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, new_inner[i, 2])
                all_normals_list.append(_edge_normals(verts))
            
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist > 2.0 * np.sqrt(3) + 0.1:
                        continue
                    verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, new_inner[i, 2])
                    verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, new_inner[j, 2])
                    overlap, _, _ = _sat_overlap(verts_i, all_normals_list[i], verts_j, all_normals_list[j])
                    if overlap > 1e-8:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                if new_s < current_s:
                    inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_s = current_s
                        best_rotation = inner[0, 2]
                        best_outer_angle = current_outer_angle
                else:
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                        inner = new_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
            else:
                # Reject invalid
                pass
        
        # Update best if improved
        if current_s < best_s:
            best_s = current_s
            best_outer_angle = current_outer_angle
            # Reconstruct inner from current
            best_inner = inner
        else:
            best_inner = inner
    
    # Final: construct output
    # Ensure best_inner is defined
    if 'best_inner' not in dir() or best_inner is None:
        c, s_ = np.cos(best_rotation), np.sin(best_rotation)
        R = np.array([[c, -s_], [s_, c]])
        rotated_centers = best_cluster @ R.T
        best_inner = np.zeros((12, 3))
        for i in range(12):
            best_inner[i, 0] = rotated_centers[i, 0]
            best_inner[i, 1] = rotated_centers[i, 1]
            best_inner[i, 2] = best_rotation * 180.0 / np.pi
    
    # Final validity check
    centers = best_inner[:, :2]
    all_verts = np.zeros((72, 2))
    all_normals_list = []
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, best_inner[i, 2])
        all_verts[6*i:6*i+6] = verts
        all_normals_list.append(_edge_normals(verts))
    
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            verts_i = all_verts[6*i:6*i+6]
            verts_j = all_verts[6*j:6*j+6]
            overlap, _, _ = _sat_overlap(verts_i, all_normals_list[i], verts_j, all_normals_list[j])
            if overlap > 1e-8:
                is_valid = False
                break
        if not is_valid:
            break
    
    # Check all vertices inside outer hexagon
    if is_valid:
        a_rad = np.radians(best_outer_angle)
        outer_verts = _hex_vertices(0, 0, best_s, best_outer_angle)
        for i in range(12):
            for v in all_verts[6*i:6*i+6]:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < -1e-6:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
    
    if not is_valid:
        # Fallback to simple valid configuration
        L = 2.0 * np.sqrt(3) + 0.01
        pts = []
        for row in range(3):
            for col in range(4):
                pts.append([L * (col + row/2.0), L * row * np.sqrt(3)/2.0, 0.0])
        pts = np.array(pts)
        center = np.mean(pts[:, :2], axis=0)
        pts[:, 0] -= center[0]
        pts[:, 1] -= center[1]
        best_inner = pts
        best_outer_angle = 0.0
        all_verts = np.zeros((72, 2))
        hex_angles = np.radians(60) * np.arange(6)
        for i in range(12):
            all_verts[6*i:6*i+6] = np.column_stack([
                pts[i, 0] + np.cos(hex_angles),
                pts[i, 1] + np.sin(hex_angles)
            ])
        best_s = _compute_min_s_for_cluster(all_verts, 0)
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
