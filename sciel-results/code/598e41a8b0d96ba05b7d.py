"""
Packing 12 unit regular hexagons inside a regular hexagon.
Strategy: Binary search on outer side length + Nelder-Mead optimization.
The optimization minimizes a penalty function based on containment violations 
and pairwise overlaps. The outer hexagon's rotation angle is also optimized.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons v1, v2.
    Returns 0 if no overlap, positive value if overlapping.
    v1, v2 are (6, 2) arrays of vertices.
    """
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            # Ensure normal points outward
            mid = (p1 + p2) / 2
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return min_overlap

def _objective(params, outer_side):
    """
    Objective function for the optimizer.
    params: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    Returns sum of squared penalties.
    """
    # Unpack
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append((x, y, a))
    
    outer_angle = params[N*3]
    
    penalty = 0.0
    
    # Precompute outer hexagon normals and apothem
    a_outer = np.radians(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    # Normals for the outer hexagon boundaries
    normal_angles = a_outer + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    cos_n = np.cos(normal_angles)
    sin_n = np.sin(normal_angles)
    
    # Precompute inner vertices for reuse
    verts_list = []
    for x, y, a in inner:
        verts_list.append(_hex_verts(x, y, 1.0, a))
        
    # 1. Disjointness Penalty
    for i in range(N):
        v1 = verts_list[i]
        for j in range(i+1, N):
            v2 = verts_list[j]
            depth = _sat_overlap_depth(v1, v2)
            if depth > 0:
                penalty += depth * depth
    
    # 2. Containment Penalty
    # For each vertex of each inner hexagon, check projection against outer normals
    for v in verts_list:
        # v is (6, 2)
        # Calculate projections for all 6 normals at once
        # proj = v @ [cos_n, sin_n]^T -> shape (6, 6)
        # We want max projection for each vertex? No, we want to check if ANY vertex violates ANY normal.
        # Actually, for a convex polygon to be inside another convex polygon defined by half-planes,
        # all vertices of the inner polygon must satisfy the half-plane inequalities of the outer polygon.
        
        # proj_matrix[i, j] is projection of vertex i on normal j
        proj_matrix = v @ np.column_stack([cos_n, sin_n])
        
        # If any projection > apothem, there is a violation
        # We sum the squared violations
        viol = proj_matrix - apothem
        # Only count positive violations
        viol = np.where(viol > 0, viol, 0)
        penalty += np.sum(viol * viol)
            
    return penalty

def _try_pack(outer_side, time_budget=5.0):
    """
    Try to find a valid packing for the given outer_side.
    Returns (success, best_params, best_penalty).
    """
    start_time = time.time()
    best_params = None
    best_penalty = float('inf')
    
    # We will do multiple random restarts and structured inits
    np.random.seed(42) # Base seed, but we vary it per restart
    
    def get_initial_config():
        """Random scatter initialization."""
        config = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 60)
            config.extend([x, y, a])
        config.append(np.random.uniform(0, 60)) # Outer angle
        return np.array(config)
        
    def get_grid_config(rows, cols, spacing_x, spacing_y, outer_angle=0.0):
        """Structured hexagonal grid initialization."""
        config = []
        count = 0
        # Calculate offset to center the grid
        # Approximate center of the grid
        total_w = (cols - 1) * spacing_x
        total_h = (rows - 1) * spacing_y
        ox = -total_w / 2.0
        oy = -total_h / 2.0
        
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                x = ox + c * spacing_x + (r % 2) * spacing_x / 2
                y = oy + r * spacing_y
                config.extend([x, y, 0.0])
                count += 1
        # Pad if less than N (should not happen for 3x4=12)
        while len(config) < N*3:
             config.extend([np.random.uniform(-3,3), np.random.uniform(-3,3), 0.0])
        config.append(outer_angle) # Outer angle
        return np.array(config[:N*3+1])

    # Initial configurations to try
    # Hexagonal grid spacing for unit hexagons:
    # Distance between centers in a row: 2 * sqrt(3) / 2 * 2? No.
    # Side length s=1.
    # Horizontal distance between centers in same row: sqrt(3) * s? No, that's for flat-top?
    # Let's use pointy-top (angle 0).
    # Vertices at 0, 60, ...
    # Width = sqrt(3). Height = 2.
    # To pack tightly:
    # Row 0: x=0, y=0
    # Row 1: x=sqrt(3)/2, y=3/2
    # So spacing_x = sqrt(3), spacing_y = 1.5
    # But we can rotate them. Let's just use a standard grid.
    
    initial_configs = [
        get_grid_config(3, 4, np.sqrt(3), 1.5, 0.0),
        get_grid_config(4, 3, np.sqrt(3), 1.5, 0.0),
        get_grid_config(3, 4, np.sqrt(3), 1.5, 30.0), # Rotated outer
        get_grid_config(4, 3, np.sqrt(3), 1.5, 30.0),
        get_initial_config(),
        get_initial_config(),
        get_initial_config()
    ]
    
    for init in initial_configs:
        if time.time() - start_time > time_budget:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                args=(outer_side,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_penalty:
                best_penalty = res.fun
                best_params = res.x
                
                # If penalty is effectively zero, we found a solution
                if best_penalty < 1e-5:
                    return True, best_params, best_penalty
                    
        except Exception:
            continue
            
    return False, best_params, best_penalty

def construct():
    start_time = time.time()
    
    # Binary search for the minimum side length
    low = 3.0
    high = 8.0
    
    best_params = None
    best_side = high
    
    # First, ensure we have a valid packing at 'high'
    # If not, increase high
    success, params, pen = _try_pack(high, time_budget=10.0)
    if not success or pen > 1e-4:
         high = 10.0
         success, params, pen = _try_pack(high, time_budget=10.0)
         if not success or pen > 1e-4:
             # Fallback: return a very large hexagon with a valid grid
             # Grid 3x4 with spacing sqrt(3), 1.5
             inner = []
             for r in range(3):
                 for c in range(4):
                     x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2
                     y = r * 1.5
                     # Center it
                     x -= 1.5 * np.sqrt(3)
                     y -= 1.5
                     inner.append([x, y, 0.0])
             # Estimate side length
             # Max x range: approx 3*sqrt(3) + 1
             # Max y range: approx 3
             # Side length approx max(width, height) / sqrt(3)
             side = 10.0
             return inner, (0.0, 0.0), side, 0.0
    else:
        best_params = params
        best_side = high

    # Binary search
    # We want to be fast. 
    # Each iteration takes ~5-10s.
    # We have ~130s left.
    # Let's do at most 10-12 iterations.
    
    for _ in range(12):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        mid = (low + high) / 2.0
        
        success, params, pen = _try_pack(mid, time_budget=10.0)
        
        if success and pen < 1e-4:
            high = mid
            best_side = mid
            best_params = params
        else:
            low = mid
            
    if best_params is None:
        # Should not happen if we found one at start
        return [[0,0,0]]*12, (0.0, 0.0), 10.0, 0.0

    # Extract result
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    outer_angle = best_params[N*3]
    
    return inner, (0.0, 0.0), best_side, outer_angle

