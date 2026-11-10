import numpy as np
from scipy.optimize import minimize_scalar
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def _min_outer_side_for_angle(vertices, outer_angle_deg):
    """Minimum side of outer hexagon with given rotation containing all vertices."""
    a = np.deg2rad(outer_angle_deg)
    normal_angles = a + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    projections = vertices @ normals.T
    max_projections = np.max(projections, axis=1)
    min_apothem = np.max(max_projections)
    side = min_apothem / np.cos(np.deg2rad(30))
    return side

def _optimal_outer_side(vertices):
    """Find optimal outer hexagon rotation to minimize side length."""
    # Due to 60-degree symmetry, search in [0, 60)
    def obj(angle):
        return _min_outer_side_for_angle(vertices, angle)
    
    result = minimize_scalar(obj, bounds=(0, 60), method='bounded', 
                             options={'xatol': 1e-10})
    return result.fun, result.x

def _axial_to_cartesian(q, r, spacing):
    """Convert axial coordinates to cartesian for honeycomb."""
    x = spacing * (q + r / 2.0)
    y = spacing * r * np.sqrt(3) / 2.0
    return x, y

def _generate_clusters():
    """Generate candidate 11-hexagon honeycomb clusters."""
    clusters = []
    
    # Cluster 1: 7-hex flower + 4 more
    # 7-hex flower: center + 6 neighbors
    # Add 4 hexes to make it more hexagonal
    # Axial coords for 7-hex flower:
    flower = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Option A: Add hexes at (2,-1), (2,0), (-2,1), (-2,0)
    cluster_a = flower + [(2,-1), (2,0), (-2,1), (-2,0)]
    clusters.append(cluster_a)
    
    # Option B: Add hexes at (2,0), (1,1), (-2,0), (-1,-1)
    cluster_b = flower + [(2,0), (1,1), (-2,0), (-1,-1)]
    clusters.append(cluster_b)
    
    # Option C: Add hexes at (2,-1), (1,1), (-2,1), (-1,-1)
    cluster_c = flower + [(2,-1), (1,1), (-2,1), (-1,-1)]
    clusters.append(cluster_c)
    
    # Option D: 3-4-3-1 pattern
    # Row 0 (y=0): 3 hexes at x = -1, 0, 1
    # Row 1 (y=sqrt(3)/2): 4 hexes
    # Row 2 (y=sqrt(3)): 3 hexes
    # Row 3: 1 hex
    # Actually let me think in axial coords more carefully
    
    # Let me try: a roughly hexagonal shape
    # Center at (0,0)
    # Ring 1: 6 hexes
    # Ring 2 partial: 4 hexes
    # To make it symmetric, place the 4 at alternating positions
    
    # Option E: Add at (2,0), (0,2), (-2,0), (0,-2)
    cluster_e = flower + [(2,0), (0,2), (-2,0), (0,-2)]
    clusters.append(cluster_e)
    
    # Option F: Add at (2,-1), (0,2), (-2,1), (2,0) -- asymmetric
    # Let me try more symmetric ones
    
    # Option G: 4 hexes at corners of a larger hexagon
    # (2,-1), (1,1), (-1,2), (-2,1), (-1,-1), (1,-2) -- that's 6, too many
    # Pick 4: (2,-1), (1,1), (-1,2), (-2,1)
    cluster_g = flower + [(2,-1), (1,1), (-1,2), (-2,1)]
    clusters.append(cluster_g)
    
    # Option H: (2,-1), (-1,1), (-1,-1), (1,-2)
    cluster_h = flower + [(2,-1), (-1,1), (-1,-1), (1,-2)]
    clusters.append(cluster_h)
    
    # Option I: (2,0), (0,1), (-2,0), (0,-1) -- but (0,1) and (0,-1) already in flower
    # Skip
    
    # Option J: (2,-1), (2,0), (-1,2), (-2,1)
    cluster_j = flower + [(2,-1), (2,0), (-1,2), (-2,1)]
    clusters.append(cluster_j)
    
    # Option K: (2,-1), (1,1), (-2,1), (-1,-1)
    cluster_k = flower + [(2,-1), (1,1), (-2,1), (-1,-1)]
    clusters.append(cluster_k)
    
    # Option L: (2,0), (1,1), (-2,0), (-1,-1)  -- same as B
    # Option M: (2,-1), (1,1), (-1,2), (-2,1) -- same as G
    
    # Let me also try non-flower-based clusters
    # A 3x4 rectangular-ish arrangement
    # In axial coords, a "rectangle" is tricky
    
    # Try: two rows of 4 and one row of 3
    # Row at r=0: q = -2, -1, 0, 1 (4 hexes)
    # Row at r=1: q = -1, 0, 1, 2 (4 hexes)  
    # Row at r=-1: q = -1, 0, 1 (3 hexes)
    cluster_m = [(-2,0), (-1,0), (0,0), (1,0),
                 (-1,1), (0,1), (1,1), (2,1),
                 (-1,-1), (0,-1), (1,-1)]
    clusters.append(cluster_m)
    
    # Option N: 4-3-4 arrangement
    cluster_n = [(-2,0), (-1,0), (0,0), (1,0),
                 (-1,1), (0,1), (1,1),
                 (-1,-1), (0,-1), (1,-1), (2,-1)]
    clusters.append(cluster_n)
    
    return clusters

def _cluster_to_positions(cluster, hex_angle_deg=0):
    """Convert axial cluster to cartesian positions with unit hexagons."""
    # Spacing between centers = sqrt(3) for touching hexagons
    spacing = np.sqrt(3)
    positions = []
    for q, r in cluster:
        x, y = _axial_to_cartesian(q, r, spacing)
        positions.append([x, y])
    return np.array(positions)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580
    
    best_side = float('inf')
    best_cluster = None
    best_outer_angle = 0
    
    clusters = _generate_clusters()
    
    for cluster in clusters:
        if time.time() > deadline - 10:
            break
            
        positions = _cluster_to_positions(cluster)
        
        # All inner hexagons have same orientation (0 degrees for honeycomb)
        inner_angle = 0.0
        
        # Get all vertices
        all_vertices = []
        for i in range(11):
            v = _hex_vertices(positions[i][0], positions[i][1], 1.0, inner_angle)
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        # Find optimal outer hexagon
        side, outer_angle = _optimal_outer_side(all_vertices)
        
        if side < best_side:
            best_side = side
            best_cluster = positions.copy()
            best_outer_angle = outer_angle
    
    # Also try rotating the inner hexagons
    for inner_angle in [0, 30, 15, 45, 60]:
        if time.time() > deadline - 10:
            break
        for cluster in clusters:
            if time.time() > deadline - 10:
                break
            positions = _cluster_to_positions(cluster)
            
            all_vertices = []
            for i in range(11):
                v = _hex_vertices(positions[i][0], positions[i][1], 1.0, inner_angle)
                all_vertices.append(v)
            all_vertices = np.vstack(all_vertices)
            
            side, outer_angle = _optimal_outer_side(all_vertices)
            
            if side < best_side:
                best_side = side
                best_cluster = positions.copy()
                best_outer_angle = outer_angle
    
    # Verify and construct result
    if best_cluster is None:
        return _get_seed_result()
    
    # Verify no overlaps
    if not _verify_no_overlap(best_cluster, 0.0):
        return _get_seed_result()
    
    # Verify containment
    all_vertices = []
    for i in range(11):
        v = _hex_vertices(best_cluster[i][0], best_cluster[i][1], 1.0, 0.0)
        all_vertices.append(v)
    all_vertices = np.vstack(all_vertices)
    
    side = _min_outer_side_for_angle(all_vertices, best_outer_angle)
    
    if side < np.sqrt(11):
        return _get_seed_result()
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_cluster[i][0], best_cluster[i][1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, side

def _verify_no_overlap(centers, angles_deg, eps=1e-6):
    """Verify all pairs of hexagons are separated."""
    n = len(centers)
    vertices_list = []
    for i in range(n):
        v = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles_deg)
        vertices_list.append(v)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            a1 = np.deg2rad(angles_deg)
            a2 = np.deg2rad(angles_deg)
            
            n1_angles = a1 + np.deg2rad(60 * np.arange(6)) + np.deg2rad(30)
            n2_angles = a2 + np.deg2rad(60 * np.arange(6)) + np.deg2rad(30)
            
            normals1 = np.column_stack((np.cos(n1_angles), np.sin(n1_angles)))
            normals2 = np.column_stack((np.cos(n2_angles), np.sin(n2_angles)))
            
            all_normals = np.vstack([normals1, normals2])
            
            min_gap = np.inf
            for n_vec in all_normals:
                proj1 = v1 @ n_vec
                proj2 = v2 @ n_vec
                
                max1 = np.max(proj1)
                min1 = np.min(proj1)
                max2 = np.max(proj2)
                min2 = np.min(proj2)
                
                gap = max(min1 - max2, min2 - max1)
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < eps:
                return False
    
    return True

def _get_seed_result():
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length
