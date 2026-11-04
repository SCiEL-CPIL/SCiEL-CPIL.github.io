import numpy as np
import time

N = 11

def get_hex_vertices(center, angle, side=1.0):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    angle: rotation in radians
    side: side length
    """
    cx, cy = center
    angles = angle + np.arange(6) * np.pi / 3.0
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects p3-p4.
    Uses cross product orientation test.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def point_in_hexagon(p, vertices):
    """
    Check if point p is inside convex polygon defined by vertices (CCW).
    vertices: (6, 2) array.
    """
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        # Vector p1->p2 and p1->p
        v1 = p2 - p1
        v2 = p - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            return False
    return True

def are_hexagons_disjoint(v1, v2):
    """
    Check if two convex hexagons are disjoint.
    v1, v2: (6, 2) arrays.
    """
    # Check if any vertex of v1 is inside v2
    for i in range(6):
        if point_in_hexagon(v1[i], v2):
            return False
    # Check if any vertex of v2 is inside v1
    for i in range(6):
        if point_in_hexagon(v2[i], v1):
            return False
            
    # Check if any edges intersect
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def compute_min_enclosing_hex_side(vertices_list):
    """
    Compute the side length of the smallest regular hexagon (angle 0, centered at origin? No, centered at centroid of points? 
    Actually, the outer hexagon can be anywhere. 
    However, for a given set of points, the minimal enclosing regular hexagon with fixed orientation 
    is determined by the support functions. 
    But we can also rotate the outer hexagon. 
    To keep it simple and fast, let's assume the outer hexagon has angle 0 and is centered at (0,0). 
    We can translate the inner hexagons so their centroid is at (0,0).
    Then L is determined by max support in 6 directions.
    """
    # We assume the configuration is centered at origin.
    # Directions for a regular hexagon with angle 0 (flat top/bottom? or pointy top?)
    # Standard hexagon vertices at angles 0, 60, ...
    # Normals are at 30, 90, 150, ...
    # Let's use normals at angles: pi/6, pi/6+pi/3, ...
    normals = []
    for k in range(6):
        ang = np.pi / 6 + k * np.pi / 3
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for v_list in vertices_list:
        # v_list is (6,2)
        # Project all vertices onto all 6 normals
        # (6,2) dot (6,2).T -> (6,6)
        projs = v_list @ normals.T
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    # Distance from center to side in a regular hexagon of side L is L * sqrt(3) / 2
    # So max_proj = L * sqrt(3) / 2
    # L = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_L = float('inf')
    best_config = None
    
    # Initial configurations
    # 1. Hexagonal lattice subset
    # Centers for N=11 in a hex lattice:
    # 1 center, 6 around it, 4 more?
    # Let's generate a few random lattices and structured ones.
    
    configs = []
    
    # Structured: 1 center, 6 neighbors, 4 further out
    # Lattice spacing for disjoint hexagons is 2 * inradius * 2 / sqrt(3)? 
    # Distance between centers of adjacent hexagons in a tiling is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732? 
    # No, side length 1. Apothem = sqrt(3)/2. Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait, if they touch, distance is 2 * apothem.
    # Apothem of unit hexagon is sqrt(3)/2. So distance is sqrt(3) ~ 1.732.
    # Let's use spacing = sqrt(3) * 1.01 to ensure disjointness.
    
    spacing = np.sqrt(3.0) * 1.01
    
    # Generate points in a hex lattice
    # Basis vectors
    e1 = np.array([spacing, 0])
    e2 = np.array([spacing * 0.5, spacing * np.sqrt(3) / 2])
    
    # Get 11 closest points to origin in this lattice
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i * e1 + j * e2
            lattice_points.append(p)
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: np.linalg.norm(p))
    init_points = lattice_points[:N]
    
    # Add this config
    configs.append((init_points, np.zeros(N)))
    
    # Random restarts
    np.random.seed(42)
    for _ in range(20):
        # Random positions in a circle
        r = np.sqrt(np.random.rand(N)) * 4.0
        theta = np.random.rand(N) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ang = np.random.rand(N) * np.pi / 3.0
        pts = np.column_stack((x, y))
        configs.append((pts, ang))
        
    for init_pts, init_angs in configs:
        if time.time() > deadline:
            break
            
        # Center the configuration
        centroid = np.mean(init_pts, axis=0)
        current_pts = init_pts - centroid
        current_angs = init_angs.copy()
        
        # Local Search
        step_size = 0.2
        for it in range(2000):
            if time.time() > deadline:
                break
                
            # Compute current L
            vertices_list = [get_hex_vertices(current_pts[i], current_angs[i]) for i in range(N)]
            # Check disjointness
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if not are_hexagons_disjoint(vertices_list[i], vertices_list[j]):
                        valid = False
                        break
                if not valid:
                    break
                    
            if not valid:
                # Try to repair or just reject
                # For speed, just reject and perturb
                pass
            else:
                L = compute_min_enclosing_hex_side(vertices_list)
                if L < best_L:
                    best_L = L
                    best_config = (current_pts.copy(), current_angs.copy())
                    
            # Perturb
            # Pick a random hexagon
            idx = np.random.randint(N)
            # Random delta
            delta_x = np.random.uniform(-step_size, step_size)
            delta_y = np.random.uniform(-step_size, step_size)
            delta_ang = np.random.uniform(-0.1, 0.1)
            
            # Apply
            current_pts[idx] += [delta_x, delta_y]
            current_angs[idx] += delta_ang
            
            # Re-center
            centroid = np.mean(current_pts, axis=0)
            current_pts -= centroid
            
    if best_config is None:
        # Fallback: return a very large hexagon
        # Just place them far apart
        inner = []
        for i in range(N):
            inner.append([10.0 * i, 0.0, 0.0])
        return inner, (0.0, 0.0), 100.0, 0.0
        
    pts, angs = best_config
    inner = []
    for i in range(N):
        x, y = pts[i]
        ang_deg = np.degrees(angs[i])
        inner.append([x, y, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the EVOLVE-BLOCK contains the code.
# The prompt asks to keep function names and signatures.
# I will put the whole logic inside the block.
# EVOLVE-BLOCK-END
