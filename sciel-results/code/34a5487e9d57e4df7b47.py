import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit, use 600 to be safe? Task says 600s wall clock.
    # Actually, the prompt says "KILLS the program after 630s".
    # "Keep an EXPLICIT deadline guard... safety margin before the limit."
    # Let's use 600s as the limit for our logic.
    
    # 1. Define inner hexagon centers
    # Core 7 hexagons
    sqrt3 = np.sqrt(3)
    core_centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5]
    ])
    
    # Edge 6 positions (Ring 2 midpoints)
    # u = (sqrt3, 0), v = (sqrt3/2, 1.5)
    # Edge positions are u+v, v+2u, etc.
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    # We pick 5 of them. Let's pick the first 5.
    edge_angles = np.deg2rad([30, 90, 150, 210, 270])
    edge_radius = 3.0 # Distance from origin
    edge_centers = np.array([
        [edge_radius * np.cos(a), edge_radius * np.sin(a)] for a in edge_angles
    ])
    
    all_centers = np.vstack([core_centers, edge_centers])
    
    # 2. Define inner hexagon angles (all 0 for optimal flat-side-out on edge hexes)
    inner_angles = np.zeros(12)
    
    inner_hex_data = np.column_stack((all_centers, inner_angles))
    
    # 3. Calculate all vertices of the 12 inner hexagons
    def get_vertices(cx, cy, d, angle_deg):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(60 * np.arange(6))
        xs = cx + d * np.cos(angles)
        ys = cy + d * np.sin(angles)
        return np.column_stack((xs, ys))
    
    all_vertices = []
    for i in range(12):
        cx, cy, ang = inner_hex_data[i]
        verts = get_vertices(cx, cy, 1.0, ang)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # 4. Optimize outer hexagon angle to minimize required side length
    # Outer hexagon centered at (0,0)
    # Side length S.
    # Condition: max projection of vertices on normals <= S * sqrt(3) / 2
    # Normals are at theta + 30 + 60k
    
    best_s = float('inf')
    best_theta = 0.0
    
    # Grid search for theta in [0, 60)
    # 3600 steps is fast
    thetas = np.linspace(0, 60, 3600, endpoint=False)
    
    for theta_deg in thetas:
        theta_rad = np.deg2rad(theta_deg)
        normal_angles = theta_rad + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Projection of all vertices on all normals
        # all_vertices shape (N, 2), normals shape (6, 2)
        # projections shape (N, 6)
        projections = all_vertices @ normals.T
        
        max_proj = np.max(projections)
        
        # S * sqrt(3) / 2 >= max_proj
        # S >= 2 * max_proj / sqrt(3)
        s_candidate = 2 * max_proj / sqrt3
        
        if s_candidate < best_s:
            best_s = s_candidate
            best_theta = theta_deg
            
        # Time check
        if time.time() > deadline - 10:
            break
            
    # 5. Construct return values
    outer_hex_data = np.array([0, 0, best_theta])
    
    return inner_hex_data, outer_hex_data, best_s
