# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _get_cluster_points(topology_id, theta_rad):
    """
    Generate the 12 center points of a rigid cluster of hexagons.
    Returns a (12, 2) array of points in the local frame (before translation/rotation? No, rotated).
    Actually, let's generate in local frame, then rotate.
    """
    sqrt3 = np.sqrt(3)
    if topology_id == 0:
        # 3x4 parallelogram
        # q in 0..2, r in 0..3
        points = []
        for q in range(3):
            for r in range(4):
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                points.append([x, y])
    elif topology_id == 1:
        # 2x6 parallelogram
        points = []
        for q in range(2):
            for r in range(6):
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                points.append([x, y])
    elif topology_id == 2:
        # "Flower" of 7 + 5 more
        # Center (0,0)
        # Neighbors at distance sqrt3
        # Let's define a specific set of 12 points that is compact
        # Hexagonal lattice points:
        # (0,0)
        # (sqrt3, 0), (-sqrt3, 0)
        # (sqrt3/2, 1.5), (-sqrt3/2, 1.5), (sqrt3/2, -1.5), (-sqrt3/2, -1.5)
        # These 7 form a hexagon of side 2.
        # Add 5 more to make it more square/compact?
        # Maybe extend to a 3x4 block is better.
        # Let's try a 3x4 block but centered.
        # For now, reuse 3x4 logic but maybe different aspect ratio?
        # Let's stick to 3x4 and 2x6.
        points = []
        for q in range(3):
            for r in range(4):
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                points.append([x, y])
    else:
        # Fallback 3x4
        points = []
        for q in range(3):
            for r in range(4):
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                points.append([x, y])
    
    pts = np.array(points)
    # Center the cluster
    pts -= pts.mean(axis=0)
    
    # Rotate by theta_rad
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    return pts

def _min_s_for_config(pts, phi_rad):
    """
    Compute minimum side s of outer hexagon (centered at 0,0, angle phi_rad)
    containing all points in pts.
    """
    # Outer hexagon normals
    # Vertices at angles phi + 60*i.
    # Edges are between vertices.
    # Normal to edge i points outward.
    # The angle of the normal to the edge between vertex i and i+1 is phi + 60*i + 30?
    # Let's derive.
    # Vertex 0: angle phi. Vertex 1: angle phi + 60.
    # Midpoint of edge 0-1 is at angle phi + 30.
    # The normal to the edge is in the direction of the midpoint from center.
    # So normal angle is phi + 30 + 60*i.
    
    normal_angles = phi_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Project all points onto each normal
    # dots shape (12, 6)
    dots = pts @ normals.T
    
    # For each normal direction i, we need s * sqrt(3)/2 >= max_j (p_j . n_i)
    # So s >= (2/sqrt(3)) * max_j (p_j . n_i)
    # We take the max over all i and j.
    
    max_proj = np.max(dots)
    
    # Wait, is it just the max projection?
    # Yes, because the constraint is p . n_i <= h for all p, i.
    # h = s * sqrt(3)/2.
    # So s = (2/sqrt(3)) * max_{i,j} (p_j . n_i).
    
    s = (2.0 / np.sqrt(3)) * max_proj
    
    return s

def _objective(params, topology_id):
    """
    params: [theta_rad, phi_rad, x, y]
    """
    theta, phi, x, y = params
    pts = _get_cluster_points(topology_id, theta)
    pts = pts + np.array([x, y])
    
    s = _min_s_for_config(pts, phi)
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    best_s = np.inf
    best_params = None
    best_topology = 0
    
    np.random.seed(42)
    
    topologies = [0, 1, 2]
    
    for topo in topologies:
        if time.time() > deadline - 60:
            break
            
        # Initial guesses for theta, phi, x, y
        # x, y should be 0 initially if we centered the cluster.
        # theta, phi can be 0.
        
        initial_guesses = [
            [0.0, 0.0, 0.0, 0.0],
            [np.pi/6, 0.0, 0.0, 0.0],
            [0.0, np.pi/6, 0.0, 0.0],
            [np.pi/12, np.pi/12, 0.0, 0.0],
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 30:
                break
                
            # Add small noise
            params = np.array(guess) + np.random.randn(4) * 0.1
            
            result = minimize(
                _objective,
                params,
                args=(topo,),
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'maxiter': 1000,
                    'adaptive': True,
                }
            )
            
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
                best_topology = topo
                
    if best_params is None:
        # Fallback
        best_params = np.array([0.0, 0.0, 0.0, 0.0])
        best_topology = 0
        best_s = _objective(best_params, best_topology)
        
    theta, phi, x, y = best_params
    
    # Generate final points
    pts = _get_cluster_points(best_topology, theta)
    pts = pts + np.array([x, y])
    
    # Construct output
    # Inner hexagons: (x, y, angle_deg)
    # Angle is theta in degrees
    theta_deg = np.degrees(theta)
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = pts[:, 0]
    inner_hex_data[:, 1] = pts[:, 1]
    inner_hex_data[:, 2] = theta_deg
    
    # Outer hexagon: (0, 0, phi_deg)
    phi_deg = np.degrees(phi)
    outer_hex_data = np.array([0.0, 0.0, phi_deg])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
