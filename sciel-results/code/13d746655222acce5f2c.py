# EVOLVE-BLOCK-START
import numpy as np
import time

def _hexagon_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _support_hex(n, angle_deg=0.0):
    """
    Support function of a unit hexagon (side 1) centered at origin, rotated by angle_deg,
    in direction n (unit vector).
    """
    # Vertices of unit hexagon at angle 0
    # Angles: 0, 60, 120, 180, 240, 300
    # We can just compute max dot product with vertices
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.max(verts @ n)

def _calc_S_for_theta(theta_rad, centers):
    """
    Calculate the minimum side length S of a regular hexagon (centered at origin, angle 0)
    that contains the cluster of 12 unit hexagons with given centers (rotated by theta_rad).
    """
    # Container normals (angle 0)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees?
    # Vertices at 0, 60... Normals at 30, 90...
    # Let's use the definition: vertices at (S cos(a+60i), S sin(a+60i)).
    # If angle=0, vertices at 0, 60, 120...
    # Edge 0 connects v0 and v1. Midpoint angle 30. Normal angle 30.
    normal_angles = np.radians(30) + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_proj = 0.0
    # Rotate centers by theta
    rot = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                    [np.sin(theta_rad), np.cos(theta_rad)]])
    rotated_centers = centers @ rot.T
    
    # For each container normal, find max projection of any hexagon in the cluster
    for n in normals:
        # h_cluster(n) = max_i ( c_i_rotated . n + h_hex(n, angle_i) )
        # All inner hexagons are aligned with each other?
        # In the honeycomb lattice, all hexagons have the same orientation.
        # So angle_i = theta_rad for all i?
        # Wait, the centers are rotated by theta. The hexagons themselves are also rotated by theta.
        # So the orientation of each inner hexagon is theta_rad.
        h_hex_val = _support_hex(n, np.degrees(theta_rad))
        
        # Projections of centers
        projs = rotated_centers @ n
        max_proj_i = np.max(projs) + h_hex_val
        if max_proj_i > max_proj:
            max_proj = max_proj_i
            
    # S * apothem_factor = max_proj
    # Apothem of container = S * sqrt(3) / 2
    # So S = max_proj / (sqrt(3) / 2)
    S = max_proj / (np.sqrt(3) / 2.0)
    return S

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s).
    """
    # Define honeycomb lattice basis
    u = np.array([1.5, np.sqrt(3)/2])
    v = np.array([0, np.sqrt(3)])
    
    # Define 12 points in lattice coordinates (i, j)
    points = [
        (0,0),
        (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    centers = np.array([i * u + j * v for i, j in points])
    
    # Center the cluster
    center = np.mean(centers, axis=0)
    centers -= center
    
    # Search for optimal rotation theta in [0, pi/3)
    # The cluster is not 6-fold symmetric, so we scan.
    # Use a fine grid first, then refine.
    best_S = np.inf
    best_theta = 0.0
    
    # Coarse search
    thetas = np.linspace(0, np.pi/3, 360)
    for theta in thetas:
        S = _calc_S_for_theta(theta, centers)
        if S < best_S:
            best_S = S
            best_theta = theta
            
    # Fine search around best_theta
    # Golden section or just finer grid
    thetas_fine = np.linspace(best_theta - 0.01, best_theta + 0.01, 200)
    for theta in thetas_fine:
        S = _calc_S_for_theta(theta, centers)
        if S < best_S:
            best_S = S
            best_theta = theta

    # Construct final result
    # Rotate centers by best_theta
    rot = np.array([[np.cos(best_theta), -np.sin(best_theta)],
                    [np.sin(best_theta), np.cos(best_theta)]])
    final_centers = centers @ rot.T
    
    # Inner array: (x, y, angle_deg)
    # Angle of each hexagon is best_theta
    angle_deg = np.degrees(best_theta)
    inner = np.column_stack([final_centers[:, 0], final_centers[:, 1], 
                             np.full(12, angle_deg)])
    
    # Outer hexagon: centered at (0,0), side best_S, angle 0
    outer = np.array([0.0, 0.0, 0.0])
    s = best_S
    
    return inner, outer, s

# EVOLVE-BLOCK-END
