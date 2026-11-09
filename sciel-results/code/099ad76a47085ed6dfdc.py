import numpy as np
import time

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    """Generate vertices of a regular hexagon."""
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _calculate_min_side(positions, angles):
    """
    Calculate the minimum side length of a regular hexagon (centered at origin, rotation 0)
    that contains all the given hexagons.
    We can rotate the outer hexagon to minimize the side, but for a symmetric cluster,
    alignment with the lattice is optimal.
    """
    all_verts = []
    for i in range(len(positions)):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon is centered at (0,0) for calculation simplicity, 
    # we will re-center later.
    # For a regular hexagon of side S, the distance from center to side (apothem) is S*sqrt(3)/2.
    # A point (x,y) is inside if its projection onto the 6 normal directions is <= apothem.
    # Normal directions for a hexagon with "flat" top/bottom (vertex at 90 deg? No, standard is vertex at 0).
    # Let's assume the outer hexagon has vertices at angles 0, 60, ...
    # Normals are at 30, 90, 150, ...
    
    normals_angles = np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    proj = all_verts @ normals.T
    max_proj = np.abs(proj).max()
    
    # max_proj = apothem = S * sqrt(3) / 2
    S = 2.0 * max_proj / np.sqrt(3.0)
    
    return S

def construct():
    start_time = time.time()
    
    # 1. Construct a valid hexagonal lattice cluster of 11 hexagons
    # Axial coordinates for a cluster
    # Center: (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 4 hexes (selected for symmetry)
    
    axial_coords = []
    axial_coords.append((0, 0))
    
    # Ring 1
    axial_coords.extend([
        (1, 0), (0, 1), (-1, 1), 
        (-1, 0), (0, -1), (1, -1)
    ])
    
    # Ring 2 - pick 4 symmetric ones
    # Full ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (3,-1), (3,-2) is not ring 2.
    # Ring 2 coords: 
    # (2,0), (2,-1), (1,-2) ... wait, let's use standard axial.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) are the 10? No, 12.
    # Let's just take the first 4 of the next ring that are roughly 90 deg apart.
    # (2,0), (0,2), (-2,0), (0,-2) are 4 points.
    axial_coords.extend([
        (2, 0), 
        (0, 2), 
        (-2, 0), 
        (0, -2)
    ])
    
    # Convert to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    for i, (q, r) in enumerate(axial_coords[:N]):
        positions[i, 0] = 1.5 * q
        positions[i, 1] = np.sqrt(3) * (r + q/2)
        angles[i] = 0.0
        
    # Center the configuration
    center = np.mean(positions, axis=0)
    positions -= center
    
    # Calculate the side length of the enclosing hexagon
    # We assume the outer hexagon is aligned with the lattice (angle 0)
    side = _calculate_min_side(positions, angles)
    
    # Add a small safety margin to ensure validity due to floating point errors
    side *= 1.001
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
    # outer_center: (0, 0) since we centered the positions
    outer_center = (0.0, 0.0)
    # outer_side
    outer_side = side
    # outer_angle
    outer_angle = 0.0
    
    # Optional: Quick optimization to tighten the fit
    # If we have time, we can try to rotate the outer hexagon to reduce side length
    # But for now, this explicit construction is robust.
    
    return inner, outer_center, outer_side, np.degrees(outer_angle)
