"""
Strategy: Topology Enumeration and Rigid Body Optimization.

Instead of using Simulated Annealing to search the 30-dimensional space of
individual hexagon positions and angles (which is non-convex and prone to
getting stuck in local minima due to overlap penalties), this program:

1. Assumes that the optimal packing for N=11 consists of a rigid cluster of
   hexagons with a fixed contact topology.
2. Enumerates several candidate topologies (e.g., "Flower" variants) where
   relative positions are determined by geometric constraints of touching
   unit hexagons.
3. For each rigid topology, optimizes the global rotation and translation
   of the entire cluster to minimize the side length of the enclosing
   regular hexagon. This reduces the problem to a 3-variable optimization
   (or even 1-variable if translation is centered) which is smooth and
   fast to solve.
4. Selects the topology and orientation that yields the smallest outer
   hexagon side length.

This approach avoids the noise of soft overlap penalties and leverages the
structural rigidity of small packings.
"""

import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_vectorized(inner_pos, inner_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute minimum outer side length to contain all inner hexagons.
    """
    cx, cy = outer_center
    pos = np.asarray(inner_pos, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _generate_topology_flower(valley_count=4, aligned_count=0):
    """
    Generate a 'flower' topology: 1 center, 6 inner ring, 4 outer.
    The 4 outer are chosen from the 6 'valley' spots (distance 3) and/or
    6 'aligned' spots (distance 2*sqrt(3)).
    """
    sqrt3 = np.sqrt(3)
    positions = [(0.0, 0.0)]
    angles = [0.0]
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        a = i * np.pi / 3
        x = sqrt3 * np.cos(a)
        y = sqrt3 * np.sin(a)
        positions.append((x, y))
        angles.append(0.0)
        
    # Ring 2: 4 hexagons
    # Valley spots: distance 3, angles 30, 90, 150, 210, 270, 330
    # Aligned spots: distance 2*sqrt(3), angles 0, 60, 120, 180, 240, 300
    
    # We want to pick 4. Let's try picking 4 from valleys.
    # To break symmetry, we pick indices 0, 1, 2, 3 of the valley spots?
    # Or maybe we should optimize which 4?
    # For now, let's pick 4 valleys: 30, 90, 150, 210.
    valley_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]
    
    # Pick first 4
    for i in range(valley_count):
        a = valley_angles[i]
        x = 3.0 * np.cos(a)
        y = 3.0 * np.sin(a)
        positions.append((x, y))
        angles.append(0.0)
        
    # If we need aligned ones (not used in this call)
    # ...
    
    return np.array(positions), np.array(angles)

def _generate_topology_line():
    """
    Generate a 'line' topology: 5 hexagons in a row, 3 on one side, 3 on other?
    5 + 3 + 3 = 11.
    """
    sqrt3 = np.sqrt(3)
    positions = []
    angles = []
    
    # Line of 5 along x-axis
    for i in range(-2, 3):
        positions.append((i * sqrt3, 0.0))
        angles.append(0.0)
        
    # 3 on top, 3 on bottom
    # Top hexagons touch the line.
    # A hexagon at (x, y) touches (x - sqrt3/2, 0) and (x + sqrt3/2, 0)?
    # Distance from (x, y) to (x - sqrt3/2, 0) is sqrt( (sqrt3/2)^2 + y^2 ) = sqrt(3/4 + y^2).
    # We want distance sqrt(3).
    # 3/4 + y^2 = 3 => y^2 = 9/4 => y = 1.5.
    # So top hexagons at y=1.5.
    # Which x?
    # To be compact, place them above the gaps?
    # Gaps are at x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3?
    # Let's place 3 top hexagons at x = -sqrt3, 0, sqrt3?
    # Check overlap with line hexagons.
    # Top hex at (0, 1.5). Line hex at (0, 0). Distance 1.5 < sqrt(3). Overlap!
    # So we can't place directly above a line hexagon.
    # We must place in the gaps.
    # Gaps between line hexagons are at x = i*sqrt3 + sqrt3/2.
    # Let's place top hexagons at x = -sqrt3/2, sqrt3/2, 3*sqrt3/2?
    # This is getting complex. Let's skip Line topology for now and stick to Flower variants.
    return None

def _optimize_rigid_cluster(positions, angles):
    """
    Optimize global rotation and translation of a rigid cluster.
    Returns best (positions, angles, outer_side).
    """
    # Center the cluster at origin
    centroid = np.mean(positions, axis=0)
    pos_centered = positions - centroid
    
    best_side = float('inf')
    best_rot = 0.0
    best_trans = (0.0, 0.0)
    
    # Coarse search for rotation
    # The cluster has some symmetry, but let's search 0-60 degrees.
    for theta_deg in np.arange(0, 60, 0.5):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        # Rotate positions
        pos_rot = pos_centered @ rot_mat.T
        # Angles also rotate
        ang_rot = angles + theta_deg
        
        # Compute min outer side with center at (0,0)
        side = _min_outer_side_vectorized(pos_rot, ang_rot)
        
        if side < best_side:
            best_side = side
            best_rot = theta
            best_trans = (0.0, 0.0) # Centered
            
    # Refine rotation around best_rot
    for theta_deg in np.arange(best_rot - 0.5, best_rot + 0.5, 0.01):
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        pos_rot = pos_centered @ rot_mat.T
        ang_rot = angles + theta_deg
        
        side = _min_outer_side_vectorized(pos_rot, ang_rot)
        
        if side < best_side:
            best_side = side
            best_rot = theta
            best_trans = (0.0, 0.0)
            
    # Apply best rotation
    c, s = np.cos(best_rot), np.sin(best_rot)
    rot_mat = np.array([[c, -s], [s, c]])
    pos_final = pos_centered @ rot_mat.T
    ang_final = angles + np.degrees(best_rot)
    
    return pos_final, ang_final, best_side

def construct():
    global START_TIME, DEADLINE
    
    best_pos = None
    best_ang = None
    best_side = float('inf')
    
    # Try different topologies
    
    # Topology 1: Flower with 4 valleys
    pos1, ang1 = _generate_topology_flower(valley_count=4)
    if time.time() < DEADLINE - 10.0:
        p, a, s = _optimize_rigid_cluster(pos1, ang1)
        if s < best_side:
            best_side = s
            best_pos = p
            best_ang = a
            
    # Topology 2: Flower with 2 valleys, 2 aligned
    # Modify _generate_topology_flower to support this?
    # For now, let's just try a different subset of valleys.
    # Pick valleys 0, 1, 4, 5 (symmetric-ish?)
    sqrt3 = np.sqrt(3)
    pos2 = [(0.0, 0.0)]
    ang2 = [0.0]
    for i in range(6):
        a = i * np.pi / 3
        pos2.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
        ang2.append(0.0)
    
    valley_angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2, 11*np.pi/6]
    # Pick 0, 1, 4, 5
    for i in [0, 1, 4, 5]:
        a = valley_angles[i]
        pos2.append((3.0 * np.cos(a), 3.0 * np.sin(a)))
        ang2.append(0.0)
        
    pos2 = np.array(pos2)
    ang2 = np.array(ang2)
    if time.time() < DEADLINE - 10.0:
        p, a, s = _optimize_rigid_cluster(pos2, ang2)
        if s < best_side:
            best_side = s
            best_pos = p
            best_ang = a
            
    # Topology 3: Flower with 4 aligned
    pos3 = [(0.0, 0.0)]
    ang3 = [0.0]
    for i in range(6):
        a = i * np.pi / 3
        pos3.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
        ang3.append(0.0)
        
    aligned_angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    # Pick 0, 1, 2, 3
    for i in range(4):
        a = aligned_angles[i]
        pos3.append((2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)))
        ang3.append(0.0)
        
    pos3 = np.array(pos3)
    ang3 = np.array(ang3)
    if time.time() < DEADLINE - 10.0:
        p, a, s = _optimize_rigid_cluster(pos3, ang3)
        if s < best_side:
            best_side = s
            best_pos = p
            best_ang = a

    if best_pos is None:
        # Fallback
        pos1, ang1 = _generate_topology_flower(valley_count=4)
        best_pos, best_ang, best_side = _optimize_rigid_cluster(pos1, ang1)
        
    # Convert to list format
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
