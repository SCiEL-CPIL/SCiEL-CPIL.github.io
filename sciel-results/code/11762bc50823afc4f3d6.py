"""
Strategy: Rigid Cluster Rotation Optimization.

Key Differences from Current Program:
1. Problem Framing: Instead of optimizing 33 independent variables (11 positions + 11 angles) 
   with a complex penalty function for overlaps, this strategy assumes the optimal internal 
   packing is a rigid hexagonal lattice (center + 6 neighbors + 4 outer).
2. Optimization Method: We optimize only the global rotation angle of the cluster. 
   The internal geometry is fixed to the minimal non-overlapping configuration.
3. Efficiency: This reduces the problem from a 33D non-convex optimization with 
   discontinuities to a 1D (or 2D if translation is included) periodic function 
   minimization, which is extremely fast and robust.
4. Containment: We compute the exact minimum side length of the outer hexagon required 
   to contain the rotated cluster by checking the support function against all vertices.
"""

import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

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

def _get_cluster_vertices(rotation_angle):
    """
    Generate vertices of the 11 unit hexagons in the rigid cluster,
    rotated by rotation_angle around the origin.
    
    The cluster topology is:
    - 1 center hex at (0,0)
    - 6 hexes at distance 2*sqrt(3) (touching)
    - 4 hexes in the gaps of the second ring, at distance 6? 
      Actually, for a perfect hexagonal packing of 19 hexes, the 2nd ring is at 4*sqrt(3).
      But we only have 11. 
      The standard "flower" of 7 hexes has radius 2*sqrt(3) (center to vertex of outer ring is 2*sqrt(3)+1).
      Adding 4 more hexes to make 11 usually means filling the "valleys" of the 7-hex flower.
      The distance from center to the center of the 4 added hexes (in the gaps) is 6.
      Wait, let's verify distances.
      Center to neighbor center: d1 = 2*sqrt(3) ≈ 3.464.
      Neighbor to neighbor (adjacent in ring): d1 = 2*sqrt(3).
      The "gap" hexes are located at angles 30, 150, 210, 330 degrees relative to the main ring.
      Distance from center to gap hex center: 
      In a hexagonal grid, if ring 1 is at r=2*sqrt(3), the next ring positions are at r=4*sqrt(3) (axial) or r=6 (diagonal/gap).
      Let's check: A hexagon at (2*sqrt(3), 0) and (sqrt(3), 3). 
      Midpoint is (1.5*sqrt(3), 1.5). Distance = sqrt(6.75 + 2.25) = 3.
      Wait, the centers of the 7 hexes form a hexagon of side 2*sqrt(3).
      The "gap" positions are the vertices of a larger hexagon.
      Let's use the known coordinates for 11 hexes in a hexagonal packing.
      
      Coordinates for 11 hexes (side 1, touching):
      0: (0, 0)
      1-6: (2*sqrt(3) * cos(i*pi/3), 2*sqrt(3) * sin(i*pi/3))
      7-10: These are in the "armpits". The distance from center is 6.
      Angles: 30, 150, 210, 330 degrees.
      
      Let's verify non-overlap for this configuration.
      Center (0,0) to Gap (6*cos(30), 6*sin(30)) = (3*sqrt(3), 3). Dist = 6.
      Min dist for non-overlap is 2*sqrt(3) ≈ 3.46. 6 > 3.46. OK.
      Ring 1 Hex 1: (2*sqrt(3), 0).
      Gap Hex 7: (3*sqrt(3), 3).
      Dist between them: dx = sqrt(3), dy = 3. d = sqrt(3+9) = sqrt(12) = 2*sqrt(3). 
      So they touch. This is a valid packing.
    """
    sqrt3 = np.sqrt(3)
    centers = []
    centers.append([0.0, 0.0])
    
    # Ring 1
    for i in range(6):
        a = i * np.pi / 3
        centers.append([2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)])
        
    # Ring 2 (Gaps)
    gap_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for a in gap_angles:
        centers.append([6.0 * np.cos(a), 6.0 * np.sin(a)])
        
    centers = np.array(centers)
    
    # Rotate centers
    a_rot = np.radians(rotation_angle)
    c_rot = np.cos(a_rot)
    s_rot = np.sin(a_rot)
    
    # Apply rotation matrix
    rotated_centers = np.column_stack([
        centers[:, 0] * c_rot - centers[:, 1] * s_rot,
        centers[:, 0] * s_rot + centers[:, 1] * c_rot
    ])
    
    # Generate all vertices for all 11 hexagons
    # Each hexagon has 6 vertices.
    # Vertex offset for hexagon i at angle theta_i (local angle)
    # But the hexagons themselves also rotate with the cluster?
    # Yes, "rigid cluster" implies the internal angles are fixed relative to the cluster frame.
    # In the initial config, all hexagons are aligned with the global axes (angle 0).
    # When we rotate the cluster, the hexagons rotate too.
    
    all_verts = []
    for cx, cy in rotated_centers:
        # The local angle of the hexagon relative to the cluster is 0.
        # So its global angle is rotation_angle.
        verts = _hex_vertices(cx, cy, 1.0, rotation_angle)
        all_verts.append(verts)
        
    return np.vstack(all_verts)

def _min_outer_side(vertices, outer_center, outer_angle):
    """
    Compute minimum outer side length to contain all vertices.
    Outer hexagon is centered at outer_center, rotated by outer_angle.
    """
    cx, cy = outer_center
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    
    a_outer = np.radians(outer_angle)
    ca = np.cos(a_outer)
    sa = np.sin(a_outer)
    
    # Rotate vertices into outer hexagon frame
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Constraints for regular hexagon of side S centered at origin:
    # |lx| <= S
    # |lx + ly/sqrt3| <= S  (Wait, this depends on orientation)
    
    # Let's derive the support function for a regular hexagon.
    # A regular hexagon with side S, flat-top orientation (vertex at 0 deg? or flat at 0 deg?)
    # Standard definition: Vertices at 0, 60, 120...
    # If vertices are at 0, 60..., the "flat" sides are at 30, 90, 150...
    # The distance from center to vertex is S (circumradius = side length for regular hex).
    # The distance from center to flat side is S * sqrt(3)/2 (inradius).
    
    # If we align the outer hexagon such that its "up" vector is along y-axis?
    # Let's stick to the coordinate system where the hexagon is defined by inequalities.
    # For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
    # The normals to the sides are at 30, 90, 150, 210, 270, 330.
    # The equation for the side with normal angle phi is x*cos(phi) + y*sin(phi) <= S * cos(30) = S * sqrt(3)/2.
    # Wait, the apothem is S * sqrt(3)/2.
    # So the constraint is: max over normals n of (dot(p, n)) <= S * sqrt(3)/2.
    # Therefore S >= max(dot(p, n)) / (sqrt(3)/2).
    
    # Normals angles: 30, 90, 150, 210, 270, 330 degrees.
    # In radians: pi/6, pi/2, 5*pi/6, 7*pi/6, 3*pi/2, 11*pi/6.
    
    normals = []
    for i in range(6):
        theta = np.pi/6 + i * np.pi/3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # Project vertices onto normals
    # projections shape: (num_verts, 6)
    projections = vertices @ normals.T  # Wait, vertices is (N, 2), normals is (6, 2).
    # We want dot product of each vertex with each normal.
    # projections[i, j] = dot(vertices[i], normals[j])
    
    # Actually, `vertices @ normals.T` gives (N, 6).
    # We want the max projection for each normal.
    max_projections = np.max(projections, axis=0)
    
    # The required side length S is such that S * sqrt(3)/2 >= max_projections
    # S >= max_projections / (sqrt(3)/2)
    
    min_side = np.max(max_projections) / (np.sqrt(3) / 2)
    
    return float(min_side)

def _objective(angle_deg):
    """
    Objective function: Minimize the side length of the outer hexagon.
    We fix the outer hexagon center at (0,0) and angle at 0.
    We rotate the inner cluster by angle_deg.
    Due to symmetry, translating the cluster might also help, but for a symmetric cluster
    in a symmetric container, the optimal translation is likely (0,0).
    However, to be safe, we can also optimize a small translation.
    Let's stick to rotation only first, as it's 1D and very fast.
    If we need to translate, we can add 2 more dimensions.
    """
    verts = _get_cluster_vertices(angle_deg)
    side = _min_outer_side(verts, (0.0, 0.0), 0.0)
    return side

def construct():
    global START_TIME, DEADLINE
    
    # The function is periodic with period 60 degrees (symmetry of hexagon).
    # We can search in [0, 60).
    
    # Coarse search
    angles_coarse = np.linspace(0, 60, 601)
    best_angle = 0
    best_side = np.inf
    
    for a in angles_coarse:
        if time.time() > DEADLINE - 5.0:
            break
        s = _objective(a)
        if s < best_side:
            best_side = s
            best_angle = a
            
    # Fine search around best_angle
    # The function is smooth. We can use scipy.optimize.minimize_scalar
    from scipy.optimize import minimize_scalar
    
    # Define bounds for fine search
    lower = max(0, best_angle - 1.0)
    upper = min(60, best_angle + 1.0)
    
    try:
        res = minimize_scalar(_objective, bounds=(lower, upper), method='bounded', 
                              options={'xatol': 1e-6})
        if res.fun < best_side:
            best_side = res.fun
            best_angle = res.x
    except Exception:
        pass
    
    # Extract solution
    # The inner hexagons are at the rotated positions.
    # We need to return their centers and angles.
    
    sqrt3 = np.sqrt(3)
    centers = []
    centers.append([0.0, 0.0])
    for i in range(6):
        a = i * np.pi / 3
        centers.append([2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)])
    gap_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for a in gap_angles:
        centers.append([6.0 * np.cos(a), 6.0 * np.sin(a)])
        
    centers = np.array(centers)
    
    a_rot = np.radians(best_angle)
    c_rot = np.cos(a_rot)
    s_rot = np.sin(a_rot)
    
    rotated_centers = np.column_stack([
        centers[:, 0] * c_rot - centers[:, 1] * s_rot,
        centers[:, 0] * s_rot + centers[:, 1] * c_rot
    ])
    
    inner_list = []
    for i in range(N):
        inner_list.append([float(rotated_centers[i, 0]), float(rotated_centers[i, 1]), float(best_angle)])
        
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_angle = 0.0
    
    return inner_list, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    construct()
