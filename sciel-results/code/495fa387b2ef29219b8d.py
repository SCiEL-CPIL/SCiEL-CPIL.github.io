"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper: vertices of a regular hexagon with side length s, center (cx, cy), rotation angle (radians)
    def hex_vertices(cx, cy, s, angle):
        # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees
        # For pointy-top, offset by 30 degrees. Let's use flat-top (standard).
        # Actually, let's define a standard hexagon.
        # Vertices of a unit hexagon (side 1) centered at origin, flat-top:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Wait, side length 1 means distance from center to vertex is 1 for regular hexagon.
        # Yes, for regular hexagon, circumradius = side length.
        angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]) + angle
        x = cx + s * np.cos(angles)
        y = cy + s * np.sin(angles)
        return np.column_stack((x, y))
    
    # Helper: distance from point to segment
    def point_segment_distance(p, a, b):
        # p, a, b are 2D points
        ap = p - a
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 == 0:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / ab2
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    # Helper: check if two convex polygons intersect (using Separating Axis Theorem or similar)
    # For simplicity and robustness, we can use a simpler check: 
    # Since all are regular hexagons, we can check distance between centers.
    # But they can rotate. 
    # A robust way: for each pair, check if any vertex of one is inside the other, 
    # or if any edges intersect.
    # Given the complexity, let's use a simpler constraint: 
    # The minimum distance between any two hexagons must be >= 0.
    # We can approximate this by checking the distance between centers minus the "radius" in that direction.
    # But rotation makes it complex.
    # Let's use a simpler initial approach: assume all hexagons are axis-aligned (angle=0) 
    # and optimize positions and outer side length. This is a simpler problem.
    # If this works well, we can add rotation later.
    
    # Let's try to pack 12 unit hexagons in a hexagonal lattice pattern.
    # A hexagonal number H_k = 3k(k-1)+1.
    # H_1 = 1
    # H_2 = 7
    # H_3 = 19
    # So 12 hexagons fit in a "ring" around a central one, but not a full ring.
    # A full ring of 6 around 1 is 7.
    # Another ring adds 12, total 19.
    # So 12 is between 7 and 19.
    # We can place 1 in the center, 6 around it, and 5 more in the next ring.
    # Let's try to construct such a configuration.
    
    # Positions for 7 hexagons (center + 6 neighbors):
    # Center: (0, 0)
    # Neighbors: distance sqrt(3) apart (for flat-top, side 1, distance between centers is sqrt(3)).
    # Actually, for flat-top hexagons of side 1, the distance between centers of adjacent hexagons is sqrt(3).
    # Let's verify: 
    # Hexagon A center (0,0), vertex at (1,0).
    # Hexagon B center (sqrt(3), 0)? No.
    # Let's calculate properly.
    # Flat-top hexagon vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # The "width" is 2 (from -1 to 1). The "height" is sqrt(3).
    # To pack them, the center-to-center distance for horizontal neighbors is 1.5? No.
    # Let's look at the gap. The edge from (1,0) to (0.5, sqrt(3)/2) has length 1.
    # The normal to this edge points in direction (sqrt(3)/2, 0.5) ? No.
    # Let's just use the known result: for regular hexagons of side 1, the minimum distance between centers 
    # to avoid overlap is sqrt(3) if they share an edge? No, if they share an edge, the distance is sqrt(3).
    # Wait, if two unit hexagons share an edge, the distance between centers is the altitude of the hexagon, which is sqrt(3).
    # Yes. So adjacent centers are sqrt(3) apart.
    
    # Let's define positions for 12 hexagons in a hexagonal grid.
    # Let dx = 1.5, dy = sqrt(3)/2 for axial coordinates?
    # Let's use cube coordinates or axial coordinates for hex grids.
    # For flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # where (q, r) are axial coordinates.
    # Distance between (0,0) and (1,0) in axial is sqrt(3).
    # Let's generate 12 positions in a compact cluster.
    
    # Axial coordinates for 12 hexagons:
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) -> 6 hexagons. Total 7.
    # Ring 2: We need 5 more.
    # (2,0), (2,-1), (1,-2), (-1,2), (-2,1) -> 5 hexagons. Total 12.
    
    axial_coords = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (2,-1), (1,-2), (-1,2), (-2,1)
    ]
    
    positions = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        positions.append([x, y, 0.0])
    
    # Now we need to find the minimum outer hexagon side length that contains all these.
    # The outer hexagon is also regular, side length S, centered at (cx, cy), rotated by angle.
    # To minimize S, we should center the outer hexagon at the centroid of the inner hexagons.
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    
    # Translate positions to be centered at (0,0)
    centered_positions = [[p[0] - cx, p[1] - cy, 0.0] for p in positions]
    
    # Now, find the minimum S such that all vertices of all inner hexagons are inside the outer hexagon.
    # The outer hexagon has vertices at distance S from center (0,0) (since circumradius = side length).
    # Wait, is the outer hexagon required to contain the inner hexagons, or just their centers?
    # "Place 12 disjoint regular hexagons of side length 1 inside a regular hexagon"
    # This implies the entire inner hexagons must be inside the outer hexagon.
    
    # So, for each inner hexagon, we need all its vertices to be inside the outer hexagon.
    # The outer hexagon is the intersection of 6 half-planes.
    # For a regular hexagon centered at origin with side S and rotation angle theta:
    # The supporting lines are at distance S * sqrt(3) / 2 from the center?
    # No, the apothem (distance from center to edge) is S * sqrt(3) / 2.
    # So the constraint for a point (x,y) to be inside is:
    # For each of the 6 edges, the distance from the point to the edge line must be <= apothem.
    # Or, using the dot product with the outward normals.
    # The outward normals for a flat-top hexagon (rotation 0) are at angles 0, 60, 120, 180, 240, 300.
    # Wait, for flat-top, the edges are vertical? No.
    # Vertices at 0, 60, ... degrees. Edges are between vertices.
    # The normal to the edge between 0 and 60 degrees points at 30 degrees.
    # So the normals are at 30, 90, 150, 210, 270, 330 degrees.
    # Let's define the normals for rotation angle theta:
    # n_k = (cos(theta + pi/6 + k*pi/3), sin(theta + pi/6 + k*pi/3)) for k=0..5.
    # The constraint is: dot(point, n_k) <= S * sqrt(3) / 2 for all k.
    # So S >= 2 * max_k |dot(point, n_k)| / sqrt(3).
    # We need this for all points (vertices of all inner hexagons).
    # So S = max over all points p and all k of (2 * |dot(p, n_k)| / sqrt(3)).
    
    # We can optimize theta to minimize this S.
    # Let's sample theta from 0 to 60 degrees (due to symmetry).
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for x, y, ang in centered_positions:
        verts = hex_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Sample theta
    for theta in np.linspace(0, np.pi/3, 600):
        normals = []
        for k in range(6):
            angle = theta + np.pi/6 + k * np.pi/3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # For each vertex, compute max dot product with normals
        # dot products shape: (N_verts, 6)
        dots = np.abs(all_vertices @ normals.T)
        max_dots = np.max(dots, axis=1)
        S_required = np.max(max_dots) * 2 / np.sqrt(3)
        
        if S_required < best_S:
            best_S = S_required
            best_theta = theta
    
    # Refine theta around best_theta
    theta_range = np.linspace(best_theta - np.pi/100, best_theta + np.pi/100, 100)
    for theta in theta_range:
        normals = []
        for k in range(6):
            angle = theta + np.pi/6 + k * np.pi/3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        dots = np.abs(all_vertices @ normals.T)
        max_dots = np.max(dots, axis=1)
        S_required = np.max(max_dots) * 2 / np.sqrt(3)
        
        if S_required < best_S:
            best_S = S_required
            best_theta = theta
    
    # Convert best_theta to degrees
    best_theta_deg = np.degrees(best_theta)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best found so far
        return centered_positions, (cx, cy), best_S, best_theta_deg
    
    # Try to improve by allowing rotation of inner hexagons?
    # This is a much harder problem. Let's stick with this for now.
    # But we can try to optimize the positions slightly.
    # Given the time constraint, let's return this result.
    
    return centered_positions, (cx, cy), best_S, best_theta_deg
# EVOLVE-BLOCK-END
