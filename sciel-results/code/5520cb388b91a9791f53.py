"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Try to find a compact arrangement of 11 unit hexagons
    # Each hexagon is represented by (x, y, angle_degrees)
    # We'll try a hexagonal grid arrangement
    
    # Unit hexagon vertices (flat-top) centered at origin
    def hex_vertices(cx, cy, angle_deg=0):
        angle = np.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            verts.append((cx + np.cos(theta), cy + np.sin(theta)))
        return np.array(verts)
    
    # Try a compact arrangement: central hex + 6 around + 4 more
    # Using hexagonal grid coordinates
    # Let's try a specific known-good arrangement
    
    # Arrangement 1: 3-4-3-1 pattern or similar
    # Let me try placing them in a roughly hexagonal cluster
    
    # For a flat-top hexagon, neighbors are at:
    # (1, 0), (0.5, √3/2), (-0.5, √3/2), (-1, 0), (-0.5, -√3/2), (0.5, -√3/2)
    
    sqrt3 = np.sqrt(3)
    
    # Try a symmetric arrangement
    positions = [
        (0, 0),           # center
        (1, 0),           # right
        (0.5, sqrt3/2),   # upper right
        (-0.5, sqrt3/2),  # upper left
        (-1, 0),          # left
        (-0.5, -sqrt3/2), # lower left
        (0.5, -sqrt3/2),  # lower right
        # 4 more to add
        (1.5, sqrt3/2),   # upper far right
        (1.5, -sqrt3/2),  # lower far right
        (-1.5, sqrt3/2),  # upper far left
        (-1.5, -sqrt3/2), # lower far left
    ]
    
    # All hexagons have angle 0 (flat-top)
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Now compute the minimum enclosing regular hexagon
    # Get all vertices
    all_verts = []
    for x, y, a in inner:
        verts = hex_vertices(x, y, a)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # Find the minimum enclosing regular hexagon
    # A regular hexagon is defined by center (cx, cy), side length s, and rotation angle
    # The hexagon contains all points if for each point, the distance to each of the 6 sides <= s*√3/2
    
    def hexagon_contains(cx, cy, s, rot_deg, points):
        """Check if all points are inside the regular hexagon."""
        rot = np.radians(rot_deg)
        # The 6 outward normals of a flat-top hexagon rotated by rot
        # For a flat-top hexagon, the sides are at angles 0, 60, 120, 180, 240, 300 degrees
        # The apothem (distance from center to side) is s * √3 / 2
        apothem = s * sqrt3 / 2
        for i in range(6):
            theta = rot + i * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            # Distance from each point to this side (signed)
            dists = (points - np.array([cx, cy])) @ np.array([nx, ny])
            if np.any(dists > apothem + 1e-9):
                return False
        return True
    
    def compute_min_hexagon(inner_pts):
        """Find minimum side length regular hexagon containing all points."""
        # Use the fact that the minimum enclosing hexagon will be tight on at least 3 sides
        # We can try different rotations and for each, compute the required side length
        
        best_s = float('inf')
        best_cx = 0
        best_cy = 0
        best_rot = 0
        
        # Try various rotations
        for rot_deg in np.arange(0, 60, 5):  # Hexagon has 60-degree symmetry
            rot = np.radians(rot_deg)
            apothem_factor = sqrt3 / 2
            
            # For a given rotation, the hexagon is defined by 6 half-planes
            # We need to find center (cx, cy) and apothem a such that all points are inside
            # This is a linear programming problem
            
            # The constraints are: for each point p and each side i:
            # (p - c) · n_i <= a
            # where n_i is the outward normal of side i
            
            # We can solve this by trying to minimize a
            # For simplicity, let's try to center the hexagon at the centroid
            
            centroid = np.mean(inner_pts, axis=0)
            cx, cy = centroid
            
            # For this center and rotation, find the minimum a needed
            a_needed = 0
            for i in range(6):
                theta = rot + i * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                dists = (inner_pts - np.array([cx, cy])) @ np.array([nx, ny])
                a_needed = max(a_needed, np.max(dists))
            
            s_needed = a_needed / apothem_factor
            if s_needed < best_s:
                best_s = s_needed
                best_cx = cx
                best_cy = cy
                best_rot = rot_deg
        
        # Refine with more rotations
        for rot_deg in np.arange(0, 60, 1):
            rot = np.radians(rot_deg)
            apothem_factor = sqrt3 / 2
            centroid = np.mean(inner_pts, axis=0)
            cx, cy = centroid
            
            a_needed = 0
            for i in range(6):
                theta = rot + i * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                dists = (inner_pts - np.array([cx, cy])) @ np.array([nx, ny])
                a_needed = max(a_needed, np.max(dists))
            
            s_needed = a_needed / apothem_factor
            if s_needed < best_s:
                best_s = s_needed
                best_cx = cx
                best_cy = cy
                best_rot = rot_deg
        
        return best_cx, best_cy, best_s, best_rot
    
    cx, cy, s, rot = compute_min_hexagon(all_verts)
    
    # Return: inner rows, outer center, side length, rotation
    return inner, (cx, cy), s, rot
# EVOLVE-BLOCK-END
