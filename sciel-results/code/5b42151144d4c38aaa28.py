import numpy as np
import time
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    # Rotate point by -angle
    cos_a = np.cos(-a)
    sin_a = np.sin(-a)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The hexagon can be described by |lx| <= s and |ly| <= s*sqrt(3)/2 and |lx + ly*sqrt(3)/2| <= s*sqrt(3)/2 etc.
    # Actually, a regular hexagon with side s has vertices at distance s from center
    # The condition for a point to be inside: max of projections on the 3 axes <= s*sqrt(3)/2
    # The 3 axes are at 0, 60, 120 degrees in the rotated frame
    # Projection on axis at angle phi: lx*cos(phi) + ly*sin(phi)
    inradius = side * np.sqrt(3) / 2.0
    for phi in [0, np.pi/3, 2*np.pi/3]:
        proj = lx * np.cos(phi) + ly * np.sin(phi)
        if abs(proj) > inradius + 1e-10:
            return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    ocx, ocy = outer_center
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (px, py), ang in zip(inner_positions, inner_angles):
        verts = _hex_vertices(px, py, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For each vertex, find the minimum side length needed
    # A point (px, py) is inside a hexagon centered at (ocx, ocy) with side s and angle a
    # if its projection on each of the 3 axes is <= s*sqrt(3)/2
    # The 3 axes directions (perpendicular to edges) are at angles a, a+60, a+120
    a = np.radians(outer_angle)
    max_proj = 0.0
    for phi in [a, a + np.pi/3, a + 2*np.pi/3]:
        # Project all vertices onto this axis
        dx = all_verts[:, 0] - ocx
        dy = all_verts[:, 1] - ocy
        projs = dx * np.cos(phi) + dy * np.sin(phi)
        max_proj = max(max_proj, np.max(np.abs(projs)))
    
    # max_proj <= s * sqrt(3)/2, so s >= max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def _check_validity(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are fully inside the outer hexagon."""
    ocx, ocy = outer_center
    for (px, py), ang in zip(inner_positions, inner_angles):
        verts = _hex_vertices(px, py, ang)
        for vx, vy in verts:
            if not _point_in_hex(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    return True

def _initial_hexagonal_packing(n):
    """Create initial hexagonal packing for n hexagons."""
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # For n=11, use a partial hexagonal arrangement
    # Center + 6 around + 4 more
    positions = [(0.0, 0.0)]
    angles = [0.0]
    
    # Ring 1: 6 hexagons at distance 2 (since unit hexagons touch at distance 2*sqrt(3)/2*... wait)
    # Actually, for unit hexagons (side=1), the distance between centers of touching hexagons is 2*sqrt(3)/2 = sqrt(3)? 
    # No. The inradius is sqrt(3)/2, so two hexagons touch when center distance = 2*inradius = sqrt(3).
    # Wait, let me reconsider. A regular hexagon with side s has:
    # - Circumradius (center to vertex) = s
    # - Inradius (center to edge midpoint) = s*sqrt(3)/2
    # Two hexagons with side 1 touch when the distance between centers equals 2 * inradius = sqrt(3)
    # if they are aligned, or the distance depends on orientation.
    
    # For simplicity, use a triangular lattice with spacing sqrt(3)
    d = np.sqrt(3)
    
    # Add ring 1: 6 hexagons
    for i in range(6):
        ang = i * np.pi / 3.0
        x = d * np.cos(ang)
        y = d * np.sin(ang)
        positions.append((x, y))
        angles.append(0.0)
    
    # Add ring 2 partial: need 4 more to reach 11
    # Ring 2 has 12 positions, pick 4
    for i in [0, 1, 2, 3]:  # Pick 4 from ring 2
        ang = i * np.pi / 3.0
        x = 2 * d * np.cos(ang)
        y = 2 * d * np.sin(ang)
        positions.append((x, y))
        angles.append(0.0)
    
    return positions, angles

def _local_search(positions, angles, max_iters=500, step_size=0.1):
    """Simple local search to minimize outer side length."""
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    best_side = _min_outer_side(positions, angles)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(positions)):
            for dx in [-step_size, 0.0, step_size]:
                for dy in [-step_size, 0.0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_positions = [p[:] for p in positions]
                    new_positions[i] = (positions[i][0] + dx, positions[i][1] + dy)
                    side = _min_outer_side(new_positions, angles)
                    if side < best_side - 1e-10:
                        best_side = side
                        best_positions = new_positions
                        best_angles = angles[:]
                        improved = True
                        # Update current positions
                        positions = new_positions
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try reducing step size
            step_size *= 0.7
            if step_size < 0.001:
                break
    
    return best_positions, best_angles, best_side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    # Initial hexagonal packing
    positions, angles = _initial_hexagonal_packing(N)
    
    # Compute initial side
    side = _min_outer_side(positions, angles)
    
    # Local search
    positions, angles, side = _local_search(positions, angles, max_iters=300, step_size=0.2)
    
    if time.time() > deadline:
        # Return best found so far
        pass
    
    # Center the outer hexagon at origin
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Format inner as list of [x, y, angle_degrees]
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    
    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END
