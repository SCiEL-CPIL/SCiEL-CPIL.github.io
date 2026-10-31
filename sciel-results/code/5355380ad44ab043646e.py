import numpy as np
from itertools import product
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def hex_support_function(cx, cy, angle_deg, direction):
    """Support function of a unit hexagon at (cx,cy) with rotation angle_deg in direction."""
    angle = np.radians(angle_deg)
    # Vertices of unit hexagon (side=1) centered at origin
    base_angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    verts = np.column_stack([np.cos(base_angles + angle), np.sin(base_angles + angle)])
    dots = verts @ direction
    return cx * direction[0] + cy * direction[1] + np.max(dots)

def compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle_deg):
    """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
    outer_angle = np.radians(outer_angle_deg)
    # The 6 outward normals of the outer hexagon
    normal_angles = outer_angle + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each normal direction, find the max extent of all inner hexagons
    max_extent = 0.0
    for i, (pos, ang) in enumerate(zip(inner_positions, inner_angles)):
        for n in normals:
            h = hex_support_function(pos[0], pos[1], ang, n)
            if h > max_extent:
                max_extent = h
    
    # The outer hexagon centered at outer_center with side s has support function
    # in direction n: outer_center dot n + s * (apothem factor)
    # For a regular hexagon with side s, the apothem (distance from center to side) is s * sqrt(3)/2
    # But the support function in the normal direction is s * sqrt(3)/2
    # Actually, for a regular hexagon with side s, the distance from center to each side is s*sqrt(3)/2
    # So support in normal direction = center dot n + s * sqrt(3)/2
    
    # We need: for all normals n, max_inner_extent(n) <= outer_center dot n + s * sqrt(3)/2
    # So s >= (max_inner_extent(n) - outer_center dot n) / (sqrt(3)/2) for all n
    # s = max over all n of (max_inner_extent(n) - outer_center dot n) / (sqrt(3)/2)
    
    oc = np.array(outer_center)
    min_s = 0.0
    for n in normals:
        required = (max_extent - oc @ n) / (np.sqrt(3) / 2)
        if required > min_s:
            min_s = required
    
    return min_s

def generate_ring_arrangement(n_inner):
    """Generate a ring arrangement: center hex + rings around it."""
    positions = [[0.0, 0.0]]
    angles = [0.0]
    
    # Ring 1: 6 hexagons around center
    # Distance between centers of adjacent unit hexagons = 2 * apothem = sqrt(3)
    # Actually, for hexagons with side 1, the distance from center to vertex is 1
    # Distance from center to side (apothem) = sqrt(3)/2
    # For two adjacent hexagons sharing an edge, center distance = 2 * sqrt(3)/2 = sqrt(3)
    
    ring1_dist = np.sqrt(3)
    for i in range(6):
        theta = i * np.pi / 3
        pos = [ring1_dist * np.cos(theta), ring1_dist * np.sin(theta)]
        positions.append(pos)
        angles.append(0.0)
    
    if len(positions) >= n_inner:
        return positions[:n_inner], angles[:n_inner]
    
    # Ring 2: add more hexagons
    # In a hexagonal packing, ring 2 has 12 positions
    # Positions at distance 2*sqrt(3) in directions 0, pi/3, ... and at sqrt(3) in directions pi/6, pi/2, ...
    # Actually, let me think more carefully.
    
    # For hexagonal tiling, the lattice points are at:
    # (i * 3/2 + j * 3/4, j * 3*sqrt(3)/4) for integer i, j... no that's not right.
    
    # Let me use the axial coordinate system for hexagons.
    # In a hex grid, neighbors of (0,0) are at 6 positions.
    # The second ring has 12 positions.
    
    # Actually, let me just enumerate candidate positions in a hex grid pattern.
    # Using offset coordinates:
    # Row 0: (0,0)
    # Row 1: (-1, 1), (0, 1), (1, 1)  -- but this depends on orientation
    
    # Let me use a different approach: generate positions on a hex grid
    # and pick the n_inner closest to origin.
    
    # Hex grid positions using axial coordinates (q, r):
    # Cartesian: x = q + r/2, y = r * sqrt(3)/2
    # But for our hexagons with flat-top orientation...
    
    # Let me just generate candidate positions and sort by distance.
    all_positions = [[0.0, 0.0]]
    
    # Generate positions in a hex grid pattern up to radius 3
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Axial to cartesian for pointy-top hexagons
            x = (q + r / 2.0) * np.sqrt(3)  # scaled for unit side hexagons
            y = r * 1.5  # Hmm, this doesn't look right either
            
            # Let me use the correct formula.
            # For a hex grid with unit side length hexagons:
            # If hexagons are "flat-top", the centers are spaced:
            # Horizontal: 3/2 * side * 2 = 3 (for adjacent columns)
            # No wait, let me think again.
            
            # For a regular hexagon with side s, the distance from center to vertex is s.
            # The distance from center to edge (apothem) is s * sqrt(3)/2.
            # For two hexagons sharing an edge, center distance = 2 * apothem = s * sqrt(3).
            
            # In a hex grid, if we orient with flat top:
            # The 6 neighbors of (0,0) are at angles 0, 60, 120, 180, 240, 300 degrees
            # at distance sqrt(3) (for s=1).
            
            # So let me just use the 6-neighbor direction approach.
            pass
    
    # Simpler: just generate all positions that are integer combinations of the two basis vectors
    # Basis vectors for hex grid (flat-top, unit side):
    # v1 = (sqrt(3), 0)  -- horizontal neighbor
    # v2 = (sqrt(3)/2, 3/2)  -- 60 degree neighbor
    # Wait, that's not quite right either.
    
    # Let me just use the fact that the 6 neighbors of origin are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate all positions i*v1 + j*v2 for i,j in range(-3, 4)
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            pos = i * v1 + j * v2
            candidates.append(pos)
    
    # Sort by distance from origin
    candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Take the n_inner closest
    for pos in candidates:
        if len(positions) >= n_inner:
            break
        if np.linalg.norm(pos) < 1e-9 and len(positions) == 1:
            continue  # Already have origin
        # Check if this position is already in our list
        is_new = True
        for existing in positions:
            if np.linalg.norm(np.array(existing) - pos) < 1e-6:
                is_new = False
                break
        if is_new:
            positions.append([pos[0], pos[1]])
            angles.append(0.0)
    
    return positions[:n_inner], angles[:n_inner]

def optimize_outer_rotation(inner_positions, inner_angles, outer_center, n_steps=360):
    """Find the outer hexagon rotation that minimizes side length."""
    best_side = float('inf')
    best_angle = 0.0
    for k in range(n_steps):
        angle = k * 360.0 / n_steps
        side = compute_outer_side(inner_positions, inner_angles, outer_center, angle)
        if side < best_side:
            best_side = side
            best_angle = angle
    return best_side, best_angle

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = float('inf')
    best_result = None
    
    # Try ring arrangement
    positions, angles = generate_ring_arrangement(N)
    
    # Optimize outer rotation
    outer_center = (0.0, 0.0)
    side, angle = optimize_outer_rotation(positions, angles, outer_center, n_steps=720)
    
    if side < best_score:
        best_score = side
        best_result = (positions, outer_center, side, angle)
    
    # Try with some inner rotations to see if it helps
    # For simplicity, keep all inner hexagons at 0 degrees for now
    # The key optimization is the outer hexagon rotation
    
    # Check time
    if time.time() > deadline:
        if best_result is not None:
            inner = [[p[0], p[1], a] for p, a in zip(best_result[0], best_result[1])]
            return inner, best_result[1], best_result[2], best_result[3]
    
    # Try shifting the outer center slightly
    for dx in [-0.5, 0.0, 0.5]:
        for dy in [-0.5, 0.0, 0.5]:
            if time.time() > deadline:
                break
            oc = (dx, dy)
            side, angle = optimize_outer_rotation(positions, angles, oc, n_steps=360)
            if side < best_score:
                best_score = side
                best_result = (positions, oc, side, angle)
    
    if best_result is not None:
        inner = [[p[0], p[1], a] for p, a in zip(best_result[0], best_result[1])]
        return inner, best_result[1], best_result[2], best_result[3]
    
    # Fallback: return the ring arrangement with a reasonable outer hexagon
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, (0.0, 0.0), side, angle
# EVOLVE-BLOCK-END
