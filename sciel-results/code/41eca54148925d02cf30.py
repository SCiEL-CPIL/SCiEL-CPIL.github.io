"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product, combinations

N = 12

# EVOLVE-BLOCK-START
import math

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a
    ca, sa = math.cos(-a), math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # In local frame, hexagon with side s has vertices at angles 0,60,...,300
    # The hexagon is the intersection of 6 half-planes
    # Distance from center to each edge is s * sqrt(3) / 2
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        theta = i * math.pi / 3.0
        nx, ny = math.cos(theta), math.sin(theta)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def hex_hex_overlap(h1_center, h1_side, h1_angle, h2_center, h2_side, h2_angle, tol=1e-9):
    """Check if two hexagons overlap. Returns True if they overlap."""
    # Check if any vertex of h1 is inside h2 or vice versa
    v1 = hex_vertices(h1_center[0], h1_center[1], h1_side, h1_angle)
    v2 = hex_vertices(h2_center[0], h2_center[1], h2_side, h2_angle)
    
    for px, py in v1:
        if point_in_hex(px, py, h2_center[0], h2_center[1], h2_side, h2_angle):
            return True
    for px, py in v2:
        if point_in_hex(px, py, h1_center[0], h1_center[1], h1_side, h1_angle):
            return True
    
    # Also check edge intersections (approximate by checking midpoints)
    # For robustness, check if centers are close enough
    dist = math.hypot(h1_center[0] - h2_center[0], h1_center[1] - h2_center[1])
    min_dist = (h1_side + h2_side) * math.sqrt(3) / 2.0 - tol
    if dist < min_dist * 0.9:
        # More detailed check
        return True
    
    return False

def outer_hex_contains_inner(cx, cy, side, angle, ix, iy, iangle, tol=1e-9):
    """Check if inner hexagon (center ix,iy, side 1, angle iangle) is inside outer hexagon."""
    # Check all vertices of inner hex are inside outer
    v_inner = hex_vertices(ix, iy, 1.0, iangle)
    for px, py in v_inner:
        if not point_in_hex(px, py, cx, cy, side, angle):
            return False
    return True

def valid_configuration(inner, center, side, angle, tol=1e-9):
    """Check if configuration is valid: all inner hexes inside outer, no overlaps."""
    cx, cy = center
    # Check containment
    for ix, iy, iang in inner:
        if not outer_hex_contains_inner(cx, cy, side, angle, ix, iy, iang, tol):
            return False
    # Check no overlaps
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if hex_hex_overlap((inner[i][0], inner[i][1]), 1.0, inner[i][2],
                              (inner[j][0], inner[j][1]), 1.0, inner[j][2], tol):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_side = 100.0
    best_config = None
    
    # Strategy: Try hexagonal lattice arrangements
    # The optimal packing of 12 unit hexagons in a hexagon likely uses a honeycomb-like pattern
    
    # Approach 1: Place hexes in a hexagonal grid pattern, then optimize positions
    # Hexagonal lattice: neighbors are at distance sqrt(3) in 6 directions
    
    # Try various initial configurations and use local optimization
    
    # Configuration 1: 2-3-4-3-2 pattern (rows)
    # Row 0: 2 hexes
    # Row 1: 3 hexes  
    # Row 2: 4 hexes
    # Row 3: 3 hexes
    # Row 4: 2 hexes
    # Total: 14 - too many. Need 12.
    
    # Try 1-3-4-3-1 = 12
    # Or 2-3-4-3 = 12 (4 rows)
    
    # Let's try a centered arrangement
    
    # Generate candidate positions on a hexagonal lattice
    # Hex lattice spacing for unit hexes: distance between centers = sqrt(3)
    d = math.sqrt(3)
    
    # Create a grid of candidate positions
    # Use axial coordinates for hex grid
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Convert axial to pixel
            x = d * (q + r/2.0)
            y = d * r * math.sqrt(3)/2.0
            candidates.append((x, y))
    
    # For each candidate subset of 12, try to fit in a hexagon
    # This is combinatorially expensive, so use smarter approach
    
    # Better: Start with known good configurations and optimize
    
    # Configuration A: Honeycomb cluster
    # Center hex + 6 around + 5 more on second ring
    config_a = []
    # Center
    config_a.append((0.0, 0.0, 0.0))
    # First ring (6 hexes)
    for i in range(6):
        angle = i * math.pi / 3.0
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        config_a.append((x, y, 0.0))
    # Second ring: pick 5 more
    # Second ring positions are at distance 2*d from center
    # There are 12 positions in second ring
    for i in range(6):
        angle = i * math.pi / 3.0
        # These are the "corner" positions of second ring
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        config_a.append((x, y, 0.0))
        if len(config_a) >= 12:
            break
    
    # Trim to 12
    config_a = config_a[:12]
    
    # Try to find minimum enclosing hexagon for each configuration
    # and optimize
    
    def min_enclosing_hex(inner_positions, max_iter=100):
        """Find minimum side length hexagon containing all inner hexes."""
        cx = sum(p[0] for p in inner_positions) / len(inner_positions)
        cy = sum(p[1] for p in inner_positions) / len(inner_positions)
        
        best_side = 10.0
        best_angle = 0.0
        
        # Try different angles
        for angle in [0, 30, 60, 90, 120, 150]:
            # Binary search on side length
            lo, hi = 0.0, 20.0
            for _ in range(30):
                mid = (lo + hi) / 2.0
                if valid_config_containment(inner_positions, cx, cy, mid, angle):
                    hi = mid
                else:
                    lo = mid
            if hi < best_side:
                best_side = hi
                best_angle = angle
        
        return cx, cy, best_side, best_angle
    
    def valid_config_containment(inner_positions, cx, cy, side, angle):
        for ix, iy, iang in inner_positions:
            if not outer_hex_contains_inner(cx, cy, side, angle, ix, iy, iang):
                return False
        return True
    
    # Evaluate configuration A
    if len(config_a) == 12:
        if not hex_overlap_check(config_a):
            cx, cy, side, angle = min_enclosing_hex(config_a)
            if valid_configuration(config_a, (cx, cy), side, angle):
                best_side = side
                best_config = (config_a, (cx, cy), side, angle)
    
    def hex_overlap_check(config):
        for i in range(len(config)):
            for j in range(i+1, len(config)):
                if hex_hex_overlap((config[i][0], config[i][1]), 1.0, config[i][2],
                                  (config[j][0], config[j][1]), 1.0, config[j][2]):
                    return True
        return False
    
    # Try more configurations
    # Configuration B: Linear-ish arrangement
    for row_offset in range(-2, 3):
        config_b = []
        # 3 rows of 4, offset
        for row in range(3):
            for col in range(4):
                x = col * d + row * d/2.0 + row_offset * 0.1
                y = row * d * math.sqrt(3)/2.0
                config_b.append((x, y, 0.0))
        if len(config_b) == 12 and not hex_overlap_check(config_b):
            cx, cy, side, angle = min_enclosing_hex(config_b)
            if valid_configuration(config_b, (cx, cy), side, angle):
                if side < best_side:
                    best_side = side
                    best_config = (config_b, (cx, cy), side, angle)
    
    # Try optimized local search
    if best_config is not None:
        inner0, center0, side0, angle0 = best_config
        # Local optimization: perturb positions slightly
        best_local = best_side
        for trial in range(20):
            if time.time() > deadline:
                break
            # Random perturbation
            inner_perturbed = []
            for ix, iy, iang in inner0:
                inner_perturbed.append((ix + (np.random.random()-0.5)*0.1,
                                       iy + (np.random.random()-0.5)*0.1,
                                       iang + (np.random.random()-0.5)*5))
            if not hex_overlap_check(inner_perturbed):
                cx, cy, side, angle = min_enclosing_hex(inner_perturbed)
                if valid_configuration(inner_perturbed, (cx, cy), side, angle):
                    if side < best_local:
                        best_local = side
                        best_config = (inner_perturbed, (cx, cy), side, angle)
                        best_side = side
    
    # Final: try a few more structured configurations
    # Configuration C: Two clusters of 6
    config_c = []
    for cluster_center in [(-1.5*d, 0), (1.5*d, 0)]:
        cx_c, cy_c = cluster_center
        config_c.append((cx_c, cy_c, 0.0))
        for i in range(6):
            angle = i * math.pi / 3.0
            x = cx_c + d * math.cos(angle)
            y = cy_c + d * math.sin(angle)
            config_c.append((x, y, 0.0))
    config_c = config_c[:12]
    if len(config_c) == 12 and not hex_overlap_check(config_c):
        cx, cy, side, angle = min_enclosing_hex(config_c)
        if valid_configuration(config_c, (cx, cy), side, angle):
            if side < best_side:
                best_side = side
                best_config = (config_c, (cx, cy), side, angle)
    
    if best_config is None:
        # Fallback: spread out hexes
        inner = []
        for i in range(12):
            angle = i * math.pi / 6.0
            inner.append((2*d*math.cos(angle), 2*d*math.sin(angle), 0.0))
        cx, cy, side, angle = min_enclosing_hex(inner)
        return inner, (cx, cy), side, angle
    
    inner, center, side, angle = best_config
    return inner, center, side, angle

# EVOLVE-BLOCK-END
