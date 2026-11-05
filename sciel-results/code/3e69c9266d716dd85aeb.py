"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np
from itertools import product
import time

N = 12
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    theta = np.radians(angle_deg)
    angles = np.arange(6) * np.pi / 3 + theta
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    theta = -np.radians(angle_deg)
    lx = dx * np.cos(theta) - dy * np.sin(theta)
    ly = dx * np.sin(theta) + dy * np.cos(theta)
    # Regular hexagon with "radius" s (center to vertex)
    # In local frame, hexagon has vertices at angles 0, 60, ...
    # The hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, |x + y/sqrt(3)| <= s, etc.
    # Actually for a regular hexagon with circumradius s:
    # It's the intersection of 6 half-planes
    # Let's use: for a hex with flat-top orientation (vertex at angle 0):
    # vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The constraints are:
    # |lx| <= s
    # |ly| <= s*sqrt(3)/2  
    # |lx + ly/sqrt(3)| <= s  ... wait let me think more carefully
    
    # For a regular hexagon with circumradius R=s, the apothem is a = s*sqrt(3)/2
    # In the local frame with a vertex at angle 0:
    # The 6 edges are at distances a from center, with normals at angles 30, 90, 150, 210, 270, 330 degrees
    # Edge normal at 30 deg: (cos30, sin30) = (sqrt(3)/2, 1/2)
    # So constraint: lx*sqrt(3)/2 + ly*1/2 <= a
    # Similarly for other edges
    
    a = s * np.sqrt(3) / 2
    # Normals at 30, 90, 150, 210, 270, 330
    normals = np.array([[np.cos(np.radians(a_)), np.sin(np.radians(a_))] for a_ in [30, 90, 150, 210, 270, 330]])
    dots = normals @ np.array([lx, ly])
    return np.all(dots <= a + 1e-9)

def _hex_in_hex(inner_cx, inner_cy, inner_s, inner_angle, outer_cx, outer_cy, outer_s, outer_angle):
    """Check if inner hex is fully inside outer hex."""
    # Check all vertices of inner hex are inside outer hex
    verts = _hex_vertices(inner_cx, inner_cy, inner_s, inner_angle)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
            return False
    return True

def _hexes_disjoint(cx1, cy1, a1, cx2, cy2, a2, s1=1.0, s2=1.0):
    """Check if two hexagons are disjoint (no overlap). Use vertex-edge and edge-edge checks."""
    # Simple check: if all vertices of hex1 are outside hex2 AND all vertices of hex2 are outside hex1,
    # they might still overlap if one contains the other, but for disjoint unit hexes this is sufficient
    # Actually we need to check if they intersect. Let's use a simpler approach:
    # Check if any vertex of hex1 is inside hex2, or any vertex of hex2 is inside hex1
    # Also check edge-edge intersections
    
    v1 = _hex_vertices(cx1, cy1, s1, a1)
    v2 = _hex_vertices(cx2, cy2, s2, a2)
    
    # Check vertices of hex1 inside hex2
    for vx, vy in v1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False  # overlap
    # Check vertices of hex2 inside hex1
    for vx, vy in v2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False  # overlap
    # Check edge-edge intersections
    def seg_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)
        if abs(denom) < 1e-12:
            return False
        t = ((x3-x1)*(y4-y3) - (y3-y1)*(x4-x3)) / denom
        u = ((x3-x1)*(y2-y1) - (y3-y1)*(x2-x1)) / denom
        return 0 <= t <= 1 and 0 <= u <= 1
    
    for i in range(6):
        p1, p2 = v1[i], v1[(i+1)%6]
        for j in range(6):
            p3, p4 = v2[j], v2[(j+1)%6]
            if seg_intersect(p1, p2, p3, p4):
                return False  # overlap
    
    return True  # disjoint

def _min_outer_side(positions, angles):
    """Given inner hex positions and angles (all side=1), find minimal enclosing regular hexagon side length."""
    # We need to find the minimal side length S and position/rotation of a regular hexagon
    # that contains all inner hexagons.
    # This is a 4-parameter optimization: outer_cx, outer_cy, S, outer_angle
    # For simplicity, assume outer hexagon is centered at centroid of inner hexes and try various angles
    
    cx_mean = np.mean([p[0] for p in positions])
    cy_mean = np.mean([p[1] for p in positions])
    
    best_S = float('inf')
    best_params = None
    
    # Try various outer hexagon angles
    for outer_angle in range(0, 60, 5):  # 0 to 55 degrees
        # For a given outer angle, find minimal S such that all inner hex vertices are inside
        # The outer hexagon with center (cx_mean, cy_mean), side S, angle outer_angle
        # contains a point (px, py) if:
        # Transform to outer hex local frame, check all 6 half-plane constraints
        
        max_required = 0
        for pos, ang in zip(positions, angles):
            verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
            for vx, vy in verts:
                dx, dy = vx - cx_mean, vy - cy_mean
                theta = -np.radians(outer_angle)
                lx = dx * np.cos(theta) - dy * np.sin(theta)
                ly = dx * np.sin(theta) + dy * np.cos(theta)
                a = 1.0 * np.sqrt(3) / 2  # apothem of unit hex... wait, we need S such that apothem >= max distance
                # For outer hex with side S, apothem = S*sqrt(3)/2
                # The constraint is: for each normal direction, dot <= S*sqrt(3)/2
                normals = np.array([[np.cos(np.radians(a_)), np.sin(np.radians(a_))] for a_ in [30, 90, 150, 210, 270, 330]])
                dots = normals @ np.array([lx, ly])
                # We need S*sqrt(3)/2 >= max(dots)
                required_S = np.max(dots) / (np.sqrt(3) / 2)
                max_required = max(max_required, required_S)
        
        if max_required < best_S:
            best_S = max_required
            best_params = (cx_mean, cy_mean, outer_angle)
    
    # Also try optimizing the center
    # For a few promising angles, do a simple grid search on center
    for outer_angle in [best_params[2]]:
        for dcx in [-0.5, -0.25, 0, 0.25, 0.5]:
            for dcy in [-0.5, -0.25, 0, 0.25, 0.5]:
                cx_test = cx_mean + dcx
                cy_test = cy_mean + dcy
                max_required = 0
                for pos, ang in zip(positions, angles):
                    verts = _hex_vertices(pos[0], pos[1], 1.0, ang)
                    for vx, vy in verts:
                        dx, dy = vx - cx_test, vy - cy_test
                        theta = -np.radians(outer_angle)
                        lx = dx * np.cos(theta) - dy * np.sin(theta)
                        ly = dx * np.sin(theta) + dy * np.cos(theta)
                        normals = np.array([[np.cos(np.radians(a_)), np.sin(np.radians(a_))] for a_ in [30, 90, 150, 210, 270, 330]])
                        dots = normals @ np.array([lx, ly])
                        required_S = np.max(dots) / (np.sqrt(3) / 2)
                        max_required = max(max_required, required_S)
                if max_required < best_S:
                    best_S = max_required
                    best_params = (cx_test, cy_test, outer_angle)
    
    return best_S, best_params

def _is_valid_packing(positions, angles, outer_cx, outer_cy, outer_S, outer_angle):
    """Check if packing is valid: all inner hexes inside outer, all inner hexes disjoint."""
    # Check all inner hexes inside outer
    for pos, ang in zip(positions, angles):
        if not _hex_in_hex(pos[0], pos[1], 1.0, ang, outer_cx, outer_cy, outer_S, outer_angle):
            return False
    # Check all pairs disjoint
    for i in range(N):
        for j in range(i+1, N):
            if not _hexes_disjoint(positions[i][0], positions[i][1], angles[i], 
                                   positions[j][0], positions[j][1], angles[j]):
                return False
    return True

# Honeycomb lattice positions for 12 hexagons
# In a honeycomb lattice, hexagons at positions:
# Row-based: for row r, columns offset
# Let's use axial coordinates for hex grid

def _axial_to_pixel(q, r, size=1.0):
    """Convert axial coordinates to pixel coordinates for pointy-top hexagons."""
    x = size * np.sqrt(3) * (q + r/2)
    y = size * 3/2 * r
    return x, y

def _generate_honeycomb_config():
    """Generate 12 hex positions in a honeycomb-like arrangement."""
    # Try a 2-3-4-3-2 pattern or similar
    # Let's use axial coordinates
    # A compact arrangement for 12 hexes:
    # Center at (0,0), then surrounding
    
    configs = []
    
    # Config 1: 1 center + 6 ring + 5 more
    # Axial coords for 7-hex cluster: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 5 more: (2,0), (1,1), (0,2), (-1,2), (-2,1) ... let me think
    
    # Actually, let me just try several hand-crafted arrangements
    
    # Arrangement A: 4-3-4-1 or similar
    # Let me use a rectangular-ish grid with hex offset
    
    # Standard hex grid: rows of alternating length
    # Row 0: 4 hexes
    # Row 1: 3 hexes (offset)
    # Row 2: 4 hexes
    # Row 3: 1 hex
    # Total: 12
    
    # Or:
    # Row 0: 3
    # Row 1: 4
    # Row 2: 3
    # Row 3: 2
    # Total: 12
    
    # Let me try the most compact: a hexagonal-ish shape
    # Using axial coordinates (q, r):
    
    # 12-hex cluster options:
    # Option 1: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2), (4,2) - too spread
    
    # Let me just use pixel coordinates directly for a honeycomb pattern
    # With flat-top hexagons, neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    
    # Generate positions on honeycomb lattice
    sqrt3 = np.sqrt(3)
    
    # Try arrangement: 2 rows of 6, with offset
    # Row 0 (y=0): x = 0, sqrt3, 2*sqrt3, 3*sqrt3, 4*sqrt3, 5*sqrt3
    # Row 1 (y=1.5): x = sqrt3/2, 3*sqrt3/2, 5*sqrt3/2, 7*sqrt3/2, 9*sqrt3/2, 11*sqrt3/2
    # This gives a 2x6 rectangle-ish shape
    
    pos_A = []
    for i in range(6):
        pos_A.append((i * sqrt3, 0.0))
    for i in range(6):
        pos_A.append((sqrt3/2 + i * sqrt3, 1.5))
    
    # Try arrangement: 3-4-3-2
    pos_B = []
    # Row 0 (y=0): 3 hexes centered
    for i in range(3):
        pos_B.append((i * sqrt3 - sqrt3, 0.0))
    # Row 1 (y=1.5): 4 hexes
    for i in range(4):
        pos_B.append((i * sqrt3 - 1.5*sqrt3, 1.5))
    # Row 2 (y=3.0): 3 hexes
    for i in range(3):
        pos_B.append((i * sqrt3 - sqrt3, 3.0))
    # Row 3 (y=4.5): 2 hexes
    for i in range(2):
        pos_B.append((i * sqrt3 - 0.5*sqrt3, 4.5))
    
    # Try arrangement: 4-4-4 (3 rows of 4)
    pos_C = []
    for i in range(4):
        pos_C.append((i * sqrt3, 0.0))
    for i in range(4):
        pos_C.append((sqrt3/2 + i * sqrt3, 1.5))
    for i in range(4):
        pos_C.append((i * sqrt3, 3.0))
    
    # Try arrangement: 2-3-4-3 (diamond-ish)
    pos_D = []
    # Row 0 (y=0): 2
    for i in range(2):
        pos_D.append((i * sqrt3, 0.0))
    # Row 1 (y=1.5): 3
    for i in range(3):
        pos_D.append((sqrt3/2 + i * sqrt3, 1.5))
    # Row 2 (y=3.0): 4
    for i in range(4):
        pos_D.append((i * sqrt3, 3.0))
    # Row 3 (y=4.5): 3
    for i in range(3):
        pos_D.append((sqrt3/2 + i * sqrt3, 4.5))
    
    configs = [pos_A, pos_B, pos_C, pos_D]
    return configs

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Generate candidate configurations
    configs = _generate_honeycomb_config()
    
    # For each config, compute minimal enclosing hexagon
    for config in configs:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        positions = [list(p) for p in config]
        angles = [0.0] * N
        
        # Center the configuration
        cx_mean = np.mean([p[0] for p in positions])
        cy_mean = np.mean([p[1] for p in positions])
        positions = [[p[0] - cx_mean, p[1] - cy_mean] for p in positions]
        
        # Find minimal enclosing hexagon
        S, (ocx, ocy, oang) = _min_outer_side(positions, angles)
        
        # Check validity
        if _is_valid_packing(positions, angles, ocx, ocy, S, oang):
            if S < best_score:
                best_score = S
                inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
                best_result = (inner, (ocx, ocy), S, oang)
    
    # If we found a valid result, try to improve with local search
    if best_result is not None and time.time() - start_time < TIME_LIMIT - 30:
        inner, center, S, oang = best_result
        positions = [[p[0], p[1]] for p in inner]
        angles = [p[2] for p in inner]
        
        # Simple local perturbation
        import random
        random.seed(42)
        for iteration in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Perturb one position slightly
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            new_positions = [p[:] for p in positions]
            new_positions[idx][0] += dx
            new_positions[idx][1] += dy
            
            # Check if still disjoint
            valid = True
            for j in range(N):
                if j == idx:
                    continue
                if not _hexes_disjoint(new_positions[idx][0], new_positions[idx][1], angles[idx],
                                       positions[j][0], positions[j][1], angles[j]):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if all still inside outer hex
            ocx, ocy = center
            all_inside = True
            for pos, ang in zip(new_positions, angles):
                if not _hex_in_hex(pos[0], pos[1], 1.0, ang, ocx, ocy, S, oang):
                    all_inside = False
                    break
            
            if not all_inside:
                continue
            
            # Recompute minimal enclosing hexagon
            S_new, (ocx_new, ocy_new, oang_new) = _min_outer_side(new_positions, angles)
            
            if S_new < S - 1e-6:
                positions = new_positions
                S = S_new
                center = (ocx_new, ocy_new)
                oang = oang_new
    
    if best_result is None:
        # Fallback: return a valid but large solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-END
