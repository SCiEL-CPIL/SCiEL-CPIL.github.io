"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from math import sqrt, pi, cos, sin, ceil, floor

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, s, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon of side s centered at (cx,cy) rotated by angle_deg."""
    a = angle_deg * pi / 180.0
    verts = []
    for i in range(6):
        theta = a + i * pi / 3.0
        x = cx + s * cos(theta)
        y = cy + s * sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, s, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s."""
    a = angle_deg * pi / 180.0
    dx = px - cx
    dy = py - cy
    # Transform to hexagon's local frame
    x = dx * cos(a) + dy * sin(a)
    y = -dx * sin(a) + dy * cos(a)
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # Width = 2s, height = sqrt(3)*s
    # For flat-top hexagon:
    #   if |y| > s*sqrt(3)/2: outside
    #   elif |y| > s*sqrt(3)/2 - 0: check x bounds
    # Actually let me use the standard check:
    # A regular hexagon with side s has vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # In local coords:
    h = s * sqrt(3) / 2.0  # half height
    # For |y| <= h: x range is [-s, s]
    # For h < |y| <= s*sqrt(3): not possible since max y is h
    # Wait, the hexagon extends from y=-h to y=h
    if abs(y) > h + 1e-10:
        return False
    if abs(x) <= s - 1e-10:
        return True
    # For the side regions:
    # When y > 0: x bound is s - y/sqrt(3)  (for the right side)
    # When y < 0: x bound is s + y/sqrt(3)  (for the right side)
    if y >= 0:
        x_max = s - y / sqrt(3)
    else:
        x_max = s + y / sqrt(3)
    return abs(x) <= x_max + 1e-10

def _hexes_disjoint(v1, v2, s1, s2):
    """Check if two hexagons are disjoint (no overlapping interior)."""
    # Simple check: if distance between centers > s1 + s2, they're disjoint
    # But this is only sufficient, not necessary. For a more accurate check,
    # we'd need to check all edge intersections. For now, use center distance.
    cx1, cy1 = v1[0]
    cx2, cy2 = v2[0]
    dx = cx1 - cx2
    dy = cy1 - cy2
    dist = sqrt(dx*dx + dy*dy)
    return dist >= s1 + s2 - 1e-9

def _all_inside_outer(inner_pts, outer_cx, outer_cy, outer_s, outer_angle):
    """Check if all inner hexagon centers are inside the outer hexagon (with margin)."""
    for (px, py, pa) in inner_pts:
        if not _point_in_hex(px, py, outer_cx, outer_cy, outer_s, outer_angle):
            return False
    return True

def _validate(inner, outer_cx, outer_cy, outer_s, outer_angle):
    """Validate the packing: all inner hexagons inside outer, and disjoint from each other."""
    # Check all inside outer
    if not _all_inside_outer(inner, outer_cx, outer_cy, outer_s, outer_angle):
        return False
    # Check pairwise disjoint
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            dx = xi - xj
            dy = yi - yj
            dist = sqrt(dx*dx + dy*dy)
            if dist < 2.0 - 1e-9:  # two unit hexagons need center distance >= 2
                return False
    return True

def _min_side_length(inner, outer_cx, outer_cy, outer_angle):
    """Find minimum outer side length such that all inner hexagons fit inside."""
    # The outer hexagon must contain all inner hexagon vertices
    # For each inner hexagon, compute the "radius" needed in the outer hexagon's frame
    # A unit hexagon centered at (px,py) with angle pa: its vertices extend at most 1 from center
    # So we need a hexagon centered at (outer_cx, outer_cy) that contains all vertices of all inner hexagons
    # 
    # For a hexagon with side s, the maximum distance from center to any point inside is s (at vertices)
    # and the inradius is s*sqrt(3)/2
    # 
    # A point (px,py) is inside a hexagon centered at (cx,cy) with side s and angle a iff
    # in the local frame, |y| <= s*sqrt(3)/2 and |x| <= s - |y|/sqrt(3) (for |y|<=s*sqrt(3)/2)
    #
    # For each point, the required side length is:
    #   s >= |y_local| / (sqrt(3)/2)  and  s >= |x_local| + |y_local|/sqrt(3)
    #
    # Take the max over all points.
    
    a = outer_angle * pi / 180.0
    ca = cos(a)
    sa = sin(a)
    
    max_s = 0.0
    for (px, py, pa) in inner:
        # Check all 6 vertices of this inner hexagon
        for k in range(6):
            theta = pa * pi / 180.0 + k * pi / 3.0
            vx = px + cos(theta)
            vy = py + sin(theta)
            # Transform to outer hexagon's local frame
            dx = vx - outer_cx
            dy = vy - outer_cy
            x_local = dx * ca + dy * sa
            y_local = -dx * sa + dy * ca
            # Required s for this point:
            h = abs(y_local)
            # s must satisfy: h <= s*sqrt(3)/2  =>  s >= h*2/sqrt(3)
            # and: |x_local| <= s - h/sqrt(3)  =>  s >= |x_local| + h/sqrt(3)
            s1 = h * 2.0 / sqrt(3)
            s2 = abs(x_local) + h / sqrt(3)
            max_s = max(max_s, s1, s2)
    
    return max_s

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Strategy: Use a hexagonal lattice arrangement of unit hexagons
    # Unit hexagons in a hexagonal lattice: centers at
    #   (i * 3/2, j * sqrt(3) + (i mod 2) * sqrt(3)/2) for integer i,j
    # Wait, for unit hexagons (side 1), the center-to-center distance in the
    # hexagonal lattice is:
    #   horizontal: 3/2 (for adjacent columns)
    #   Actually, for regular hexagons of side 1, the distance between centers
    #   of adjacent hexagons in a honeycomb is sqrt(3) (for flat-top) or 3/2 (pointy-top)
    #
    # Let me think more carefully. A regular hexagon with side 1 has:
    #   width (flat-to-flat) = sqrt(3)
    #   height (vertex-to-vertex) = 2
    #
    # For a hexagonal tiling with flat-top hexagons:
    #   Column spacing: 3/2 * s = 3/2
    #   Row spacing: sqrt(3) * s = sqrt(3)
    #   Odd columns offset by sqrt(3)/2
    #
    # For pointy-top:
    #   Row spacing: 3/2
    #   Column spacing: sqrt(3)
    #   Odd rows offset by sqrt(3)/2
    #
    # The minimal enclosing hexagon for a cluster of unit hexagons in a hexagonal
    # lattice arrangement can be computed.
    #
    # For N=11, a good arrangement is a "flower" pattern:
    #   1 center + 6 around it + 4 more
    #   Or: a 3x3 block minus 2 corners, etc.
    #
    # Let me try the standard hexagonal number arrangement:
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # 11 is not a hexagonal number, so we need a non-perfect hexagonal cluster.
    #
    # For N=11, one good arrangement:
    #   Row 0 (center): 3 hexagons
    #   Row 1 (above): 4 hexagons
    #   Row 2 (above): 4 hexagons
    # Total: 3+4+4 = 11
    #
    # Or the "compact" arrangement:
    #   Center row: 4 hexagons
    #   Above: 3 hexagons
    #   Below: 3 hexagons
    #   Top: 1
    #   Bottom: 1
    # Total: 4+3+3+1+1 = 12 (too many)
    #
    # Let me try: a 3-wide by 3-tall hexagonal patch minus 2 corners.
    #
    # Actually, let me just generate positions on a hexagonal lattice and find
    # the minimum enclosing hexagon.
    
    # Generate candidate positions on a hexagonal lattice
    # For flat-top unit hexagons:
    #   (i * 1.5, j * sqrt(3) + (i % 2) * sqrt(3)/2) for integer i, j
    #   But we need to be careful with the modulo for negative i.
    
    # Let me generate a grid of candidate center positions
    # and pick N of them that minimize the enclosing hexagon side length.
    
    # First, generate a reasonable range of lattice points
    range_size = 5  # -4 to 4 in each direction
    
    candidates = []
    for i in range(-range_size, range_size + 1):
        for j in range(-range_size, range_size + 1):
            x = i * 1.5
            y = j * sqrt(3) + (i % 2) * (sqrt(3) / 2) if i >= 0 else j * sqrt(3) + ((i % 2) * (sqrt(3) / 2) if i % 2 != 0 else 0)
            # Actually, let me use a cleaner formula
            x = i * 1.5
            # For hexagonal lattice with flat-top:
            # y offset depends on column parity
            if i % 2 == 0:
                y = j * sqrt(3)
            else:
                y = j * sqrt(3) + sqrt(3) / 2
            candidates.append((x, y, 0.0))
    
    # Remove duplicates (in case of floating point issues)
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = (round(c[0], 6), round(c[1], 6))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Sort by distance from origin to prioritize central positions
    unique_candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
    
    # Now, for N=11, we want to pick 11 positions that minimize the enclosing hexagon.
    # A greedy approach: start from center, add the next closest position that keeps
    # all hexagons disjoint.
    
    selected = [unique_candidates[0]]  # start with origin
    remaining = unique_candidates[1:]
    
    for _ in range(N - 1):
        best_candidate = None
        best_dist = float('inf')
        for c in remaining:
            # Check if this candidate is disjoint from all selected
            disjoint = True
            for s in selected:
                dx = c[0] - s[0]
                dy = c[1] - s[1]
                dist = sqrt(dx*dx + dy*dy)
                if dist < 2.0 - 1e-9:
                    disjoint = False
                    break
            if disjoint:
                # Prefer candidates that keep the cluster compact
                # Use max distance from centroid as a heuristic
                cx = sum(s[0] for s in selected) / len(selected)
                cy = sum(s[1] for s in selected) / len(selected)
                d = sqrt((c[0]-cx)**2 + (c[1]-cy)**2)
                if d < best_dist:
                    best_dist = d
                    best_candidate = c
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    inner = selected
    
    # Compute the minimum enclosing hexagon
    # Try different outer orientations
    best_side = float('inf')
    best_outer_angle = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # The outer hexagon center should be near the centroid of the inner hexagons
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    
    # Try a range of outer angles
    for outer_angle_deg in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
        side = _min_side_length(inner, cx, cy, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg
    
    # Also try shifting the outer center slightly
    for dcx in [-0.5, 0.0, 0.5]:
        for dcy in [-0.5, 0.0, 0.5]:
            test_cx = cx + dcx
            test_cy = cy + dcy
            for outer_angle_deg in [0, 30, 60, 90]:
                side = _min_side_length(inner, test_cx, test_cy, outer_angle_deg)
                if side < best_side:
                    best_side = side
                    best_outer_angle = outer_angle_deg
                    best_cx = test_cx
                    best_cy = test_cy
    
    # Ensure best_side is at least a reasonable value
    if best_side < 2.0:
        best_side = 2.0
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return what we have
        pass
    
    return inner, (best_cx, best_cy), best_side, best_outer_angle

# EVOLVE-BLOCK-END
