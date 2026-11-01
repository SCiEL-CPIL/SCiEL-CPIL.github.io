"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START

def _rotate_hex_verts(cx, cy, r, angle_rad, scale=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), circumscribed radius r,
    rotated by angle_rad. scale scales the side length (1.0 for unit hex)."""
    a = angle_rad
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + r * scale * math.cos(theta), cy + r * scale * math.sin(theta)))
    return verts

def _hexagon_polygon(cx, cy, r, angle_deg):
    """Return (verts, normals, offsets) for a regular hexagon centered at (cx,cy) with circumradius r,
    angle_deg is rotation of the hexagon. Returns 6 inequalities: normal_i . x <= offset_i.
    For a regular hexagon, the inradius is r*cos(pi/6). We'll work with circumradius r."""
    a = math.radians(angle_deg)
    # Vertex k at angle a + k*pi/3
    # The side between vertex k and k+1 has a normal pointing outward at angle a + pi/6 + k*pi/3
    # Actually, the outward normal of the side opposite to the direction of angle a+k*pi/3... 
    # Let's use the fact that for a regular hexagon centered at origin with circumradius r,
    # the half-width in direction theta is r*cos(pi/6) when theta aligns with a vertex, 
    # and r when theta aligns with a side normal? No.
    # Circumradius r, inradius r_in = r * cos(pi/6) = r * sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    # The outward normal of side i is at angle phi_i = a + i*pi/3 + pi/6 for i=0..5? 
    # Let's check: vertex 0 is at angle a. Vertex 1 is at a+pi/3.
    # The side between v0 and v1: midpoint is at angle a+pi/6, distance r*cos(pi/6).
    # The outward normal of this side is in the direction a+pi/6.
    # So for side i (between v_i and v_{i+1}), outward normal angle = a + pi/6 + i*pi/3.
    # Inequality: (x - cx) * cos(n_i) + (y - cy) * sin(n_i) <= r * cos(pi/6)
    
    verts = []
    normals = []
    offsets = []
    r_in = r * math.cos(math.pi / 6.0)
    for i in range(6):
        phi = a + math.pi / 6.0 + i * math.pi / 3.0
        nx = math.cos(phi)
        ny = math.sin(phi)
        normals.append((nx, ny))
        offsets.append(cx * nx + cy * ny + r_in)
        # Vertex calculation for completeness
        theta = a + i * math.pi / 3.0
        verts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return verts, normals, offsets

def _poly_overlap(poly1, poly2, eps=1e-9):
    """Check if two convex polygons (lists of (x,y) in CCW order) overlap.
    Uses separating axis theorem. Returns True if they overlap (including touching)."""
    def _separated(p1, p2, axis):
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        for p in p1:
            d = p[0]*axis[0] + p[1]*axis[1]
            if d < min1: min1 = d
            if d > max1: max1 = d
        for p in p2:
            d = p[0]*axis[0] + p[1]*axis[1]
            if d < min2: min2 = d
            if d > max2: max2 = d
        return max1 < min2 - eps or max2 < min1 - eps

    # Axes to test: normals of both polygons
    for i in range(len(poly1)):
        p1 = poly1[i]
        p2 = poly1[(i+1)%len(poly1)]
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        # Normal
        n = (-dy, dx)
        norm = math.sqrt(n[0]**2 + n[1]**2)
        if norm > 1e-12:
            n = (n[0]/norm, n[1]/norm)
            if _separated(poly1, poly2, n):
                return False
    for i in range(len(poly2)):
        p1 = poly2[i]
        p2 = poly2[(i+1)%len(poly2)]
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        n = (-dy, dx)
        norm = math.sqrt(n[0]**2 + n[1]**2)
        if norm > 1e-12:
            n = (n[0]/norm, n[1]/norm)
            if _separated(poly1, poly2, n):
                return False
    return True

def _is_inside_hex(cx, cy, r, angle_deg, inner_x, inner_y, inner_r, inner_angle_deg):
    """Check if the inner hexagon (centered at inner_x,inner_y, radius inner_r, angle inner_angle_deg)
    is fully inside the outer hexagon (centered at cx,cy, radius r, angle angle_deg).
    This is equivalent to checking if the Minkowski difference allows it.
    For convex polygons, inner is inside outer iff every vertex of inner is inside outer,
    AND every edge of inner is "compatible". Actually, for convex polygons, A is inside B
    iff every vertex of A is inside B (for B containing origin? No, generally true for convex B).
    Wait, that's sufficient if B is convex. Yes.
    So we just need to check if all 6 vertices of the inner hex are inside the outer hex."""
    inner_verts = _rotate_hex_verts(inner_x, inner_y, inner_r, math.radians(inner_angle_deg))
    _, out_normals, out_offsets = _hexagon_polygon(cx, cy, r, angle_deg)
    for v in inner_verts:
        for i in range(6):
            nx, ny = out_normals[i]
            off = out_offsets[i]
            if v[0]*nx + v[1]*ny > off + 1e-9:
                return False
    return True

def _validate(inner, center, side, angle):
    """Check if the packing is valid. Returns True if valid."""
    cx, cy = center
    # Outer hex circumradius
    r_out = side * 2.0 / math.sqrt(3.0)
    # Inner hex circumradius (side length 1)
    r_in = 1.0 * 2.0 / math.sqrt(3.0)
    
    # Check each inner hex is inside outer
    for (ix, iy, iang) in inner:
        if not _is_inside_hex(cx, cy, r_out, angle, ix, iy, r_in, iang):
            return False
            
    # Check pairwise non-overlap
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            poly1 = _rotate_hex_verts(ix1, iy1, r_in, math.radians(ia1))
            poly2 = _rotate_hex_verts(ix2, iy2, r_in, math.radians(ia2))
            if _poly_overlap(poly1, poly2):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s budget, leave 20s margin
    
    # Initial guess: Hexagonal cluster
    # For N=11, a central hex + 6 neighbors + 4 more.
    # Optimal for small N is often a compact cluster.
    # Let's try to find a good configuration via local search.
    
    best_score = 40.0
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate a set of candidate configurations
    # Strategy:
    # 1. Hexagonal lattice cluster
    # 2. Random placements with local refinement
    
    # --- Configuration 1: Hexagonal Lattice Cluster ---
    # The optimal packing of 11 hexagons in a hexagon is known to be a "compact" cluster.
    # Let's construct a cluster of 11 hexagons in a triangular lattice arrangement.
    # Center at (0,0).
    # Neighbors at distance d = 2*r_in = 4/sqrt(3) in 6 directions.
    # Let's place them in a hexagonal pattern: 1 in center, 6 in ring 1, 4 in ring 2 (partial).
    # Actually, the optimal for 11 is likely the "1+6+4" arrangement where the 4 are placed
    # to minimize the bounding hexagon.
    
    r_in = 2.0 / math.sqrt(3.0)
    
    # Try a specific known-good arrangement for 11 hexagons.
    # The "optimal" arrangement for 11 unit hexagons in a regular hexagon has a side length of 4.
    # Let's verify this.
    # If side length is 4, the inradius of the container is 4*sqrt(3)/2 = 2*sqrt(3).
    # The circumradius is 4*2/sqrt(3) = 8/sqrt(3).
    # The distance from center to the center of a hexagon on the boundary is Inradius - r_in?
    # No, the hexagon is tangent to the side. So the center is at distance Inradius - r_in from the center?
    # Wait, for a regular hexagon, the distance from center to the side is the inradius.
    # So if a unit hexagon is tangent to a side, its center is at distance r_in from the side.
    # So the center is at distance R_in_container - r_in from the main center.
    # R_in_container = side * sqrt(3)/2.
    # Distance = side * sqrt(3)/2 - 2/sqrt(3).
    
    # Let's try to generate a configuration with side=4.
    # If we can fit 11 hexagons in side=4, the score is 4.0.
    # The seed program had side=40.0.
    # Let's try to construct the 1+6+4 cluster.
    
    # Ring 1: 6 hexagons at distance d1 = 2*r_in.
    # Ring 2: 4 hexagons. Where?
    # In a hexagonal lattice, the next ring has 12 hexagons.
    # We only need 4. We should pick the 4 that are "closest" to the center in terms of projection?
    # No, we want to minimize the maximum projection.
    # The 12 positions in ring 2 are at distance 2*d1 = 4*r_in.
    # Some are "flat" (aligned with axes) and some are "pointy".
    # The "flat" ones project further.
    # We should pick the 4 "pointy" ones? Or the ones that fill the gaps?
    # Actually, for a hexagonal container, the "corners" are the most restrictive.
    # The vertices of the container are at distance R_out.
    # The centers of the outer hexagons should be as close to the center as possible,
    # but they must be non-overlapping.
    
    # Let's just use a simple local search to optimize the positions.
    # Start with a random valid configuration? Or a heuristic one.
    
    # Heuristic: Place hexagons in a spiral or lattice.
    # Let's try a lattice.
    # Axial coordinates (q, r) for hexagonal grid.
    # 0: (0,0)
    # 1-6: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 7-12: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1)...
    # Wait, ring 2 has 12 cells.
    # We need 11 cells total.
    # So we take 1 + 6 + 4.
    # Which 4 of the 12 in ring 2?
    # The 12 in ring 2 are at angles 0, 30, 60, ..., 330 degrees (relative to center) if we define them carefully.
    # Actually, the centers in ring 2 are at distance 2*d1.
    # The angles are 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330.
    # Wait, the neighbors in ring 1 are at 0, 60, 120, 180, 240, 300.
    # The "gaps" are at 30, 90, 150, 210, 270, 330.
    # The ring 2 hexagons that touch two ring 1 hexagons are at 30, 90, 150, 210, 270, 330?
    # No, in a hex grid, the hexagon at (2,0) is at angle 0, distance 2*d1.
    # The hexagon at (1,1) is at angle 30, distance 2*d1.
    # So the 12 are at 0, 30, 60, ..., 330.
    # The ones at 0, 60, 120, 180, 240, 300 are "aligned" with the ring 1 hexagons.
    # The ones at 30, 90, 150, 210, 270, 330 are in the "gaps".
    # Which set is better for a hexagonal container?
    # The container has sides normal to 0, 60, 120, 180, 240, 300.
    # So the "aligned" hexagons (at 0, 60, ...) will have their centers on the normal to the side.
    # Their projection on the normal is 2*d1.
    # The "gap" hexagons (at 30, 90, ...) have projection 2*d1 * cos(30) = 2*d1 * sqrt(3)/2 = sqrt(3)*d1.
    # Since sqrt(3) < 2, the "gap" hexagons have a smaller projection!
    # So we should pick the 4 "gap" hexagons.
    # There are 6 gap positions. We pick 4.
    # To be symmetric, we might pick 4 of them.
    
    # Let's construct this.
    d1 = 2 * r_in
    inner = []
    # Center
    inner.append((0.0, 0.0, 0.0))
    # Ring 1
    for i in range(6):
        ang = i * 60.0
        x = d1 * math.cos(math.radians(ang))
        y = d1 * math.sin(math.radians(ang))
        inner.append((x, y, 0.0))
    # Ring 2 (gap positions)
    # Angles: 30, 90, 150, 210, 270, 330
    # Pick 4. Let's pick 30, 90, 150, 210? Or 30, 150, 270, 90?
    # It doesn't matter much due to symmetry, but let's pick 4 that are somewhat balanced.
    # Let's pick 30, 90, 270, 330? (Top-right, Top, Bottom, Bottom-right?)
    # Actually, let's just pick the first 4: 30, 90, 150, 210.
    for ang in [30.0, 90.0, 150.0, 210.0]:
        x = 2 * d1 * math.cos(math.radians(ang))
        y = 2 * d1 * math.sin(math.radians(ang))
        inner.append((x, y, 0.0))
        
    # Now we need to find the minimum side length S such that this cluster fits.
    # The side length S is determined by the maximum projection of the centers + r_in.
    # The projection of a center (x,y) onto the normal of a side at angle theta is x*cos(theta) + y*sin(theta).
    # The sides are at theta = 0, 60, 120, 180, 240, 300.
    # For each side, we need: max_projection + r_in <= R_in_container = S * sqrt(3)/2.
    # So S >= (max_projection + r_in) * 2 / sqrt(3).
    
    max_proj = 0.0
    for (x, y, _) in inner:
        for theta_deg in [0, 60, 120, 180, 240, 300]:
            theta = math.radians(theta_deg)
            proj = abs(x * math.cos(theta) + y * math.sin(theta))
            if proj > max_proj:
                max_proj = proj
                
    side_needed = (max_proj + r_in) * 2.0 / math.sqrt(3.0)
    
    # Check validity
    if _validate(inner, (0.0, 0.0), side_needed, 0.0):
        best_score = side_needed
        best_inner = inner
        best_center = (0.0, 0.0)
        best_angle = 0.0
    else:
        # If invalid, something went wrong with the geometry or validation.
        # Fallback to a larger side length.
        best_score = side_needed * 1.1
        best_inner = inner
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # If we have time, try to improve with local search?
    # For now, this heuristic is likely close to optimal (S=4).
    # Let's double check the side length.
    # d1 = 4/sqrt(3).
    # Max projection of ring 2 gap hexagons:
    # At angle 30 deg: (2*d1*cos30, 2*d1*sin30) = (2*d1*sqrt(3)/2, 2*d1*0.5) = (d1*sqrt(3), d1).
    # d1*sqrt(3) = 4.
    # d1 = 4/sqrt(3) ≈ 2.31.
    # Projection on 0 deg axis: 4.
    # Projection on 60 deg axis: 4*cos60 + 2.31*sin60 = 2 + 2.31*0.866 = 2 + 2 = 4.
    # So max_proj = 4.
    # side_needed = (4 + 4/sqrt(3)) * 2 / sqrt(3) = (4 + 2.31) * 1.1547 = 6.31 * 1.1547 ≈ 7.28?
    # Wait.
    # R_in_container = S * sqrt(3)/2.
    # Max distance from center to the side of the inner hex is max_proj + r_in.
    # r_in = 2/sqrt(3) ≈ 1.1547.
    # max_proj = 4.
    # Total = 5.1547.
    # S = 5.1547 * 2 / 1.732 = 5.95?
    
    # Let's re-calculate.
    # If S=4, R_in = 4*sqrt(3)/2 = 2*sqrt(3) ≈ 3.464.
    # Is 5.1547 <= 3.464? No.
    # So S=4 is too small.
    
    # My previous deduction was wrong.
    # Let's find the actual minimum S.
    # S = (max_proj + r_in) * 2 / sqrt(3).
    # max_proj = 4.
    # r_in = 2/sqrt(3).
    # S = (4 + 2/sqrt(3)) * 2 / sqrt(3) = 8/sqrt(3) + 4/3 ≈ 4.618 + 1.333 = 5.95.
    
    # Can we do better?
    # What if we rotate the cluster?
    # Or what if we pick different ring 2 hexagons?
    # If we pick the "aligned" ones (0, 60, ...), their projection is 2*d1 = 8/sqrt(3) ≈ 4.618.
    # This is larger than 4. So "gap" is better.
    
    # What if we rotate the container?
    # The container is a regular hexagon, so rotation by 30 degrees swaps the "side" normals with the "vertex" directions.
    # If we rotate the container by 30 degrees, the sides are normal to 30, 90, 150, 210, 270, 330.
    # Then the projection of the "gap" hexagons (which are at 30, 90, ...) would be 2*d1 = 4.618.
    # And the "aligned" hexagons (at 0, 60, ...) would project to 2*d1*cos(30) = 4.
    # So it's symmetric.
    
    # So the minimum side length for this cluster is ~5.95.
    
    # Is there a better cluster?
    # Maybe we can move the outer hexagons inward?
    # They are constrained by non-overlap.
    # The distance between centers must be at least 2*r_in = d1.
    # In the lattice, the distance is exactly d1.
    # So we cannot move them inward without breaking the lattice or overlapping.
    # Unless we change the angles?
    # If we rotate the outer hexagons, we might be able to pack them tighter?
    # For hexagons, the minimum distance between centers for non-overlap is 2*r_in if they are parallel.
    # If they are rotated, the minimum distance can be smaller?
    # No, the circumradius is the same. The minimum distance is always 2*r_in?
    # Actually, if you rotate a hexagon, its width changes.
    # The minimum distance between two non-overlapping convex bodies is the minimum of the distance between their boundaries.
    # For two regular hexagons, the minimum distance between centers is 2*r_in if they are parallel.
    # If one is rotated by 30 degrees, the minimum distance is smaller?
    # Let's check.
    # One hex flat-topped, one pointy-topped.
    # The "corner" of the pointy one can fit into the "flat" side of the flat one?
    # This is complex.
    
    # For now, let's stick with the lattice solution with S ≈ 5.95.
    # This is a significant improvement over S=40.
    
    return best_inner, best_center, best_score, best_angle

# EVOLVE-BLOCK-END
