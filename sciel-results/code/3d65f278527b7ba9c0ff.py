"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # Unit regular hexagon vertices (flat-top orientation, side length 1)
    def hex_vertices(cx, cy, angle_deg=0.0):
        angle_rad = math.radians(angle_deg)
        # Flat-top hexagon: vertices at 0, 60, 120, 180, 240, 300 degrees
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            verts.append((cx + math.cos(a), cy + math.sin(a)))
        return np.array(verts)
    
    # Check if a point is inside a regular hexagon centered at (cx,cy) with side length s
    def point_in_hexagon(px, py, cx, cy, s, angle_deg=0.0):
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        # Rotate point by -angle to get into hexagon's frame
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # For a flat-top regular hexagon with side s, the vertices are at:
        # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # The hexagon can be described as: |x| <= s and |x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2 and |x/2 - y*sqrt(3)/2| <= s*sqrt(3)/2
        # Actually, for a regular hexagon with "radius" (center to vertex) = s:
        # The three pairs of parallel sides are at distances s*sqrt(3)/2 from center
        # Let me use: the hexagon is the intersection of 6 half-planes
        # For flat-top: normals at 0, 60, 120, 180, 240, 300 degrees
        # Distance from center to each side = s * sqrt(3) / 2
        d = s * math.sqrt(3) / 2
        for i in range(6):
            a = i * math.pi / 3
            nx = math.cos(a)
            ny = math.sin(a)
            if lx * nx + ly * ny > d:
                return False
        return True
    
    # Check if two hexagons overlap (approximate by checking if any vertex of one is inside the other,
    # and if any vertex of the other is inside the first)
    def hexagons_overlap(verts1, verts2):
        # Check if any vertex of hex1 is inside hex2, and vice versa
        for v in verts1:
            if point_in_hexagon(v[0], v[1], np.mean(verts2[:,0]), np.mean(verts2[:,1]), 1.0, 0.0):
                # Need to check with proper center and angle
                pass
        # Simpler: check all 6 vertices of each hex against the other
        c1 = np.mean(verts1, axis=0)
        c2 = np.mean(verts2, axis=0)
        for v in verts1:
            if point_in_hexagon(v[0], v[1], c2[0], c2[1], 1.0, 0.0):
                return True
        for v in verts2:
            if point_in_hexagon(v[0], v[1], c1[0], c1[1], 1.0, 0.0):
                return True
        return False
    
    # Better overlap check: two unit hexagons overlap if center distance < some threshold
    # For two identical regular hexagons with the same orientation, they don't overlap if
    # center distance >= sqrt(3) (distance between centers of edge-sharing hexagons)
    # But with rotation, it's more complex. Let's use a conservative check.
    
    # For the initial design, let's assume all inner hexagons have the same orientation (0 degrees)
    # and use a hexagonal lattice with spacing sqrt(3)
    
    # Generate hexagonal lattice points for a compact arrangement of 11 hexagons
    # Hex lattice: use axial coordinates
    # A hexagon of "radius" r has 1 + 3*r*(r+1) hexagons
    # r=1: 7, r=2: 19. So for 11, use r=1 (7) plus 4 more
    
    # Let me use a different approach: try several candidate arrangements and pick the best
    
    # Arrangement 1: 3-4-3-4-3 pattern won't work for 11. Let's do 2-3-4-3-2 = 14, too many
    # 1-3-5-3-1 = 13, too many
    # Let's do: center, 6 around, then 4 in the next ring
    # Or: a line of 11 (bad), or a hexagon-like shape
    
    # Best compact arrangement for 11: take the 7-hexagon hexagon and add 4 adjacent to it
    # Hex lattice coordinates (axial): (q, r) where q + r + s = 0
    # Neighbors of (0,0): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    # Convert axial to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    def axial_to_cartesian(q, r, spacing=math.sqrt(3)):
        x = spacing * (q + r / 2.0)
        y = spacing * 0.75 * r  # 3/2 * r * (spacing/sqrt(3))... let me recalculate
        # For pointy-top hexagons with side length 1:
        # x = sqrt(3) * (q + r/2), y = 3/2 * r
        # But our hexagons are flat-top. Let me just use the standard hex lattice.
        # For flat-top hexagons with side 1, center-to-center distance for adjacent = sqrt(3)
        # Axial coords: x = sqrt(3)*(q + r/2), y = (3/2)*r * (2/sqrt(3)) ... 
        # Let me just use: x = sqrt(3)*(q + r/2), y = 3*r/2
        return (spacing * (q + r / 2.0), 1.5 * r)
    
    # The 7 hexagons in a hexagon of radius 1:
    # (0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    base_7 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    # Add 4 more: extend in some direction
    # Options: (2,0), (2,-1), (2,-2), or (0,-2), (-1,-1), etc.
    # Let's try adding (2,0), (2,-1), (2,-2), (1,-2) to make a more compact shape
    # Or add them symmetrically: (2,0), (2,-1), (-2,0), (-2,1) - but that's spread out
    
    # Let me try: base 7 + (2,0), (2,-1), (0,-2), (-1,-1)
    candidate1 = base_7 + [(2,0), (2,-1), (0,-2), (-1,-1)]
    
    # Another: base 7 + (2,0), (1,-2), (0,-2), (-1,-1)
    candidate2 = base_7 + [(2,0), (1,-2), (0,-2), (-1,-1)]
    
    # Another: a 3-4-3-4 pattern won't give 11. Let's try 2-3-4-3-2 = 14, no.
    # How about just a hexagon of radius 2 (19 hexes) minus 8? Too complex.
    
    # Let me try multiple candidates and evaluate
    
    candidates = [
        base_7 + [(2,0), (2,-1), (0,-2), (-1,-1)],
        base_7 + [(2,0), (1,-2), (0,-2), (-1,-1)],
        base_7 + [(2,0), (2,-1), (2,-2), (1,-2)],
        base_7 + [(0,-2), (-1,-1), (-2,0), (-2,1)],  # extend downward
        base_7 + [(2,0), (2,-1), (-2,0), (-2,1)],  # extend left and right
    ]
    
    def evaluate_arrangement(axial_points, cluster_angle_deg=0.0, outer_angle_deg=0.0):
        """Given axial coords of inner hex centers, cluster rotation, and outer hex angle,
        find the minimum outer side length s such that all inner hexagons fit inside."""
        
        # Convert to Cartesian
        centers = []
        for q, r in axial_points:
            x, y = axial_to_cartesian(q, r)
            centers.append((x, y))
        
        # Rotate centers by cluster_angle
        ca = math.radians(cluster_angle_deg)
        cos_ca, sin_ca = math.cos(ca), math.sin(ca)
        rotated_centers = []
        for cx, cy in centers:
            rx = cx * cos_ca - cy * sin_ca
            ry = cx * sin_ca + cy * cos_ca
            rotated_centers.append((rx, ry))
        
        # For a given outer side length s and outer angle, check if all inner hexagons fit
        # Inner hexagons have side 1, angle = cluster_angle (same as cluster rotation)
        # Outer hexagon has side s, angle = outer_angle, centered at (0,0)
        
        # Binary search on s
        def fits(s, outer_angle_deg):
            oa = math.radians(outer_angle_deg)
            # Outer hexagon: center (0,0), side s, angle outer_angle
            # For each inner hex, check all 6 vertices are inside outer hex
            for ix, iy in rotated_centers:
                # Inner hex has side 1, angle = cluster_angle_deg
                inner_angle = cluster_angle_deg
                for vi in range(6):
                    va = math.radians(inner_angle) + vi * math.pi / 3
                    vx = ix + math.cos(va)
                    vy = iy + math.sin(va)
                    # Check if (vx, vy) is inside outer hex
                    if not point_in_hexagon(vx, vy, 0.0, 0.0, s, outer_angle_deg):
                        return False
            return True
        
        # Binary search for minimum s
        lo = 0.0
        hi = 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if fits(mid, outer_angle_deg):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Try different cluster angles and outer angles
    best_s = float('inf')
    best_config = None
    
    for cand_idx, cand in enumerate(candidates):
        # Try various cluster rotations
        for cluster_angle in [0, 15, 30, 45, 60, 75, 90]:
            # Try various outer rotations
            for outer_angle in [0, 15, 30, 45, 60, 75, 90]:
                s = evaluate_arrangement(cand, cluster_angle, outer_angle)
                if s < best_s:
                    best_s = s
                    best_config = (cand_idx, cluster_angle, outer_angle)
        
        if time.time() > deadline:
            break
    
    if best_config is None:
        # Fallback: just return the line arrangement with a reasonable bound
        inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    cand_idx, cluster_angle, outer_angle = best_config
    cand = candidates[cand_idx]
    
    # Build inner list
    inner = []
    for q, r in cand:
        x, y = axial_to_cartesian(q, r)
        # Rotate by cluster_angle
        ca = math.radians(cluster_angle)
        rx = x * math.cos(ca) - y * math.sin(ca)
        ry = x * math.sin(ca) + y * math.cos(ca)
        inner.append([rx, ry, float(cluster_angle)])
    
    return inner, (0.0, 0.0), float(best_s), float(outer_angle)
# EVOLVE-BLOCK-END

