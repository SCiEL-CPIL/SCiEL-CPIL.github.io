import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Hexagon geometry constants
    # Unit hexagon (side 1) has width (flat-to-flat) = sqrt(3)
    # Height (point-to-point) = 2
    # In a hexagonal lattice of flat-top hexagons:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Offset for alternating rows: sqrt(3)/2
    
    SQRT3 = math.sqrt(3)
    
    # Generate a set of candidate lattice positions
    # We need enough to choose N=11 from
    # A cluster of 19 hexagons (center + 6 + 12) is a good candidate pool
    # Let's generate positions for a larger area to be safe
    candidates = []
    # Axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    max_radius = 3 # Radius in hex grid steps
    for r in range(-max_radius, max_radius + 1):
        q_min = max(-max_radius, -r - max_radius)
        q_max = min(max_radius, -r + max_radius)
        for q in range(q_min, q_max + 1):
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            candidates.append((x, y))
            
    # Sort candidates by distance from origin to prefer central packing
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # We only need to consider the closest ~20-30 positions for N=11
    # Taking the first 25 ensures we have plenty of options for a compact shape
    candidate_positions = candidates[:25]
    
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """
        Find minimum S such that all inner hexagons fit in outer hexagon.
        Inner hexagons are unit side, aligned with outer (angle 0 relative to outer).
        This means they are all flat-top if outer is flat-top.
        
        The constraint is that for each of the 6 side normals of the outer hex,
        the max projection of any inner vertex onto that normal (relative to outer center)
        must be <= S * apothem.
        
        Since inner hexagons are aligned, their vertices are at fixed offsets from their centers.
        """
        inner_verts = []
        for (ix, iy) in inner:
            # Vertices of a unit flat-top hexagon centered at (ix, iy)
            # Angles: 0, 60, 120, 180, 240, 300
            # x = ix + cos(theta), y = iy + sin(theta)
            verts = []
            for i in range(6):
                theta = math.radians(60 * i)
                verts.append((ix + math.cos(theta), iy + math.sin(theta)))
            inner_verts.append(verts)
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        
        # Outer hexagon is flat-top (angle 0)
        # Normals are at 30, 90, 150, 210, 270, 330 degrees
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    # Projection relative to outer center (0,0)
                    proj = v[0] * nx + v[1] * ny
                    if proj > max_proj:
                        max_proj = proj
            # Need S * apothem_factor >= max_proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    # Enumerate subsets of candidate positions
    # N=11 is small enough that we can be smart about it.
    # The optimal packing for 11 hexagons is likely a hexagonal cluster of radius 2 (19 hexes)
    # with 8 removed, or a specific shape.
    # Actually, the densest packing of 11 hexagons is known.
    # Let's just check the first K candidates (closest to center) and try to find a valid set.
    
    # A very strong heuristic: The optimal arrangement is a subset of the closest positions.
    # Let's try all combinations of 11 from the first 15 closest positions?
    # C(15, 11) = C(15, 4) = 1365. This is very fast.
    
    from itertools import combinations
    
    # Take the 15 closest positions to origin
    top_candidates = candidate_positions[:15]
    
    found = False
    for combo in combinations(range(len(top_candidates)), N):
        if time.time() > deadline:
            break
            
        inner_pos = [top_candidates[i] for i in combo]
        
        # Calculate the required outer side length for this configuration
        # Outer center is (0,0), angle 0
        S = compute_outer_hex(inner_pos, 0.0, 0.0, 0.0)
        
        if S < best_S:
            best_S = S
            best_inner = [[x, y, 0.0] for x, y in inner_pos]
            best_center = (0.0, 0.0)
            best_angle = 0.0
            found = True

    # If we didn't find anything (shouldn't happen), fallback
    if not found:
        # Just place them in a line
        best_inner = [[k * SQRT3, 0.0, 0.0] for k in range(-N//2, N - N//2 + 1)]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_S = compute_outer_hex(best_inner, 0.0, 0.0, 0.0)

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
