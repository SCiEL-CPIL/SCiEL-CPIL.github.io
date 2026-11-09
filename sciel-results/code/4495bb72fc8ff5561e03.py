"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.spatial import cKDTree

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle."""
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # For a regular hexagon with circumradius R=side, the inradius is R*sqrt(3)/2.
        # The 6 supporting lines are at angles 30, 90, 150, 210, 270, 330 degrees relative to local x-axis.
        # Constraint: dot((lx,ly), normal_k) <= inradius
        inradius = side * math.sqrt(3) / 2
        max_dot = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= inradius + 1e-9

    def outer_hexagon_contains_inner(px, py, angle_deg, ocx, ocy, oangle_deg, oside):
        """Check if inner hexagon at (px,py) with angle is inside outer hexagon."""
        verts = hexagon_vertices(px, py, angle_deg)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ocx, ocy, oangle_deg, oside):
                return False
        return True

    def check_valid(inner, ocx, ocy, oside, oangle_deg):
        """Check if all inner hexagons are inside outer and don't overlap."""
        # Check containment
        for x, y, a in inner:
            if not outer_hexagon_contains_inner(x, y, a, ocx, ocy, oangle_deg, oside):
                return False
        
        # Check non-overlap
        # For same orientation (angle 0), min distance is 2.0.
        # We will assume all inner hexagons are at angle 0 for simplicity in this version,
        # or we can just check distance >= 2.0 which is conservative for any rotation?
        # Actually, if rotations differ, min distance can be less than 2.0.
        # But for this initial improvement, let's stick to angle 0 for inner hexagons
        # as it allows tight packing in a honeycomb grid.
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Conservative: require distance >= 2.0
                if dist < 2.0 - 1e-6:
                    return False
        return True

    def compute_min_outer_hexagon(positions, angle_deg=0.0):
        """
        Compute the minimum side length of a regular hexagon (with given orientation)
        centered at the centroid of positions to contain all positions (as points).
        Note: This treats the inner hexagons as points. We must add the 'radius' of the inner hexagon.
        The 'radius' of a unit hexagon in the direction of the outer hexagon's normals varies.
        A safe bound is the circumradius of the inner hexagon, which is 1.0.
        However, we can be more precise.
        
        Let's find the minimal scaling factor s such that if we scale the outer hexagon (centered at centroid)
        by s, it contains all points.
        Then the side length is s * (max distance to side in unit hexagon)?
        No, the outer hexagon side length S has inradius S*sqrt(3)/2.
        The constraint for a point p relative to center c is:
        max_k dot(p-c, n_k) <= S * sqrt(3)/2
        So S >= max_k dot(p-c, n_k) * 2 / sqrt(3) for all p.
        So S = max_p max_k dot(p-c, n_k) * 2 / sqrt(3).
        
        Then we need to ensure the INNER hexagons (size 1) are inside.
        The inner hexagon extends to distance 1 from its center in the direction of its vertices.
        The outer hexagon boundary is defined by normals.
        The distance from the inner center to the outer boundary in direction n_k is S*sqrt(3)/2 - dot(p-c, n_k).
        The inner hexagon vertex in direction n_k is at distance 1 (if aligned) or less.
        Actually, the support function of the inner hexagon in direction n_k is 1 * cos(angle_diff)?
        For a regular hexagon with circumradius 1, the support function h(theta) is:
        h(theta) = cos(theta - theta_0) * something?
        No, for a regular polygon with circumradius R, the support function is R * cos( (theta - theta_vertex) mod 60deg - 30deg )?
        Let's just use the bounding circle for safety first: add 1.0 to the computed S?
        No, that's too loose.
        
        Let's use the property:
        The inner hexagon is contained in the outer hexagon iff for all 6 normals n_k of the outer hexagon,
        max_{v in inner_hex} dot(v, n_k) <= dot(center, n_k) + S*sqrt(3)/2.
        max_{v in inner_hex} dot(v, n_k) = dot(inner_center, n_k) + support_inner(n_k).
        So we need: dot(inner_center, n_k) + support_inner(n_k) <= dot(center, n_k) + S*sqrt(3)/2.
        Rearranging: S >= (dot(inner_center - center, n_k) + support_inner(n_k)) * 2 / sqrt(3).
        
        We need this to hold for all inner hexagons and all 6 normals.
        So S = max_{i, k} ( dot(p_i - c, n_k) + support_inner(n_k) ) * 2 / sqrt(3).
        
        What is support_inner(n_k)?
        The inner hexagon has angle 0 (vertices at 0, 60, ...).
        The normal n_k is at angle 30 + 60k.
        The angle between the normal and the nearest vertex of the inner hexagon is 30 degrees.
        The support function of a regular hexagon (circumradius 1) in a direction 30 degrees away from a vertex is:
        The vertex is at (1, 0). The normal is at (cos(30), sin(30)).
        dot((1,0), (cos30, sin30)) = cos(30) = sqrt(3)/2.
        Is there a point further?
        The edge adjacent to vertex 0 goes to vertex 60.
        Midpoint of edge is at angle 30, distance sqrt(3)/2.
        dot(midpoint, normal) = (sqrt(3)/2 * cos(30) + 0) = 3/4? No.
        Midpoint coords: (sqrt(3)/2 * cos(30), sqrt(3)/2 * sin(30)) = (3/4, sqrt(3)/4).
        Normal: (sqrt(3)/2, 1/2).
        Dot: (3/4)(sqrt(3)/2) + (sqrt(3)/4)(1/2) = 3sqrt(3)/8 + sqrt(3)/8 = 4sqrt(3)/8 = sqrt(3)/2.
        So the support is sqrt(3)/2 for all normals at 30, 90, etc.
        Wait, the normals of the OUTER hexagon are at 30, 90, ...
        The INNER hexagon vertices are at 0, 60, ...
        So the angle difference is always 30 degrees (or 90, etc, which is also 30 from a vertex? No, 90 is 30 from 60 and 120).
        Yes, the support function of a unit hexagon in any direction is constant?
        No, that's only if it's a circle.
        For a regular hexagon, the support function varies between inradius (sqrt(3)/2) and circumradius (1).
        The inradius is the support in the direction of the edge normal (30, 90...).
        The circumradius is the support in the direction of the vertex (0, 60...).
        Here, the outer normals are at 30, 90... which correspond to the edge normals of the inner hexagon.
        So support_inner(n_k) = sqrt(3)/2.
        
        So S = max_{i, k} ( dot(p_i - c, n_k) + sqrt(3)/2 ) * 2 / sqrt(3).
        """
        if not positions:
            return 0.0, (0.0, 0.0)
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        max_val = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            for x, y in positions:
                val = (x - cx) * nx + (y - cy) * ny
                if val > max_val:
                    max_val = val
        
        # S = (max_val + sqrt(3)/2) * 2 / sqrt(3)
        S = (max_val + math.sqrt(3) / 2) * 2 / math.sqrt(3)
        return S, (cx, cy)

    # Generate candidate positions in a hexagonal grid pattern
    # Triangular lattice with spacing 2.0
    # x = 2*c + (r%2)*1, y = r*sqrt(3)
    max_radius = 10
    positions = []
    for r in range(-max_radius, max_radius + 1):
        for c in range(-max_radius, max_radius + 1):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * math.sqrt(3)
            if x * x + y * y <= (max_radius * 2) ** 2:
                positions.append((x, y))
    
    # Sort by distance from origin
    positions.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])

    best_result = None
    best_score = float('inf')

    import random
    random.seed(42)

    # Strategy 1: Greedy placement from grid
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        placed = []
        # Start with a random point near the center to ensure compactness?
        # Or just start with the closest to origin.
        # Let's try starting with the origin if it's in the list, else closest.
        # Origin is (0,0) for r=0, c=0.
        
        # We want a compact cluster.
        # Let's pick the first N positions from the sorted list?
        # That would be the most compact set of points on the lattice.
        # Let's try that first.
        
        candidate_positions = positions.copy()
        if attempt % 2 == 0:
            # Try the N closest points to origin
            placed = candidate_positions[:N]
        else:
            # Try random perturbation / random selection with distance check
            # This is slower, so we rely on the "closest" strategy mostly.
            # For now, just use closest.
            placed = candidate_positions[:N]
        
        if len(placed) < N:
            continue
            
        # Check non-overlap (should be fine for lattice)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = placed[i][0] - placed[j][0]
                dy = placed[i][1] - placed[j][1]
                if math.sqrt(dx*dx + dy*dy) < 2.0 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        oside, (ocx, ocy) = compute_min_outer_hexagon(placed)
        
        # Verify with the full geometry check (slow, so only if promising)
        if oside < best_score:
            inner = [[x, y, 0.0] for x, y in placed]
            if check_valid(inner, ocx, ocy, oside, 0.0):
                best_score = oside
                best_result = (inner, (ocx, ocy), oside, 0.0)

    # Strategy 2: Local optimization / shifting
    # If we found a result, try to shift the cluster to reduce the bounding box?
    # The centroid is fixed by the points. The orientation of the outer hexagon is fixed at 0.
    # We could try rotating the outer hexagon?
    # The task allows the outer hexagon to have any rotation.
    # My compute_min_outer_hexagon assumes outer angle 0.
    # Let's try a few outer angles.
    
    if best_result is not None:
        inner, (ocx, ocy), oside, oangle = best_result
        # Try rotating the outer hexagon
        for oangle_try in [0, 30, 60, 90, 120, 150]:
            # Recompute min side for this outer angle
            # We need to re-evaluate S for this orientation.
            # S = max_{i,k} ( dot(p_i - c, n_k) + support_inner(n_k) ) * 2 / sqrt(3)
            # n_k depends on oangle_try.
            # support_inner(n_k) depends on the angle between n_k and inner hexagon vertices.
            
            # Let's just use check_valid with a slightly larger S and binary search?
            # Or just compute the required S directly.
            
            max_val = 0
            for k in range(6):
                theta = math.radians(oangle_try + 30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                
                # Calculate support of inner hexagon in direction (nx, ny)
                # Inner hexagon angle is 0.
                # Angle of normal is phi = oangle_try + 30 + 60k.
                # Vertices at 0, 60, ...
                # Support is max dot(v, n) for v in inner hexagon.
                # This is 1 * cos(delta) where delta is the angular distance to the nearest vertex?
                # No, for a regular polygon, support function h(theta) = R cos( (theta - theta_0) mod 60 - 30 )?
                # Let's just compute it numerically for the 6 vertices and 6 edge midpoints?
                # Actually, max dot(v, n) for v in convex polygon is at a vertex.
                # So we check the 6 vertices of the inner hexagon.
                inner_verts = hexagon_vertices(0, 0, 0.0)
                max_dot_inner = 0
                for vx, vy in inner_verts:
                    d = vx * nx + vy * ny
                    if d > max_dot_inner:
                        max_dot_inner = d
                
                for x, y in inner:
                    val = (x - ocx) * nx + (y - ocy) * ny + max_dot_inner
                    if val > max_val:
                        max_val = val
            
            new_side = max_val * 2 / math.sqrt(3)
            
            if new_side < oside:
                # Verify
                if check_valid(inner, ocx, ocy, new_side, oangle_try):
                    oside = new_side
                    oangle = oangle_try
                    best_result = (inner, (ocx, ocy), oside, oangle)
                    best_score = oside

    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
