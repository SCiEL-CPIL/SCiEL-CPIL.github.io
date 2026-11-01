import math
import time
import random
import numpy as np
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute hex vertices for side 1, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
    
    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(verts_a, verts_b):
        # Fast check: center distance
        cx_a = sum(v[0] for v in verts_a) / 6
        cy_a = sum(v[1] for v in verts_a) / 6
        cx_b = sum(v[0] for v in verts_b) / 6
        cy_b = sum(v[1] for v in verts_b) / 6
        dist_sq = (cx_a - cx_b)**2 + (cy_a - cy_b)**2
        # Minimum distance between centers of disjoint unit hexagons is sqrt(3)
        # If dist < sqrt(3), they MUST overlap.
        if dist_sq < 3.0 - 1e-9:
            return False
        # If dist is large, they are disjoint.
        # Max distance where they can touch is 2 (vertex to vertex) + 2 (vertex to vertex)?
        # Actually, max distance between centers of touching hexagons is 2 (flat to flat) or sqrt(3) (point to point)?
        # Wait. 
        # Flat-to-flat distance: 2 * apothem = sqrt(3) ~ 1.732
        # Point-to-point distance: 2 * radius = 2
        # So if dist > 2.0, they are definitely disjoint.
        if dist_sq > 4.0 + 1e-9:
            return True
        
        # Exact check
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        S = compute_outer_hex(inner, ocx, ocy, oangle)
        if all_hexagons_fit(inner, ocx, ocy, S, oangle):
            if S < best_S:
                best_S = S
                best_inner = [list(x) for x in inner]
                best_center = (ocx, ocy)
                best_angle = oangle
                best_valid = True
                return True
        return False

    # Lattice generation
    # Triangular lattice spacing s.
    # For unit hexagons, ideal spacing is sqrt(3).
    # We use a slightly smaller spacing to allow denser packing if rotations help,
    # but since we are restricted to lattice positions, we must ensure disjointness.
    # If we use spacing sqrt(3), adjacent hexagons on the lattice are disjoint (touching).
    # We can use this as a base.
    
    def generate_lattice_points(max_radius, spacing):
        points = []
        # a1 = (spacing, 0)
        # a2 = (spacing/2, spacing*sqrt(3)/2)
        a1 = (spacing, 0.0)
        a2 = (spacing * 0.5, spacing * math.sqrt(3) / 2.0)
        
        range_r = int(max_radius / spacing) + 2
        for i in range(-range_r, range_r + 1):
            for j in range(-range_r, range_r + 1):
                x = i * a1[0] + j * a2[0]
                y = i * a1[1] + j * a2[1]
                if x*x + y*y <= max_radius*max_radius:
                    points.append((x, y))
        
        # Sort by distance from origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        return points

    def greedy_packing(points, N, seed, max_backtracks=1000):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        rng = random.Random(seed)
        
        # We will try to place N hexagons.
        # We assume angle 0 for all hexagons initially to speed up disjoint check.
        # Rotations can be introduced later or we can try a few fixed angles.
        
        # To make it a "search", we will shuffle the points slightly or pick a subset.
        # Actually, let's use a recursive backtracking with a limit.
        
        # Precompute vertices for all candidate points? Too much memory.
        # Compute on the fly.
        
        # We want to pack as compactly as possible.
        # So we should prefer points closer to the center.
        
        # Let's try a few different lattice spacings and angles.
        
        # Strategy: 
        # 1. Pick a random subset of lattice points (or all)
        # 2. Sort by distance
        # 3. Greedily pick the first available point that doesn't conflict.
        # 4. If we run out of points, backtrack.
        
        # To avoid exponential blowup, we limit the branching factor or depth.
        # Given N=11, and a dense lattice, greedy usually works well if we start from center.
        
        # Let's try multiple "views" of the lattice.
        
        # 1. Standard lattice, angle 0
        # 2. Rotated lattice (by 30 deg)?
        # 3. Different spacings?
        
        # For now, let's stick to spacing = sqrt(3) * 0.95 to allow some "sliding" if we add jitter later?
        # No, the task says "construct" returns specific coordinates.
        # If we use a lattice, the coordinates are fixed.
        # We can add a small local search step afterwards.
        
        # Let's generate a grid of candidate centers.
        # We'll use a "super-lattice" approach:
        # Try several lattice spacings and rotations.
        
        spacings = [math.sqrt(3) * 0.9, math.sqrt(3) * 0.95, math.sqrt(3) * 1.0, math.sqrt(3) * 1.05]
        lattice_angles = [0, 15, 30, 45, 60]
        
        for sp in spacings:
            if time.time() > deadline:
                return
            for la in lattice_angles:
                if time.time() > deadline:
                    return
                
                # Generate points
                # Max radius estimate: side length of outer hex ~ 3-4?
                # 11 hexagons. Area ~ 11 * 2.598 = 28.6.
                # Outer hex area = (3*sqrt(3)/2) * S^2 ~ 2.6 * S^2.
                # 2.6 S^2 > 28.6 => S^2 > 11 => S > 3.3.
                # So radius ~ 3.3 + 1 (inner hex radius) = 4.3.
                max_r = 5.0
                points = generate_lattice_points(max_r, sp)
                
                # Rotate points around origin
                if la != 0:
                    a = math.radians(la)
                    ca, sa = math.cos(a), math.sin(a)
                    rotated_points = []
                    for x, y in points:
                        rx = x * ca - y * sa
                        ry = x * sa + y * ca
                        rotated_points.append((rx, ry))
                    points = rotated_points
                    points.sort(key=lambda p: p[0]**2 + p[1]**2)
                
                # Greedy placement
                # We try to place N hexagons.
                # We allow a small "jitter" or we stick to exact lattice?
                # Exact lattice is faster.
                
                # We'll do a beam search or simple greedy with random restarts on the point order.
                
                # To improve on pure greedy, we can try swapping points.
                # But first, let's see if simple greedy works.
                
                # We'll run a few randomizations of the point order (perturbed by distance).
                for restart in range(20):
                    if time.time() > deadline:
                        return
                    
                    # Perturb the sorting key to break ties and explore different clusters
                    perturbed_points = []
                    for x, y in points:
                        # Add small noise to distance to randomize order of nearby points
                        dist = math.sqrt(x*x + y*y) + rng.uniform(0, 0.5)
                        perturbed_points.append((dist, x, y))
                    
                    perturbed_points.sort(key=lambda t: t[0])
                    
                    # Greedy fill
                    placed = []
                    placed_verts = []
                    
                    for dist, x, y in perturbed_points:
                        if len(placed) == N:
                            break
                        if time.time() > deadline:
                            return
                        
                        # Check if (x, y) is disjoint from all placed
                        # We assume angle 0 for inner hexagons for speed.
                        # We can try a few angles for the inner hexagons if we want.
                        # Let's stick to angle 0 for now, as the lattice is symmetric.
                        v_new = transform_verts(x, y, 1.0, 0.0)
                        
                        is_disjoint = True
                        for v_placed in placed_verts:
                            if not hexagons_disjoint(v_new, v_placed):
                                is_disjoint = False
                                break
                        
                        if is_disjoint:
                            placed.append([x, y, 0.0])
                            placed_verts.append(v_new)
                    
                    if len(placed) == N:
                        # Evaluate
                        # Try a few outer angles
                        for oangle in [0, 30, 60]:
                            update_best(placed, 0.0, 0.0, oangle)
                            
                            # Local search: try to move a hexagon to a nearby lattice point to shrink
                            # Or just try to remove the outermost hexagon and replace with a closer one?
                            # Simple local search: for each hex, try moving to a neighbor.
                            # This might be too slow. Let's just update best.
                            
                            # Actually, we can try to "compress" by checking if we can swap a point 
                            # with a closer unused point.
                            # But the greedy already picked the closest available.
                            # The issue is that picking a close point might block 3 others.
                            # Backtracking would solve this, but is expensive.
                            
                            # Let's try a simple "shake":
                            # Pick a random placed hex, remove it, try to place it back or elsewhere.
                            # This is effectively a random walk on the configuration space.
                            
                            # Given time constraints, we'll rely on the multiple restarts and lattice variations.
                            
    # Run the lattice search
    greedy_packing(None, N, 42)
    
    # Fallback if lattice didn't find a valid solution (unlikely, but just in case)
    if not best_valid:
        # Simple radial fallback
        sp = math.sqrt(3) * 1.01
        hex_config = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = math.radians(60 * i)
            hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in [0, 2, 4]:
            theta = math.radians(60 * i + 30)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        for i in [1, 3, 5]:
            theta = math.radians(60 * i)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        while len(hex_config) < N:
            hex_config.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        
        # Try to fix with local search
        # We don't have a local search function in this new version, so we just return
        # if valid.
        for oangle in [0, 30, 60]:
            if update_best(hex_config[:N], 0.0, 0.0, oangle):
                break

    if not best_valid or best_inner is None:
        # Last resort: linear chain
        sp = math.sqrt(3)
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
