import math
import time
import random
import numpy as np
from itertools import combinations, product

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute hex vertices for side 1, angle 0
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
        apothem_factor = SQRT3 / 2
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

    def local_search(inner, ocx, ocy, oangle, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
        cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)
        
        if not cur_valid:
            for _ in range(50):
                idx = rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] = old_x + rng.gauss(0, 0.5)
                cur_inner[idx][1] = old_y + rng.gauss(0, 0.5)
                cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
                cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)
                if cur_valid:
                    break
            if not cur_valid:
                return
        
        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
        
        # Simulated annealing style local search
        for it in range(3000):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.1)
            cur_inner[idx][1] += rng.gauss(0, 0.1)
            if rng.random() < 0.05:
                cur_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])
            
            new_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
            new_valid = all_hexagons_fit(cur_inner, ocx, ocy, new_S, oangle)
            
            if new_valid and new_S < cur_S:
                cur_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True
            elif new_valid:
                # Accept worse valid solutions with some probability (annealing)
                if rng.random() < 0.1:
                    cur_S = new_S
            else:
                cur_inner = old

    def try_greedy_lattice(spacing, angle_deg, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        # Generate lattice points
        max_coord = 12
        points = []
        a1 = (spacing, 0)
        a2 = (spacing / 2, spacing * SQRT3 / 2)
        
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = i * a1[0] + j * a2[0]
                y = i * a1[1] + j * a2[1]
                points.append((x, y))
        
        # Sort by distance from origin to prioritize central placements
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Rotate the lattice
        rad = math.radians(angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        
        rotated_points = []
        for x, y in points:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rotated_points.append((rx, ry))
        
        # Greedy placement
        used_indices = set()
        placed_points = [rotated_points[0]]
        used_indices.add(0)
        
        for _ in range(N - 1):
            if time.time() > deadline:
                return
                
            cx = sum(p[0] for p in placed_points) / len(placed_points)
            cy = sum(p[1] for p in placed_points) / len(placed_points)
            
            best_idx = -1
            best_dist = float('inf')
            
            search_limit = min(300, len(rotated_points))
            
            for idx in range(search_limit):
                if idx in used_indices:
                    continue
                px, py = rotated_points[idx]
                d = (px - cx)**2 + (py - cy)**2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            
            if best_idx == -1:
                for idx in range(len(rotated_points)):
                    if idx not in used_indices:
                        best_idx = idx
                        break
            
            if best_idx != -1:
                placed_points.append(rotated_points[best_idx])
                used_indices.add(best_idx)
        
        # Center the configuration
        cx = sum(p[0] for p in placed_points) / N
        cy = sum(p[1] for p in placed_points) / N
        
        inner = []
        for x, y in placed_points:
            inner.append([x - cx, y - cy, 0.0])
        
        # Try a few outer angles
        for oangle in [0, 30, 60]:
            S = compute_outer_hex(inner, 0.0, 0.0, oangle)
            if all_hexagons_fit(inner, 0.0, 0.0, S, oangle):
                if S < best_S:
                    best_S = S
                    best_inner = [list(x) for x in inner]
                    best_center = (0.0, 0.0)
                    best_angle = oangle
                    best_valid = True
                    # Run local search to refine
                    local_search(inner, 0.0, 0.0, oangle, seed)

    # NEW STRATEGY: Topological constraint satisfaction with contact graph
    # Define a fixed contact graph (3-4-3 triangular lattice pattern)
    # and solve for global orientation and container parameters
    
    def solve_contact_graph():
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        # Define contact graph: 11 nodes in a 3-4-3 triangular arrangement
        # Node positions in a regular triangular lattice (unit spacing)
        # Row 0: 3 nodes, Row 1: 4 nodes, Row 2: 3 nodes
        # This forms a compact hexagonal cluster
        
        # Axial coordinates for hexagonal lattice
        # Center at (0,0)
        # Neighbors at distance sqrt(3)
        d = SQRT3  # Distance between touching hexagon centers
        
        # 3-4-3 arrangement:
        # Row 0 (top): 3 nodes
        # Row 1 (middle): 4 nodes  
        # Row 2 (bottom): 3 nodes
        
        # Let's define positions relative to center
        # Using a coordinate system where rows are horizontal
        
        # Actually, let's use a simpler approach: define the contact graph
        # as a set of relative positions that form a valid packing
        
        # For 11 hexagons, a good arrangement is:
        # Center + 6 around it + 4 more in the gaps
        # This forms a larger hexagon of side 2 (minus corners)
        
        # Standard hexagonal packing: center + ring of 6 + ring of 12
        # But we only have 11, so: center + 6 + 4 from the second ring
        
        # Positions for center + 6 neighbors:
        base_positions = [(0.0, 0.0)]
        for i in range(6):
            theta = math.radians(60 * i)
            base_positions.append((d * math.cos(theta), d * math.sin(theta)))
        
        # Add 4 more from the second ring (positions at distance 2*d)
        # Second ring positions are at angles 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
        # Pick 4 that create a compact shape
        second_ring_angles = [30, 90, 150, 210]  # These create a symmetric pattern
        for angle in second_ring_angles:
            theta = math.radians(angle)
            base_positions.append((2 * d * math.cos(theta), 2 * d * math.sin(theta)))
        
        # Now we have 11 positions. Center them.
        cx = sum(p[0] for p in base_positions) / N
        cy = sum(p[1] for p in base_positions) / N
        centered_positions = [(x - cx, y - cy) for x, y in base_positions]
        
        # Now optimize the global rotation and container parameters
        # The key insight: if we fix the internal configuration (contact graph),
        # we only need to optimize the global rotation of the entire cluster
        # and the outer hexagon orientation
        
        best_local_S = float('inf')
        best_local_angle = 0.0
        
        # Try many global rotations of the cluster
        for cluster_angle in range(0, 360, 3):
            if time.time() > deadline:
                break
            
            # Rotate all positions
            rad = math.radians(cluster_angle)
            ca, sa = math.cos(rad), math.sin(rad)
            rotated_positions = []
            for x, y in centered_positions:
                rx = x * ca - y * sa
                ry = x * sa + y * ca
                rotated_positions.append((rx, ry))
            
            # For each cluster rotation, try different outer hexagon orientations
            for outer_angle in [0, 30, 60]:
                inner = [[x, y, 0.0] for x, y in rotated_positions]
                S = compute_outer_hex(inner, 0.0, 0.0, outer_angle)
                
                if all_hexagons_fit(inner, 0.0, 0.0, S, outer_angle):
                    if S < best_local_S:
                        best_local_S = S
                        best_local_angle = cluster_angle
                        best_local_inner = [list(x) for x in inner]
                        best_local_outer_angle = outer_angle
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_inner = best_local_inner
            best_center = (0.0, 0.0)
            best_angle = best_local_outer_angle
            best_valid = True
            # Run local search to refine
            local_search(best_inner, 0.0, 0.0, best_local_outer_angle, 12345)
    
    # Run lattice search
    base_spacing = SQRT3
    spacings = [base_spacing * (1 + i * 0.001) for i in range(5)]
    angles = [0, 15, 30, 45, 60]
    
    seed_counter = 100
    for sp in spacings:
        for ang in angles:
            if time.time() > deadline:
                break
            try_greedy_lattice(sp, ang, seed_counter)
            seed_counter += 1
    
    # Run contact graph optimization
    solve_contact_graph()
    
    # Fallback if lattice search didn't find anything good
    if not best_valid:
        initial_configs = []
        sp = SQRT3
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
        initial_configs.append((hex_config[:N], (0.0, 0.0), 0.0))
        
        for i, (cfg, c, a) in enumerate(initial_configs):
            if time.time() > deadline:
                break
            local_search(cfg, c[0], c[1], a, i * 1000 + 42)
    
    if not best_valid or best_inner is None:
        sp = SQRT3
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        for _ in range(100):
            if all_hexagons_fit(fallback_inner, 0.0, 0.0, S, 0.0):
                return fallback_inner, (0.0, 0.0), S, 0.0
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0
    
    return best_inner, best_center, best_S, best_angle
