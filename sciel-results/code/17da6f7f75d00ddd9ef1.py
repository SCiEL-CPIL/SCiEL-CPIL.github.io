import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    def point_in_polygon(p, polygon):
        """Ray casting point in polygon test."""
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

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
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

    def hexagons_disjoint(verts_a, verts_b):
        """Check if two hexagons (given as vertex lists) are disjoint."""
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            if segments_intersect(verts_a[i], verts_a[(i+1) % 6], verts_b[i], verts_b[(i+1) % 6]):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """Find minimum S such that all inner hexagons fit in outer hexagon at (ocx, ocy) with angle oangle."""
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(hex_vertices(ix, iy, 1.0, iangle))
        
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
        """Check if all inner hexagons are inside outer and mutually disjoint."""
        outer_verts = hex_vertices(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = hex_vertices(ix, iy, 1.0, iangle)
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

    def validate_and_update(inner, ocx, ocy, oangle):
        """Validate config and update best if improved."""
        nonlocal best_S, best_inner, best_center, best_angle
        S = compute_outer_hex(inner, ocx, ocy, oangle)
        if S < best_S and all_hexagons_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
        return S

    def local_search(inner, ocx, ocy, oangle, seed, max_iters=5000):
        """Local search with simulated annealing-like acceptance."""
        nonlocal best_S, best_inner, best_center, best_angle
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
        cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)
        if not cur_valid:
            # Try to fix initial config
            for fix_iter in range(100):
                idx = rng.randrange(N)
                cur_inner[idx][0] += rng.gauss(0, 0.5)
                cur_inner[idx][1] += rng.gauss(0, 0.5)
                cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
                if all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle):
                    cur_valid = True
                    break
            if not cur_valid:
                return

        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            step_size = 0.3 * max(0.1, 1 - no_improve / max_iters)
            cur_inner[idx][0] += rng.gauss(0, step_size)
            cur_inner[idx][1] += rng.gauss(0, step_size)
            if rng.random() < 0.08:
                cur_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])
            
            new_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
            new_valid = all_hexagons_fit(cur_inner, ocx, ocy, new_S, oangle)
            
            if new_valid and new_S < cur_S:
                cur_inner = [list(x) for x in cur_inner]
                cur_S = new_S
                no_improve = 0
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
            elif new_valid and rng.random() < 0.15:
                cur_inner = [list(x) for x in cur_inner]
                cur_S = new_S
                no_improve += 1
            else:
                cur_inner = old
                no_improve += 1
    
    # Initial configuration 1: hexagonal lattice (center + 6 neighbors + 4 more)
    sp = math.sqrt(3)
    hex_config = []
    hex_config.append([0.0, 0.0, 0.0])
    for i in range(6):
        theta = math.radians(60 * i)
        hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # Second ring positions
    for i in range(6):
        theta = math.radians(60 * i)
        r = 2 * sp
        hex_config.append([r * math.cos(theta), r * math.sin(theta), 0.0])
    # Add diagonal positions to reach 11
    for i in range(6):
        theta = math.radians(60 * i + 30)
        r = sp * math.sqrt(3)
        hex_config.append([r * math.cos(theta), r * math.sin(theta), 0.0])
    hex_config = hex_config[:N]
    validate_and_update(hex_config, 0.0, 0.0, 0.0)
    local_search(hex_config, 0.0, 0.0, 0.0, 42, 8000)
    
    # Initial configuration 2: linear arrangement
    lin_config = []
    for k in range(N):
        lin_config.append([(k - N // 2) * sp, 0.0, 0.0])
    validate_and_update(lin_config, 0.0, 0.0, 0.0)
    local_search(lin_config, 0.0, 0.0, 0.0, 123, 6000)
    
    # Initial configuration 3: mixed orientations for better packing
    mixed_config = []
    mixed_config.append([0.0, 0.0, 0.0])
    for i in range(6):
        theta = math.radians(60 * i)
        ang = 0 if i % 2 == 0 else 30
        mixed_config.append([sp * math.cos(theta), sp * math.sin(theta), ang])
    for i in range(4):
        theta = math.radians(60 * i + 30)
        ang = 0 if i % 2 == 0 else 30
        mixed_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), ang])
    mixed_config = mixed_config[:N]
    validate_and_update(mixed_config, 0.0, 0.0, 0.0)
    local_search(mixed_config, 0.0, 0.0, 0.0, 789, 6000)
    
    # More random restarts
    for restart in range(15):
        if time.time() > deadline:
            break
        rng = random.Random(1000 + restart * 17)
        rand_config = []
        for _ in range(N):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            ang = rng.choice([0, 30, 60, 90, 120, 150])
            rand_config.append([x, y, ang])
        validate_and_update(rand_config, 0.0, 0.0, 0.0)
        local_search(rand_config, 0.0, 0.0, 0.0, 5000 + restart * 100, 4000)
    
    # Try different outer angles
    for oangle in [30, 60, 15]:
        if time.time() > deadline:
            break
        hex_config2 = []
        hex_config2.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = math.radians(60 * i)
            hex_config2.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in range(4):
            theta = math.radians(60 * i + 30)
            hex_config2.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        hex_config2 = hex_config2[:N]
        validate_and_update(hex_config2, 0.0, 0.0, oangle)
        local_search(hex_config2, 0.0, 0.0, oangle, 9000 + oangle, 5000)
    
    if best_inner is None:
        # Fallback: linear config
        sp = math.sqrt(3)
        config = []
        for k in range(N):
            config.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(config, 0.0, 0.0, 0.0)
        return config, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
