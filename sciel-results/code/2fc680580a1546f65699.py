import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

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
        
        for it in range(2000):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.15)
            cur_inner[idx][1] += rng.gauss(0, 0.15)
            if rng.random() < 0.03:
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
                if rng.random() < 0.1:
                    cur_S = new_S
            else:
                cur_inner = old

    def run_md_packing(seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if time.time() > deadline:
            return

        rng = random.Random(seed)
        
        # 1. Initialization: Place hexagons on a loose triangular lattice
        # Spacing slightly larger than sqrt(3) to avoid initial overlaps
        spacing = math.sqrt(3) * 1.1
        positions = []
        # Generate a grid of points and pick N closest to origin
        candidates = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * spacing + j * spacing * 0.5
                y = j * spacing * math.sqrt(3) / 2
                candidates.append((x, y))
        
        candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
        # Pick N points
        for i in range(N):
            x, y = candidates[i]
            # Add small random jitter to break symmetry
            x += rng.gauss(0, 0.1)
            y += rng.gauss(0, 0.1)
            positions.append([x, y])
        
        # Angles: start with 0, maybe some random
        angles = [0.0] * N
        for i in range(N):
            if rng.random() < 0.2:
                angles[i] = rng.choice([0, 30, 60])

        # 2. Force-Directed Simulation (Molecular Dynamics)
        # We simulate the system for a fixed number of steps or until convergence.
        # Forces:
        # - Repulsive force between overlapping hexagons (approximated by center distance)
        #   Actually, for hexagons, center distance > 2*apothem is not sufficient for non-overlap if rotated.
        #   But for a coarse MD step, we can use a smooth potential based on minimum distance between vertices?
        #   That's expensive. 
        #   Alternative: Use a "soft" constraint. If they overlap, push apart.
        #   Let's use a simpler heuristic: Minimize the maximum distance from centroid? No, that's not packing.
        #   Let's use a standard MD approach with a potential energy function.
        #   E = sum(overlap_penalty) + boundary_penalty
        #   
        #   To make it efficient, we will use a simplified collision detection:
        #   If distance between centers < 2*apothem (sqrt(3)), assume overlap and apply repulsion.
        #   This is an approximation but works well for initial compression.
        #   We will refine with exact geometry later.
        
        apothem = math.sqrt(3) / 2
        min_dist = 2 * apothem # sqrt(3)
        
        # Boundary: We want to compress into a hexagon.
        # Let's define a "target" hexagon size that we slowly shrink?
        # Or just let them cluster and then compute the outer hex.
        # Better: Add a force pulling all centers towards the centroid to keep them compact.
        
        dt = 0.01
        damping = 0.9
        k_repel = 10.0
        k_center = 0.1
        steps = 2000
        
        velocities = [[0.0, 0.0] for _ in range(N)]
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            forces = [[0.0, 0.0] for _ in range(N)]
            
            # Calculate center of mass
            com_x = sum(p[0] for p in positions) / N
            com_y = sum(p[1] for p in positions) / N
            
            for i in range(N):
                # Force towards center (compression)
                dx_c = com_x - positions[i][0]
                dy_c = com_y - positions[i][1]
                forces[i][0] += k_center * dx_c
                forces[i][1] += k_center * dy_c
                
                for j in range(i + 1, N):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    # If overlapping (dist < min_dist), apply repulsive force
                    # We use a smooth potential to avoid discontinuities
                    if dist < min_dist * 1.2:
                        # Linear repulsion if overlapping, smooth transition
                        overlap = min_dist - dist
                        if overlap > 0:
                            f = k_repel * overlap
                        else:
                            # Smooth tail
                            f = k_repel * 0.1 * (1.2 * min_dist - dist) / (0.2 * min_dist)
                        
                        if dist > 1e-6:
                            fx = f * dx / dist
                            fy = f * dy / dist
                        else:
                            # Random push if exactly same position
                            fx = rng.gauss(0, 0.1)
                            fy = rng.gauss(0, 0.1)
                        
                        forces[i][0] -= fx
                        forces[i][1] -= fy
                        forces[j][0] += fx
                        forces[j][1] += fy
            
            # Update velocities and positions
            for i in range(N):
                velocities[i][0] = (velocities[i][0] + forces[i][0] * dt) * damping
                velocities[i][1] = (velocities[i][1] + forces[i][1] * dt) * damping
                positions[i][0] += velocities[i][0] * dt
                positions[i][1] += velocities[i][1] * dt
                
            # Center the configuration at origin to prevent drift
            com_x = sum(p[0] for p in positions) / N
            com_y = sum(p[1] for p in positions) / N
            for i in range(N):
                positions[i][0] -= com_x
                positions[i][1] -= com_y

        # 3. Evaluate the resulting configuration
        # The MD simulation used an approximate repulsion. We must verify exact disjointness.
        # Also, the MD might result in a configuration that is valid but not optimally rotated for the outer hex.
        
        inner = []
        for i in range(N):
            inner.append([positions[i][0], positions[i][1], angles[i]])
        
        # Try a few outer angles
        for oangle in [0, 30, 60]:
            S = compute_outer_hex(inner, 0.0, 0.0, oangle)
            # Check validity
            if all_hexagons_fit(inner, 0.0, 0.0, S, oangle):
                if S < best_S:
                    best_S = S
                    best_inner = [list(x) for x in inner]
                    best_center = (0.0, 0.0)
                    best_angle = oangle
                    best_valid = True
                    # Run local search to refine
                    local_search(inner, 0.0, 0.0, oangle, seed)
            else:
                # If invalid, try to relax slightly or just skip
                # A quick local search might fix minor overlaps if we start close
                # But local_search expects a valid start or tries to fix.
                # Let's just try local search anyway, it has a repair phase.
                local_search(inner, 0.0, 0.0, oangle, seed)

    # Run multiple MD simulations with different seeds
    for i in range(20):
        if time.time() > deadline:
            break
        run_md_packing(i * 100 + 1)

    # Fallback if MD didn't find a valid solution
    if not best_valid:
        # Simple lattice fallback
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
        
        local_search(hex_config[:N], 0.0, 0.0, 0.0, 999)

    if not best_valid or best_inner is None:
        sp = math.sqrt(3)
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
# EVOLVE-BLOCK-END
