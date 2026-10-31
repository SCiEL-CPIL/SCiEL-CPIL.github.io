import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute base hex vertices for side 1, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = []
    for i in range(6):
        theta = math.radians(60 * i)
        base_verts.append((math.cos(theta), math.sin(theta)))
    
    # Convert to numpy array for faster vectorized operations if needed, 
    # but list of tuples is fine for N=11 and small vertex counts.
    
    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            # Rotate
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            # Scale and Translate
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
        # Check if any vertex of A is inside B or vice versa
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        # Check edge intersections
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """
        Computes the minimum side length S of an outer hexagon centered at (ocx, ocy) 
        with rotation oangle that contains all inner hexagons.
        """
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        # The outer hexagon is defined by 6 half-planes.
        # For a regular hexagon of side S, the distance from center to each side is S * sqrt(3)/2.
        # The normal vectors to the sides are at angles a + 30 + 60*i (if vertices are at a + 60*i).
        # Wait, let's verify the geometry.
        # If vertices are at angles theta_k = a + 60*k, the sides are perpendicular to the bisectors.
        # Bisectors are at a + 30 + 60*k.
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    # Projection of vector from center to vertex onto normal
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        """
        Checks if all inner hexagons are inside the outer hexagon and disjoint.
        """
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        
        # Check containment
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        
        # Check disjointness
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def get_initial_config():
        """
        Generates a valid initial configuration using a hexagonal lattice cluster.
        """
        sp = math.sqrt(3) * 1.01 # Slightly larger to ensure strict disjointness initially
        # Create a cluster of N hexagons on a triangular lattice
        # We want a compact shape.
        # For N=11, a hexagonal number H_2 = 7 is too small. H_3 = 19 is too big.
        # So we take a H_2 cluster (7) and add 4 more around it.
        # Or just take the 11 closest points in a lattice.
        
        points = []
        max_coord = 5
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = i * sp + j * sp / 2
                y = j * sp * math.sqrt(3) / 2
                points.append((x, y))
        
        # Sort by distance from origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        inner = []
        for k in range(N):
            x, y = points[k]
            inner.append([x, y, 0.0])
            
        # Center the configuration
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        for p in inner:
            p[0] -= cx
            p[1] -= cy
            
        return inner, (0.0, 0.0), 0.0

    def simulated_annealing(initial_inner, initial_center, initial_angle, seed, steps, T_start, T_end):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        rng = random.Random(seed)
        cur_inner = [list(x) for x in initial_inner]
        cur_center = list(initial_center)
        cur_angle = initial_angle
        
        cur_S = compute_outer_hex(cur_inner, cur_center[0], cur_center[1], cur_angle)
        cur_valid = all_hexagons_fit(cur_inner, cur_center[0], cur_center[1], cur_S, cur_angle)
        
        if not cur_valid:
            # Try to fix validity quickly
            for _ in range(50):
                idx = rng.randrange(N)
                cur_inner[idx][0] += rng.gauss(0, 0.2)
                cur_inner[idx][1] += rng.gauss(0, 0.2)
                cur_S = compute_outer_hex(cur_inner, cur_center[0], cur_center[1], cur_angle)
                cur_valid = all_hexagons_fit(cur_inner, cur_center[0], cur_center[1], cur_S, cur_angle)
                if cur_valid:
                    break
            if not cur_valid:
                return

        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_center = tuple(cur_center)
            best_angle = cur_angle
            best_valid = True

        T = T_start
        step_size = (T_start - T_end) / steps
        
        for it in range(steps):
            if time.time() > deadline:
                break
            
            # Generate a new candidate
            new_inner = [list(x) for x in cur_inner]
            new_center = list(cur_center)
            new_angle = cur_angle
            
            # Move one inner hexagon
            idx = rng.randrange(N)
            new_inner[idx][0] += rng.gauss(0, 0.2)
            new_inner[idx][1] += rng.gauss(0, 0.2)
            
            # Occasionally rotate an inner hexagon
            if rng.random() < 0.05:
                new_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])
            
            # Occasionally move the outer center or rotate the outer hexagon
            if rng.random() < 0.1:
                new_center[0] += rng.gauss(0, 0.1)
                new_center[1] += rng.gauss(0, 0.1)
            if rng.random() < 0.05:
                new_angle += rng.gauss(0, 5.0)
            
            new_S = compute_outer_hex(new_inner, new_center[0], new_center[1], new_angle)
            new_valid = all_hexagons_fit(new_inner, new_center[0], new_center[1], new_S, new_angle)
            
            if new_valid:
                # Accept if better, or with probability exp(-delta/T) if worse
                if new_S < cur_S:
                    cur_inner = new_inner
                    cur_center = new_center
                    cur_angle = new_angle
                    cur_S = new_S
                    if cur_S < best_S:
                        best_S = cur_S
                        best_inner = [list(x) for x in cur_inner]
                        best_center = tuple(cur_center)
                        best_angle = cur_angle
                        best_valid = True
                else:
                    # Accept worse solution with probability
                    delta = new_S - cur_S
                    if delta > 0 and T > 0:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            cur_inner = new_inner
                            cur_center = new_center
                            cur_angle = new_angle
                            cur_S = new_S
            else:
                # Invalid solution: reject immediately (or accept with very low prob to escape?)
                # For packing, invalid usually means overlap. Better to reject to maintain feasibility.
                pass
                
            T -= step_size
            if T < 0:
                T = 0

    # Main Loop: Multiple Restarts of Simulated Annealing
    rng_master = random.Random(42)
    
    # We will run several SA runs with different initial configurations and parameters.
    # Initial configurations are generated by perturbing the lattice cluster.
    
    num_restarts = 10
    steps_per_sa = 5000
    T_start = 5.0
    T_end = 0.01
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10: # Stop if close to deadline
            break
            
        # Generate initial config
        init_inner, init_center, init_angle = get_initial_config()
        
        # Perturb initial config to break symmetry and explore different basins
        for i in range(N):
            init_inner[i][0] += rng_master.gauss(0, 0.5)
            init_inner[i][1] += rng_master.gauss(0, 0.5)
            # Randomize angles slightly
            if rng_master.random() < 0.2:
                init_inner[i][2] = rng_master.choice([0, 30, 60, 90, 120, 150])
        
        # Randomize outer angle
        init_angle = rng_master.uniform(0, 60)
        
        # Run SA
        simulated_annealing(init_inner, init_center, init_angle, seed=1000 + restart * 17, steps=steps_per_sa, T_start=T_start, T_end=T_end)
        
        # If we found a valid solution, try a quick local refinement (hill climbing)
        if best_valid:
            # Local search to polish
            local_rng = random.Random(2000 + restart)
            cur_inner = [list(x) for x in best_inner]
            cur_center = list(best_center)
            cur_angle = best_angle
            cur_S = best_S
            
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                idx = local_rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] += local_rng.gauss(0, 0.05)
                cur_inner[idx][1] += local_rng.gauss(0, 0.05)
                
                new_S = compute_outer_hex(cur_inner, cur_center[0], cur_center[1], cur_angle)
                new_valid = all_hexagons_fit(cur_inner, cur_center[0], cur_center[1], new_S, cur_angle)
                
                if new_valid and new_S < cur_S:
                    cur_S = new_S
                    if cur_S < best_S:
                        best_S = cur_S
                        best_inner = [list(x) for x in cur_inner]
                        best_center = tuple(cur_center)
                        best_angle = cur_angle
                else:
                    cur_inner[idx][0] = old_x
                    cur_inner[idx][1] = old_y
                    cur_inner[idx][2] = old_a

    if not best_valid or best_inner is None:
        # Fallback: Return the initial lattice config scaled up to fit
        init_inner, init_center, init_angle = get_initial_config()
        S = compute_outer_hex(init_inner, init_center[0], init_center[1], init_angle)
        # Scale up if needed to ensure fit (should already fit if disjoint)
        # But compute_outer_hex gives the min S that fits. 
        # We need to ensure disjointness.
        if all_hexagons_fit(init_inner, init_center[0], init_center[1], S, init_angle):
            return init_inner, init_center, S, init_angle
        else:
            # Increase S until valid? No, increasing S doesn't fix disjointness.
            # If disjointness fails, we need to move them apart.
            # For fallback, just return a valid linear arrangement
            fallback_inner = []
            sp = math.sqrt(3) * 1.01
            for k in range(N):
                fallback_inner.append([k * sp, 0.0, 0.0])
            S = compute_outer_hex(fallback_inner, (N-1)*sp/2, 0.0, 0.0)
            return fallback_inner, ((N-1)*sp/2, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
