import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon (pointy-top by default)."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

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

    def hexagons_disjoint(verts_a, verts_b):
        """Check if two hexagons (given as vertex lists) are disjoint (no overlap)."""
        # Check if any vertex of A is inside B
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        # Check if any vertex of B is inside A
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
        # Check each inner hex is inside outer
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        # Check mutual disjointness
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
            # Try to fix by expanding or adjusting
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

    def run_differential_evolution(seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        
        # Initial population
        # Structure: [x1, y1, angle_idx1, ..., xN, yN, angle_idxN, S, ocx, ocy, oangle]
        # Angle indices: 0->0, 1->30, 2->60
        angle_map = [0, 30, 60]
        
        # Bounds for each hexagon position
        # Estimate reasonable bounds based on seed solution ~3.6
        # Max distance from center is roughly S
        # Let's use a box of [-4, 4] for x, y
        bounds = [(-4.0, 4.0)] * (2 * N)
        # Angle indices are discrete, but DE works on continuous. 
        # We will round to nearest valid index in the objective function.
        # However, standard DE doesn't handle integers well directly.
        # We can treat them as continuous and round.
        bounds.extend([(0.0, 2.0)] * N)
        # S (side length)
        bounds.append((2.0, 6.0))
        # Outer center x, y
        bounds.append((-2.0, 2.0))
        bounds.append((-2.0, 2.0))
        # Outer angle
        bounds.append((0.0, 180.0))
        
        def objective(params):
            if time.time() > deadline:
                return 1e6
            # Unpack
            inner = []
            for i in range(N):
                x = params[2*i]
                y = params[2*i + 1]
                ang_idx = int(round(params[2*N + i]))
                ang_idx = max(0, min(2, ang_idx))
                inner.append([x, y, angle_map[ang_idx]])
            
            S = params[3*N]
            ocx = params[3*N + 1]
            ocy = params[3*N + 2]
            oangle = params[3*N + 3]
            
            # Check validity
            if not all_hexagons_fit(inner, ocx, ocy, S, oangle):
                # Penalty for invalidity
                # Calculate how much they overlap or go out of bounds
                # Simple penalty: large number
                return 100.0 + S
                
            return S
        
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                seed=seed,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                maxfevals=2000
            )
            
            if result.success and result.fun < 100.0:
                params = result.x
                inner = []
                for i in range(N):
                    x = params[2*i]
                    y = params[2*i + 1]
                    ang_idx = int(round(params[2*N + i]))
                    ang_idx = max(0, min(2, ang_idx))
                    inner.append([x, y, angle_map[ang_idx]])
                
                S = params[3*N]
                ocx = params[3*N + 1]
                ocy = params[3*N + 2]
                oangle = params[3*N + 3]
                
                # Refine with local search
                local_search(inner, ocx, ocy, oangle, seed + 10000)
        except Exception:
            pass

    # Generate initial configurations for local search (fallback and seeds)
    initial_configs = []
    
    # Configuration 1: hexagonal lattice
    sp = math.sqrt(3)
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
    
    # Configuration 2: linear
    lin_config = []
    for k in range(N):
        lin_config.append([(k - N // 2) * sp, 0.0, 0.0])
    initial_configs.append((lin_config, (0.0, 0.0), 0.0))
    
    # Run local search on these first to establish a baseline
    for i, (cfg, c, a) in enumerate(initial_configs):
        if time.time() > deadline:
            break
        local_search(cfg, c[0], c[1], a, i * 1000 + 42)

    # Run Differential Evolution
    # We have limited time. Let's run a few DE instances.
    num_de_runs = 5
    for run in range(num_de_runs):
        if time.time() > deadline:
            break
        run_differential_evolution(run * 100 + 1)

    if not best_valid or best_inner is None:
        # Fallback: use a simple valid configuration
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
