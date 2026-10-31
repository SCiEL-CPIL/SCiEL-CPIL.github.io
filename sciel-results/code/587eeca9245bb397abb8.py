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

    # Precompute unit hex vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300)
    # Using "pointy-top" relative to x-axis for standard math orientation
    # Vertices at angles 0, 60, ...
    base_verts = [(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)]

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(cx + S*(vx*ca - vy*sa), cy + S*(vx*sa + vy*ca)) for vx, vy in base_verts]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def segments_intersect(p1, p2, p3, p4):
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        return False

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(va, vb):
        # Fast check: bounding boxes
        min_xa = min(v[0] for v in va); max_xa = max(v[0] for v in va)
        min_ya = min(v[1] for v in va); max_ya = max(v[1] for v in va)
        min_xb = min(v[0] for v in vb); max_xb = max(v[0] for v in vb)
        min_yb = min(v[1] for v in vb); max_yb = max(v[1] for v in vb)
        if max_xa < min_xb or max_xb < min_xa or max_ya < min_yb or max_yb < min_ya:
            return True
        
        for v in va:
            if point_in_polygon(v, vb): return False
        for v in vb:
            if point_in_polygon(v, va): return False
        for i in range(6):
            if segments_intersect(va[i], va[(i+1)%6], vb[i], vb[(i+1)%6]):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0]-ocx)*nx + (v[1]-ocy)*ny
                    if proj > max_proj: max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S: max_S = S_needed
        return max_S

    def all_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        # Check containment
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        # Check disjointness
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, S, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S and all_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
            return True
        return False

    # --- STRATEGY: Discrete Lattice Search with Local Refinement ---
    # The optimal packing for 11 hexagons is likely the "flower" pattern (1 center, 6 around, 4 more?) 
    # or a 3-4-3 strip. 
    # For N=11, the 3-4-3 pattern (rows of 3, 4, 3) is very tight.
    # Let's generate candidate configurations based on hexagonal lattice coordinates.
    
    # Hex lattice vectors for side length 1 (touching)
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3)
    dist = math.sqrt(3)
    
    # Generate a set of candidate lattice points around the origin
    # Axial coordinates (q, r)
    lattice_points = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            # Convert axial to pixel (pointy-top)
            # x = dist * (q + r/2)
            # y = dist * (sqrt(3)/2 * r)
            x = dist * (q + r/2.0)
            y = dist * (math.sqrt(3)/2.0 * r)
            lattice_points.append((x, y))
            
    # We need to select 11 points from this lattice that form a valid packing.
    # Since lattice points are exactly touching (distance sqrt(3)), any subset of them is disjoint (touching is allowed? "disjoint" usually means interiors disjoint. Touching boundaries is typically allowed in packing problems unless "strictly disjoint" is specified. The problem says "disjoint", which in geometry often means no common interior points. Touching is fine.)
    # However, we can also scale them slightly to ensure strict disjointness if needed, or just rely on the fact that lattice points are valid.
    
    # We want to minimize the bounding hexagon size.
    # Symmetric configurations are likely optimal.
    # Let's try to find the 11 points with the smallest bounding box (or smallest enclosing hexagon).
    
    # To speed up, we can restrict to "connected" clusters or just take the 11 closest to the origin?
    # Taking the 11 closest to the origin gives the "flower" (1 center + 6 neighbors + 4 next-neighbors).
    # Let's evaluate the "flower" configuration first.
    
    def evaluate_config(points):
        # points: list of (x, y)
        # Center the configuration
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        inner = [[p[0]-cx, p[1]-cy, 0.0] for p in points]
        
        # Compute minimal enclosing hexagon size
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        
        # Verify fit
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return S
        return float('inf')
    
    # Candidate 1: 11 closest lattice points to origin
    sorted_points = sorted(lattice_points, key=lambda p: p[0]**2 + p[1]**2)
    candidate_1 = sorted_points[:11]
    S1 = evaluate_config(candidate_1)
    if S1 < best_S:
        update_best([[p[0] - sum(c[0] for c in candidate_1)/11, p[1] - sum(c[1] for c in candidate_1)/11, 0.0] for p in candidate_1], 0.0, 0.0, S1, 0.0)
        
    # Candidate 2: 3-4-3 Pattern
    # Rows of 3, 4, 3
    # y-coords: -dist*sqrt(3)/2, 0, dist*sqrt(3)/2
    # Row 0 (bottom, 3 hexes): x = -dist, 0, dist  (shifted by 0.5*dist to center? No, lattice alignment)
    # Let's construct 3-4-3 explicitly on the lattice.
    # Axial coords for 3-4-3:
    # Top row (y=1): (0,1), (1,0), (2,-1) ? No, let's use pixel coords directly.
    # Center at (0,0).
    # Middle row (y=0): 4 hexes. x = -1.5*dist, -0.5*dist, 0.5*dist, 1.5*dist
    # Top row (y=dist*sqrt(3)/2): 3 hexes. x = -1.0*dist, 0.0*dist, 1.0*dist
    # Bottom row (y=-dist*sqrt(3)/2): 3 hexes. x = -1.0*dist, 0.0*dist, 1.0*dist
    
    candidate_2 = [
        (-1.0*dist, dist*math.sqrt(3)/2),
        (0.0, dist*math.sqrt(3)/2),
        (1.0*dist, dist*math.sqrt(3)/2),
        (-1.5*dist, 0.0),
        (-0.5*dist, 0.0),
        (0.5*dist, 0.0),
        (1.5*dist, 0.0),
        (-1.0*dist, -dist*math.sqrt(3)/2),
        (0.0, -dist*math.sqrt(3)/2),
        (1.0*dist, -dist*math.sqrt(3)/2),
        (0.0, 0.0) # Wait, 3+4+3 = 10. Need 11. 
    ]
    # The 3-4-3 pattern usually has 10. To get 11, we might add a center one if not present, or it's a different shape.
    # Actually, the "flower" (1+6+4) is 11.
    # Let's try a "bar" of 11? No, that's long.
    # Let's try to optimize the "flower" by rotating the outer hexagon? 
    # The current `compute_outer_hex` assumes the outer hexagon is aligned with the axes (angle 0).
    # We should also check if rotating the outer hexagon helps.
    # But for symmetric clusters, alignment is usually optimal.
    
    # Let's do a local search on the best candidate found so far to fine-tune positions slightly (allowing for non-lattice optimalities if any, though unlikely for hexagons).
    
    if best_valid and best_inner is not None:
        def local_refine(inner, seed):
            nonlocal best_S, best_inner, best_center, best_angle, best_valid
            rng = random.Random(seed)
            cur = [list(x) for x in inner]
            cx = sum(p[0] for p in cur) / N
            cy = sum(p[1] for p in cur) / N
            cur = [[p[0]-cx, p[1]-cy, p[2]] for p in cur]
            cur_S = compute_outer_hex(cur, 0.0, 0.0, 0.0)
            
            # Try to shrink by moving hexagons towards center
            # Or just random perturbation
            for it in range(2000):
                if time.time() > deadline:
                    break
                old = [list(x) for x in cur]
                idx = rng.randrange(N)
                # Small perturbation
                cur[idx][0] += rng.gauss(0, 0.01)
                cur[idx][1] += rng.gauss(0, 0.01)
                
                new_S = compute_outer_hex(cur, 0.0, 0.0, 0.0)
                if new_S < cur_S and all_fit(cur, 0.0, 0.0, new_S, 0.0):
                    cur_S = new_S
                    if new_S < best_S:
                        best_S = new_S
                        best_inner = [list(x) for x in cur]
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                        best_valid = True
                else:
                    cur = old

        local_refine(best_inner, 42)

    # If we still haven't found a valid one (unlikely), fallback to DE
    if not best_valid:
        def objective(params):
            if time.time() > deadline:
                return 1e10
            inner = []
            for i in range(N):
                x = params[3*i]
                y = params[3*i+1]
                a = params[3*i+2]
                inner.append([x, y, a])
            cx = sum(p[0] for p in inner) / N
            cy = sum(p[1] for p in inner) / N
            inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
            S = compute_outer_hex(inner_c, 0.0, 0.0, 0.0)
            if not all_fit(inner_c, 0.0, 0.0, S, 0.0):
                return S + 100.0
            return S

        bounds = [(-6, 6), (-6, 6), (0, 60)] * N
        try:
            de_result = differential_evolution(objective, bounds, maxiter=50, popsize=10, tol=1e-4, seed=1, polish=False)
            if de_result.fun < best_S:
                params = de_result.x
                inner = []
                for i in range(N):
                    inner.append([params[3*i], params[3*i+1], params[3*i+2]])
                cx = sum(p[0] for p in inner) / N
                cy = sum(p[1] for p in inner) / N
                inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
                S = compute_outer_hex(inner_c, 0.0, 0.0, 0.0)
                if all_fit(inner_c, 0.0, 0.0, S, 0.0):
                    best_S = S
                    best_inner = [list(x) for x in inner_c]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
                    best_valid = True
        except Exception:
            pass

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Absolute fallback
    sp = math.sqrt(3)
    inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
    S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
    for _ in range(200):
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        inner = [[p[0]*1.05, p[1]*1.05, p[2]] for p in inner]
        S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
