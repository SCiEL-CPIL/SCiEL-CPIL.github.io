import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees)
    # This orientation is standard for "pointy top" hexes.
    # Actually, let's stick to the vertex definition used in the seed to ensure compatibility, 
    # but the code below handles arbitrary angles.
    # Standard unit hexagon (side 1) vertices relative to center:
    # If pointy-top: (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    # If flat-top: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # The problem allows rotation, so the base shape doesn't matter as long as it's consistent.
    # Let's use flat-top for simplicity in lattice calculations (horizontal rows).
    base_verts = [(1, 0), (0.5, math.sqrt(3)/2), (-0.5, math.sqrt(3)/2), (-1, 0), (-0.5, -math.sqrt(3)/2), (0.5, -math.sqrt(3)/2)]

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

    # --- STRATEGY: Topological Search with Global Optimization ---
    # Generate candidate topologies (sets of lattice points)
    # Lattice vectors for flat-top hexagons (side 1):
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3)
    # Axial coordinates (q, r) to pixel (x, y) for flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    dist_x = 1.5
    dist_y = math.sqrt(3)

    def axial_to_pixel(q, r):
        x = dist_x * q
        y = dist_y * (r + 0.5 * q)
        return (x, y)

    # Define candidate topologies as lists of (q, r)
    # We need 11 points.
    candidates = []
    
    # 1. "Flower" + 4 more (1 center, 6 neighbors, 4 next-neighbors)
    # Center (0,0)
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Next neighbors (touching neighbors but not center):
    # (2,0), (1,1), (-2,1), (-2,0), (-1,-1), (2,-1) ... 
    # Let's just take the 11 closest to origin on the lattice.
    lattice_points = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            p = axial_to_pixel(q, r)
            lattice_points.append((q, r, p[0], p[1]))
            
    lattice_points.sort(key=lambda x: x[2]**2 + x[3]**2)
    top_11 = [axial_to_pixel(q, r) for q, r, x, y in lattice_points[:11]]
    candidates.append(top_11)
    
    # 2. 3-4-3 Pattern (Rows of 3, 4, 3)
    # This is a very tight packing.
    # Row 0 (y=0): 4 hexes. q = -1, 0, 1, 2 ? No, need to be contiguous.
    # Let's construct explicitly.
    # Middle row (4): (0,0), (1,0), (2,0), (3,0) -> x: 0, 1.5, 3, 4.5. y: 0, 0, 0, 0.
    # Top row (3): Needs to sit on top.
    # Hex at (1,0) has neighbors (1,1), (0,1).
    # Hex at (2,0) has neighbors (2,1), (1,1).
    # Hex at (3,0) has neighbors (3,1), (2,1).
    # So top row can be (0,1), (1,1), (2,1).
    # Bottom row (3): (0,-1), (1,-1), (2,-1).
    # Total: 4 + 3 + 3 = 10. We need 11.
    # Add one more? Maybe (3,1) or (-1,0)?
    # Let's try a 2-5-2-2? Or 1-3-4-3-1?
    
    # Let's generate a few specific compact shapes manually.
    
    # Shape A: 3-4-3 + 1 extra
    # Base 3-4-3 (10 hexes) centered.
    # Middle: (0,0), (1,0), (2,0), (3,0)
    # Top: (0,1), (1,1), (2,1)
    # Bottom: (0,-1), (1,-1), (2,-1)
    # This is 10.
    # Add (3,1) -> 11.
    shape_A = [
        axial_to_pixel(0,0), axial_to_pixel(1,0), axial_to_pixel(2,0), axial_to_pixel(3,0),
        axial_to_pixel(0,1), axial_to_pixel(1,1), axial_to_pixel(2,1),
        axial_to_pixel(0,-1), axial_to_pixel(1,-1), axial_to_pixel(2,-1),
        axial_to_pixel(3,1)
    ]
    candidates.append(shape_A)
    
    # Shape B: 2-5-2-2 ?
    # Let's try a "diamond" or "rhombus" shape.
    # 2 on top, 3, 3, 2, 1?
    
    # Shape C: 1-2-3-4-3-2-1 ? (Too many)
    
    # Let's use a simple heuristic: Generate all connected subsets of size 11 from a small grid?
    # Too slow.
    
    # Instead, let's define a few known good topologies for N=11.
    # The "Flower" (1+6+4) is often optimal for small N.
    # Let's ensure the "11 closest" is the flower.
    # Center (0,0)
    # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 next-neighbors: (2,0), (1,1), (-2,1), (-1,-1) ? 
    # Distances:
    # (0,0): 0
    # Neighbors: dist^2 = 3
    # Next neighbors: dist^2 = 9 (e.g., (2,0) -> x=3, y=1.5 -> 9+2.25=11.25? No.)
    # Axial dist squared = q^2 + r^2 + qr * 3? No.
    # Pixel dist squared for (q,r):
    # x = 1.5q, y = sqrt(3)(r + 0.5q)
    # d^2 = 2.25q^2 + 3(r + 0.5q)^2 = 2.25q^2 + 3r^2 + 3qr + 0.75q^2 = 3q^2 + 3r^2 + 3qr = 3(q^2 + r^2 + qr)
    # So we minimize q^2 + r^2 + qr.
    
    # q=0, r=0: 0
    # q=1, r=0: 1
    # q=0, r=1: 1
    # q=-1, r=1: 1
    # q=-1, r=0: 1
    # q=0, r=-1: 1
    # q=1, r=-1: 1
    # (6 neighbors)
    
    # q=2, r=0: 4
    # q=1, r=1: 3
    # q=-1, r=2: 3
    # q=-2, r=1: 3
    # q=-2, r=0: 4
    # q=-1, r=-1: 3
    # q=1, r=-2: 3
    # q=2, r=-1: 3
    # (8 next-neighbors)
    
    # So the 11 closest are: 1 center + 6 neighbors + 4 of the 8 next-neighbors.
    # Which 4? It depends on symmetry.
    # The "Flower" usually picks 4 that form a symmetric shape.
    # e.g., (2,0), (-2,1), (-1,-1), (1,1)? 
    # Let's just take the first 4 from the sorted list.
    
    # Shape D: 3-4-3-1?
    # Let's try a "bar" of 3 rows: 3, 5, 3.
    # Middle row 5: (0,0) to (4,0)
    # Top row 3: (0,1), (1,1), (2,1)
    # Bottom row 3: (0,-1), (1,-1), (2,-1)
    # Total 11.
    shape_D = [
        axial_to_pixel(0,0), axial_to_pixel(1,0), axial_to_pixel(2,0), axial_to_pixel(3,0), axial_to_pixel(4,0),
        axial_to_pixel(0,1), axial_to_pixel(1,1), axial_to_pixel(2,1),
        axial_to_pixel(0,-1), axial_to_pixel(1,-1), axial_to_pixel(2,-1)
    ]
    candidates.append(shape_D)
    
    # Shape E: 2-5-2-2?
    # Top 2: (1,1), (2,1)
    # Middle 5: (0,0), (1,0), (2,0), (3,0), (4,0)
    # Bottom 2: (1,-1), (2,-1)
    # Extra 2: (3,1), (3,-1)?
    # Let's try 1-3-4-3-1? No, 12.
    
    # Let's add a random set of 50 random connected clusters to increase diversity.
    # We'll do a simple random walk on the lattice to generate connected sets.
    for _ in range(50):
        if time.time() > deadline - 30: break
        visited = set()
        cluster = [(0,0)]
        visited.add((0,0))
        while len(cluster) < 11:
            # Pick a random point in cluster
            q, r = random.choice(cluster)
            # Pick a random neighbor
            neighbors = [(q+1, r), (q, r+1), (q-1, r+1), (q-1, r), (q, r-1), (q+1, r-1)]
            random.shuffle(neighbors)
            added = False
            for nq, nr in neighbors:
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    cluster.append((nq, nr))
                    added = True
                    break
            if not added:
                # Dead end, restart
                break
        if len(cluster) == 11:
            # Center it
            px = [axial_to_pixel(q, r) for q, r in cluster]
            candidates.append(px)

    # Optimization function for a fixed topology
    # We optimize the global rotation of the cluster and its center position.
    # The inner hexagons are fixed relative to each other (lattice positions).
    
    def optimize_topology(points):
        if time.time() > deadline - 10:
            return
        
        # Center the points
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        rel_points = [[p[0]-cx, p[1]-cy] for p in points]
        
        # Objective: Minimize S
        # Parameters: [cluster_angle, ox, oy]
        # ox, oy is the center of the outer hexagon relative to the cluster center?
        # Or cluster center relative to outer?
        # Let's fix outer center at (0,0) and move the cluster.
        # Params: [angle, tx, ty]
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            angle = params[0]
            tx = params[1]
            ty = params[2]
            
            # Transform cluster points
            a = math.radians(angle)
            ca, sa = math.cos(a), math.sin(a)
            inner = []
            for px, py in rel_points:
                nx = ca*px - sa*py + tx
                ny = sa*px + ca*py + ty
                inner.append([nx, ny, 0.0]) # Inner hexagons are not rotated individually, only the cluster.
                # Wait, the problem says "each inner hexagon may translate and rotate".
                # If we assume they are all aligned (angle 0), we might miss solutions where individual rotations help.
                # However, for tight packings, they are usually aligned.
                # Let's allow individual rotations in a second pass if needed, or just assume 0.
                # Given the "rigid body" directive, we assume they are locked.
            
            S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
            
            # Check validity
            if not all_fit(inner, 0.0, 0.0, S, 0.0):
                return S + 100.0
            
            return S

        bounds = [(0, 360), (-5, 5), (-5, 5)]
        
        try:
            de_result = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-4, seed=1, polish=False)
            if de_result.fun < best_S:
                params = de_result.x
                angle = params[0]
                tx = params[1]
                ty = params[2]
                a = math.radians(angle)
                ca, sa = math.cos(a), math.sin(a)
                inner = []
                for px, py in rel_points:
                    nx = ca*px - sa*py + tx
                    ny = sa*px + ca*py + ty
                    inner.append([nx, ny, 0.0])
                
                S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
                if all_fit(inner, 0.0, 0.0, S, 0.0):
                    update_best(inner, 0.0, 0.0, S, 0.0)
        except Exception:
            pass

    # Run optimization for each candidate
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        optimize_topology(cand)
        
    # Fallback if no valid solution found
    if not best_valid:
        # Simple linear arrangement
        sp = math.sqrt(3)
        inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        for _ in range(200):
            if all_fit(inner, 0.0, 0.0, S, 0.0):
                return inner, (0.0, 0.0), S, 0.0
            inner = [[p[0]*1.05, p[1]*1.05, p[2]] for p in inner]
            S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
        return inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
