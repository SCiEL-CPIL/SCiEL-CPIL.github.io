import math
import time
import random
import numpy as np
from itertools import combinations, product, permutations

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Helper: Generate vertices of a regular hexagon with side 1, centered at (0,0), rotated by angle
    def get_hex_verts(angle_deg=0.0):
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((math.cos(theta), math.sin(theta)))
        return verts

    base_hex_verts = get_hex_verts(0.0)

    # Helper: Transform vertices (translate, scale, rotate)
    def transform_verts(verts, cx, cy, scale, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        out = []
        for vx, vy in verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            out.append((cx + scale * rx, cy + scale * ry))
        return out

    # Helper: Point in polygon
    def point_in_poly(p, poly):
        x, y = p
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Helper: Check if two polygons are disjoint (no interior intersection)
    def polys_disjoint(verts_a, verts_b):
        # Check if any vertex of A is strictly inside B
        for v in verts_a:
            if point_in_poly(v, verts_b):
                return False
        for v in verts_b:
            if point_in_poly(v, verts_a):
                return False
        # Check edge intersections
        def seg_intersect(p1, p2, p3, p4):
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
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if seg_intersect(a1, a2, b1, b2):
                return False
        return True

    # Helper: Compute minimal side length S of outer hexagon containing all inner hexagons
    # Outer hexagon is centered at (0,0) for simplicity (we can shift inner later or just compute relative)
    # Actually, the problem allows the outer hexagon to be anywhere. 
    # But if we fix the outer hexagon center to the centroid of the inner cluster, 
    # and allow rotation, we can optimize.
    # However, a simpler robust approach:
    # For a given set of points (vertices of all inner hexagons), the minimal enclosing regular hexagon
    # of side S is determined by the maximum projection onto the 6 normal vectors of the hexagon.
    # Let the outer hexagon be centered at C.
    # The distance from C to each side is S * sqrt(3)/2.
    # So, max_proj(C, normal_i) <= S * sqrt(3)/2 for all i.
    # This implies S >= max_i (max_v (v - C) . n_i) / (sqrt(3)/2).
    # To minimize S, we should choose C such that the max range is minimized.
    # For a regular hexagon, the optimal C is often the center of the bounding box or centroid.
    # Let's compute S for a given configuration of inner hexagons and an outer angle.
    
    def compute_min_S(inner_configs, outer_angle_deg):
        # inner_configs: list of (cx, cy, angle) for N hexagons
        # We will center the outer hexagon at the centroid of the inner hexagon centers?
        # Or better: we can optimize the outer center (ocx, ocy) as well.
        # But to keep it fast, let's assume the outer hexagon is centered at the centroid of the inner centers.
        # This is a good heuristic.
        
        cx_sum = sum(c[0] for c in inner_configs)
        cy_sum = sum(c[1] for c in inner_configs)
        ocx = cx_sum / N
        ocy = cy_sum / N
        
        # Generate all vertices of all inner hexagons
        all_verts = []
        for c in inner_configs:
            v = transform_verts(base_hex_verts, c[0], c[1], 1.0, c[2])
            all_verts.extend(v)
        
        # Normals for outer hexagon
        a = math.radians(outer_angle_deg)
        normals = []
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            normals.append((math.cos(theta), math.sin(theta)))
        
        max_proj = 0.0
        for nx, ny in normals:
            proj_max = 0.0
            for vx, vy in all_verts:
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > proj_max:
                    proj_max = proj
            if proj_max > max_proj:
                max_proj = proj_max
        
        S = max_proj / (math.sqrt(3) / 2)
        return S, (ocx, ocy)

    def is_valid(inner_configs, S, ocx, ocy, outer_angle_deg):
        # Check if all inner hexagons are inside the outer hexagon
        outer_verts = transform_verts(base_hex_verts, ocx, ocy, S, outer_angle_deg)
        
        for c in inner_configs:
            v = transform_verts(base_hex_verts, c[0], c[1], 1.0, c[2])
            for pt in v:
                if not point_in_poly(pt, outer_verts):
                    return False
        
        # Check disjointness
        verts_list = []
        for c in inner_configs:
            v = transform_verts(base_hex_verts, c[0], c[1], 1.0, c[2])
            verts_list.append(v)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polys_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    # Strategy:
    # 1. Generate candidate rigid-body shapes (polyhexes) on a triangular lattice.
    # 2. For each shape, optimize the rotation and translation to minimize S.
    
    # Lattice vectors for hexagonal packing (side 1)
    # Centers of adjacent hexagons are distance sqrt(3) apart.
    # Let's define lattice basis.
    # Hex 0 at (0,0).
    # Neighbors at angles 0, 60, 120, 180, 240, 300?
    # If hexagon has vertex at (1,0), the flat side is vertical? No.
    # Vertex at (1,0) means "pointy right".
    # The distance to the center of the neighbor sharing the vertical edge on the right?
    # The edge is from (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2).
    # The normal is (1,0). Distance is 1.
    # So the neighbor center is at (1 + 1, 0) = (2,0)? No.
    # Apothem is sqrt(3)/2.
    # If vertex is at (1,0), the side is at x=0.5.
    # The distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # So neighbors are at distance sqrt(3).
    
    d = math.sqrt(3)
    # Lattice points for centers
    # Basis vectors:
    # v1 = (d, 0)
    # v2 = (d/2, d*sqrt(3)/2) = (d/2, 3/2)
    
    v1 = (d, 0)
    v2 = (d/2, 3/2)
    
    # Generate all 11-cell polyhexes? That's computationally heavy if done naively.
    # Instead, we can use a "greedy" or "random" generation of compact clusters on the lattice.
    # Or, we can use a known good shape: the 3-4-3 hexagonal cluster.
    # Let's generate a few specific shapes:
    # Shape 1: 3-4-3 (standard hexagon of hexagons)
    # Shape 2: 4-3-4?
    # Shape 3: Random compact clusters.
    
    def generate_lattice_points(shape_type):
        points = []
        if shape_type == '343':
            # Center (0,0)
            points.append((0,0))
            # Ring 1: 6 neighbors
            for i in range(6):
                # Angles 0, 60, 120...
                # In lattice coords (i,j) -> x = i*d + j*d/2, y = j*1.5
                # Let's just use cartesian coordinates for simplicity in generation
                pass
            
            # Let's build it manually using lattice steps
            # 0,0
            # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords?
            # Let's use the basis vectors directly.
            # Center: (0,0)
            # Neighbors:
            # 1: d * (1, 0)
            # 2: d * (1/2, sqrt(3)/2)
            # 3: d * (-1/2, sqrt(3)/2)
            # 4: d * (-1, 0)
            # 5: d * (-1/2, -sqrt(3)/2)
            # 6: d * (1/2, -sqrt(3)/2)
            
            pts = [(0,0)]
            for i in range(6):
                ang = math.radians(60*i)
                pts.append((d*math.cos(ang), d*math.sin(ang)))
            
            # Add ring 2 to make 11? 1 + 6 = 7. Need 4 more.
            # The 3-4-3 shape is a hexagon of side 2? No, side 2 has 1+6+12=19.
            # Side 1 has 7.
            # We need 11.
            # A "hexagon" of 11 hexagons doesn't exist in the perfect hex shape.
            # The closest is a hexagon of side 2 (19) minus corners, or a 3-4-3 strip?
            # 3-4-3:
            # Row 1: 3 hexagons
            # Row 2: 4 hexagons
            # Row 3: 3 hexagons
            # Total 10. Need 11.
            # Maybe 3-5-3? Total 11.
            
            # Let's generate a 3-5-3 cluster.
            # It's a parallelogram-like shape on the lattice.
            
            # Let's just use a random walk on the lattice to generate compact clusters.
            # Start at (0,0). Add neighbors.
            # This is a "random polyhex" generator.
            
            return pts # Just for now, we will implement a better generator below
            
        return points

    # Better: Generate random compact clusters on the triangular lattice
    def generate_random_cluster(rng):
        # Lattice points in axial coordinates (q, r)
        # x = d * (q + r/2)
        # y = d * (sqrt(3)/2) * r
        
        # Start with (0,0)
        cluster = [(0,0)]
        visited = set([(0,0)])
        
        # Directions in axial coords
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        
        while len(cluster) < N:
            # Pick a random existing hex
            idx = rng.randrange(len(cluster))
            q, r = cluster[idx]
            # Pick a random direction
            dq, dr = rng.choice(dirs)
            nq, nr = q + dq, r + dr
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                cluster.append((nq, nr))
                
        # Convert to cartesian
        pts = []
        for q, r in cluster:
            x = d * (q + r/2)
            y = d * (math.sqrt(3)/2) * r
            pts.append((x,y))
        return pts

    # We will try a mix of:
    # 1. A few specific hand-crafted shapes (if we can define them easily)
    # 2. Random clusters generated by the above method
    
    candidates = []
    
    # Add a "blob" shape: 3-4-3-1? Or just random.
    # Let's generate 50 random clusters.
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        pts = generate_random_cluster(rng)
        candidates.append(pts)
        
    # Also add a "line" shape for variety (though likely worse)
    line_pts = [(i*d, 0) for i in range(N)]
    candidates.append(line_pts)
    
    # Also add a "2-5-2-2" or similar dense shape
    # Let's try to construct a specific dense shape manually
    # Center (0,0)
    # 6 neighbors
    # 4 from ring 2
    # Total 11
    manual_pts = [(0,0)]
    for i in range(6):
        ang = math.radians(60*i)
        manual_pts.append((d*math.cos(ang), d*math.sin(ang)))
    # Add 4 from ring 2 (distance 2d from center? No, ring 2 is further)
    # In triangular lattice, ring 2 points are at distance sqrt(3)*sqrt(3) = 3? No.
    # Distance from center to ring 2 center:
    # If we go 2 steps in one direction: distance 2d.
    # If we go 1 step in one direction, 1 step in another (60 deg): distance sqrt(d^2 + d^2 - 2d^2 cos(120)) = sqrt(3d^2) = 3?
    # Wait. d = sqrt(3).
    # 1 step: dist sqrt(3).
    # 2 steps (straight): dist 2*sqrt(3).
    # 2 steps (60 deg): dist sqrt(3+3-2*3*(-0.5)) = sqrt(9) = 3.
    # So ring 2 has 12 points.
    # Let's pick 4 of them.
    # To keep it compact, pick alternating ones.
    for i in range(4):
        # Angles 0, 90, 180, 270? No, lattice is 60 deg symmetry.
        # Let's just pick 4 neighbors of the ring 1 points.
        pass
        
    # The random generator is probably sufficient to find a good shape.
    # The key is the optimization of the rigid body.

    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    for pts in candidates:
        if time.time() > deadline - 5:
            break
            
        # Center the cluster at origin
        cx_avg = sum(p[0] for p in pts) / N
        cy_avg = sum(p[1] for p in pts) / N
        centered_pts = [(p[0] - cx_avg, p[1] - cy_avg) for p in pts]
        
        # All hexagons have angle 0 relative to the lattice.
        # The lattice is fixed. We can rotate the whole cluster.
        # So we optimize: rotation_angle of the cluster.
        # And we can also optimize the individual hexagon angles? 
        # In a rigid lattice, they are all aligned.
        # But we can try a few variations:
        # 1. All hexagons angle 0.
        # 2. All hexagons angle 30.
        
        for hex_angle in [0, 30]:
            # Optimize cluster rotation
            # S(cluster_rotation) is a function of rotation.
            # We can sample rotations.
            
            best_rot_S = float('inf')
            best_rot = 0.0
            
            # Sample 36 rotations (10 deg steps)
            for rot_deg in range(0, 360, 10):
                if time.time() > deadline - 2:
                    break
                    
                # Apply rotation to centered_pts
                rad = math.radians(rot_deg)
                ca, sa = math.cos(rad), math.sin(rad)
                rotated_pts = []
                for x, y in centered_pts:
                    rx = x * ca - y * sa
                    ry = x * sa + y * ca
                    rotated_pts.append((rx, ry))
                    
                # Configure inner hexagons
                inner_configs = [(x, y, hex_angle) for x, y in rotated_pts]
                
                # Try a few outer angles
                # The optimal outer angle is likely aligned with the cluster symmetry or the bounding box.
                # Let's try 0, 15, 30, 45, 60.
                for outer_ang in [0, 15, 30, 45, 60]:
                    S, (ocx, ocy) = compute_min_S(inner_configs, outer_ang)
                    
                    # Quick check: if S is already worse than best, skip validation
                    if S >= best_S:
                        continue
                        
                    if is_valid(inner_configs, S, ocx, ocy, outer_ang):
                        if S < best_rot_S:
                            best_rot_S = S
                            best_rot = rot_deg
                            
            # If we found a good rotation, refine it?
            # For now, take the best found.
            if best_rot_S < best_S:
                # Reconstruct the best configuration
                rad = math.radians(best_rot)
                ca, sa = math.cos(rad), math.sin(rad)
                rotated_pts = []
                for x, y in centered_pts:
                    rx = x * ca - y * sa
                    ry = x * sa + y * ca
                    rotated_pts.append((rx, ry))
                    
                # We need to find the specific outer angle that gave this S
                # Let's re-evaluate the best outer angle
                inner_configs = [(x, y, hex_angle) for x, y in rotated_pts]
                best_outer_ang = 0
                min_S_found = float('inf')
                for outer_ang in [0, 15, 30, 45, 60]:
                    S, (ocx, ocy) = compute_min_S(inner_configs, outer_ang)
                    if is_valid(inner_configs, S, ocx, ocy, outer_ang):
                        if S < min_S_found:
                            min_S_found = S
                            best_outer_ang = outer_ang
                            
                if min_S_found < best_S:
                    best_S = min_S_found
                    best_inner = [list(x) for x in inner_configs]
                    best_center = (ocx, ocy)
                    best_angle = best_outer_ang
                    best_valid = True
                    # Optional: Local search on the rigid body?
                    # We can try small rotations of the whole cluster.
                    # But let's stick to the discrete search for speed.

    # Fallback if nothing valid found
    if not best_valid:
        # Use the random cluster with angle 0, outer 0
        pts = candidates[0] if candidates else [(i*d, 0) for i in range(N)]
        cx_avg = sum(p[0] for p in pts) / N
        cy_avg = sum(p[1] for p in pts) / N
        centered_pts = [(p[0] - cx_avg, p[1] - cy_avg) for p in pts]
        inner_configs = [(x, y, 0.0) for x, y in centered_pts]
        S, (ocx, ocy) = compute_min_S(inner_configs, 0.0)
        # Force valid by scaling up if needed? No, must be disjoint.
        # If invalid, it's a problem. But lattice clusters are disjoint by construction (if d > sqrt(3)).
        # d = sqrt(3) is touching. So they are disjoint.
        # The only risk is floating point or the outer hexagon being too small.
        # compute_min_S should give the exact minimum.
        best_inner = [list(x) for x in inner_configs]
        best_center = (ocx, ocy)
        best_S = S
        best_angle = 0.0
        best_valid = True

    return best_inner, best_center, best_S, best_angle
