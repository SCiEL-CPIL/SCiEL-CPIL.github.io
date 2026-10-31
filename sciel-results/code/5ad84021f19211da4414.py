import math
import time
import random
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 11
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def construct():
    deadline = DEADLINE
    
    # Precompute useful constants
    SQRT3 = math.sqrt(3)
    APOTHEM = SQRT3 / 2.0  # apothem of unit hexagon
    
    def hex_vertices(cx, cy, S, angle_deg):
        """Return 6 vertices of regular hexagon."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts
    
    def support_function_dist(verts_a, verts_b):
        """
        Compute the minimum distance between two convex polygons (hexagons).
        If they overlap, return negative value.
        Uses separating axis theorem for convex polygons.
        Returns the signed minimum distance (negative = overlap).
        """
        # For two convex polygons, if no separating axis exists, they overlap.
        # We compute the maximum of the minimum separations over all edge normals.
        # Actually, let's use a simpler approach: check if any vertex of A is inside B
        # or any vertex of B is inside A, or edges intersect.
        # For optimization, we need a smooth constraint. Let's use:
        # distance = min over all vertex pairs and edge-vertex distances
        # But that's not smooth. Instead, use the fact that for convex sets,
        # they are disjoint iff there exists a separating line.
        
        # Practical approach: compute the minimum of (projection onto each edge normal)
        # This gives us a "separation" measure.
        
        # Get all edge normals from both polygons
        def get_edge_normals(verts):
            normals = []
            n = len(verts)
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % n]
                # Edge direction
                dx, dy = x2 - x1, y2 - y1
                # Normal (perpendicular)
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1e-12:
                    continue
                nx, ny = -dy / length, dx / length
                normals.append((nx, ny))
            return normals
        
        normals_a = get_edge_normals(verts_a)
        normals_b = get_edge_normals(verts_b)
        
        max_min_sep = -1e10  # We want the maximum over axes of the minimum separation
        
        for nx, ny in normals_a + normals_b:
            # Project all vertices of A
            proj_a = [(v[0] * nx + v[1] * ny) for v in verts_a]
            proj_b = [(v[0] * nx + v[1] * ny) for v in verts_b]
            
            min_a, max_a = min(proj_a), max(proj_a)
            min_b, max_b = min(proj_b), max(proj_b)
            
            # Separation on this axis
            if max_a < min_b:
                sep = min_b - max_a
            elif max_b < min_a:
                sep = min_a - max_b
            else:
                sep = 0.0
            
            if sep > max_min_sep:
                max_min_sep = sep
        
        # If max_min_sep > 0, polygons are separated (disjoint)
        # If max_min_sep == 0, they touch or overlap
        # For our constraint, we need separation >= 0 (disjoint, touching allowed)
        # But for strict non-overlap in optimization, we need separation >= epsilon
        # Actually, the constraint is that they should be disjoint (non-overlapping).
        # The minimum distance between the two convex sets is max_min_sep if they don't overlap.
        # If they overlap, max_min_sep will be 0 (since some axis has 0 separation).
        # Wait, that's not right. If they overlap, for every axis the projections overlap,
        # so sep = 0 for all axes, and max_min_sep = 0.
        # If they don't overlap, there exists at least one axis where sep > 0.
        # So max_min_sep > 0 means disjoint, max_min_sep == 0 means overlapping or touching.
        
        # For the constraint, we want max_min_sep >= 0 (which is always true)
        # Actually we want them to NOT overlap, i.e., the actual distance >= 0.
        # The issue is that max_min_sep as computed above is 0 when overlapping.
        # We need a different measure. Let's compute the actual minimum distance.
        
        # For convex polygons, min distance = max over separating axes of the gap.
        # But if they overlap, min distance = 0 (or negative if we define signed distance).
        # For optimization, let's use: constraint = max_min_sep (which is >= 0 always)
        # and require max_min_sep >= 0, which is trivially satisfied.
        # That doesn't help. We need to enforce non-overlap.
        
        # Better: the constraint should be that the polygons do not overlap.
        # A convex polygon A is inside B iff all vertices of A are inside B AND no edges intersect.
        # For optimization, let's use a penalty-based approach or a different formulation.
        
        # Alternative: use the fact that two convex polygons overlap iff
        # there is no separating axis. So the constraint "no overlap" is equivalent to
        # "there exists a separating axis", which is an OR condition - not smooth.
        
        # Practical approach for optimization: use a smooth approximation.
        # Define f = max over all axes of (separation on that axis).
        # If f > 0, they are separated. If f <= 0, they might overlap.
        # But f is always >= 0 by construction (sep is max of 0 and actual gap).
        
        # Let me reconsider. The correct measure:
        # For each axis, compute the gap = max(0, min_b - max_a) if max_a < min_b,
        # or max(0, min_a - max_b) if max_b < min_a, else 0.
        # The polygons are disjoint iff there exists an axis with gap > 0.
        # The minimum distance between the polygons = max over axes of gap.
        # Wait no, that's the width of the union, not the distance.
        
        # I'm overcomplicating this. Let me use a direct approach:
        # Compute the minimum distance between any vertex of A and any edge of B,
        # and any vertex of B and any edge of A, and between vertex pairs.
        # Take the minimum of all these. If any vertex is inside the other polygon, distance = 0 (overlap).
        
        def point_in_convex_poly(px, py, poly):
            """Check if point is inside convex polygon using cross products."""
            n = len(poly)
            sign = 0
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if cross > 1e-9:
                    if sign < 0:
                        return False
                    sign = 1
                elif cross < -1e-9:
                    if sign > 0:
                        return False
                    sign = -1
            return True
        
        # Check if any vertex of A is inside B
        for v in verts_a:
            if point_in_convex_poly(v[0], v[1], verts_b):
                return -1.0  # Overlapping
        for v in verts_b:
            if point_in_convex_poly(v[0], v[1], verts_a):
                return -1.0  # Overlapping
        
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
            if seg_intersect(verts_a[i], verts_a[(i+1)%6], verts_b[i], verts_b[(i+1)%6]):
                return -1.0
            if seg_intersect(verts_a[i], verts_a[(i+1)%6], verts_b[(i+1)%6], verts_b[i]):
                return -1.0
        
        # They are disjoint. Compute minimum distance.
        min_dist = 1e10
        # Vertex-vertex distances
        for v1 in verts_a:
            for v2 in verts_b:
                d = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
                if d < min_dist:
                    min_dist = d
        # Vertex-edge distances
        def point_to_segment_dist(px, py, ax, ay, bx, by):
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                return math.sqrt((px-ax)**2 + (py-ay)**2)
            t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            cx, cy = ax + t*dx, ay + t*dy
            return math.sqrt((px-cx)**2 + (py-cy)**2)
        
        for v in verts_a:
            for i in range(6):
                d = point_to_segment_dist(v[0], v[1], verts_b[i][0], verts_b[i][1], 
                                          verts_b[(i+1)%6][0], verts_b[(i+1)%6][1])
                if d < min_dist:
                    min_dist = d
        for v in verts_b:
            for i in range(6):
                d = point_to_segment_dist(v[0], v[1], verts_a[i][0], verts_a[i][1], 
                                          verts_a[(i+1)%6][0], verts_a[(i+1)%6][1])
                if d < min_dist:
                    min_dist = d
        
        return min_dist
    
    def compute_S_and_check(x, ocx, ocy, oangle):
        """
        x: flat array of length 3*N: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN]
        Compute the required outer side length S and check validity.
        Returns (S, is_valid)
        """
        inner_verts = []
        for i in range(N):
            ix, iy, iang = x[3*i], x[3*i+1], x[3*i+2]
            iv = hex_vertices(ix, iy, 1.0, iang)
            inner_verts.append(iv)
        
        # Compute minimum S needed
        a = math.radians(oangle)
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
                S_needed = max_proj / APOTHEM
                if S_needed > max_S:
                    max_S = S_needed
        
        # Check mutual disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dist = support_function_dist(inner_verts[i], inner_verts[j])
                if dist < -1e-6:  # Overlapping
                    return max_S, False
        
        return max_S, True
    
    def objective_and_constraints(x, ocx, ocy, oangle):
        """
        For scipy minimize with constraints.
        We minimize S (outer side length) subject to non-overlap constraints.
        But S depends on x, so we can't directly minimize S with constraints on x.
        Instead, let's use a penalty approach:
        objective = S + lambda * max(0, -min_dist)^2 for all pairs
        """
        inner_verts = []
        for i in range(N):
            ix, iy, iang = x[3*i], x[3*i+1], x[3*i+2]
            iv = hex_vertices(ix, iy, 1.0, iang)
            inner_verts.append(iv)
        
        a = math.radians(oangle)
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
                S_needed = max_proj / APOTHEM
                if S_needed > max_S:
                    max_S = S_needed
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = support_function_dist(inner_verts[i], inner_verts[j])
                if dist < 0:
                    penalty += (-dist) ** 2
        
        return max_S + 10.0 * penalty
    
    def make_initial_configs():
        """Generate diverse initial configurations."""
        configs = []
        
        # Config 1: Hexagonal cluster (1 center + 6 ring + 4 in second ring)
        sp = SQRT3  # distance between centers of touching same-orientation hexagons
        hex_cfg = [0.0, 0.0, 0.0]  # center
        for i in range(6):
            theta = math.radians(60 * i)
            hex_cfg.extend([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        # Second ring: 4 hexagons at specific positions
        # For a hexagonal lattice, second ring positions are at distance 2*sp in 6 directions
        # but we only need 4 more. Place them at angles 0, 60, 120, 180
        for i in range(4):
            theta = math.radians(60 * i)
            hex_cfg.extend([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        configs.append(np.array(hex_cfg[:3*N]))
        
        # Config 2: Linear arrangement
        lin_cfg = []
        for k in range(N):
            lin_cfg.extend([(k - N//2) * sp, 0.0, 0.0])
        configs.append(np.array(lin_cfg))
        
        # Config 3: Two rows
        row_cfg = []
        for k in range(6):
            row_cfg.extend([(k - 2.5) * sp, 0.0, 0.0])
        for k in range(5):
            row_cfg.extend([(k - 2) * sp, sp * SQRT3, 0.0])  # offset row
        configs.append(np.array(row_cfg[:3*N]))
        
        # Config 4: Compact triangle-ish
        tri_cfg = []
        positions = [(0,0), (sp,0), (sp/2, sp*SQRT3/2), 
                      (2*sp,0), (3*sp/2, sp*SQRT3/2), (sp/2, -sp*SQRT3/2),
                      (-sp/2, sp*SQRT3/2), (-sp, 0), 
                      (2*sp, sp*SQRT3/2), (3*sp, 0), (sp, sp*SQRT3)]
        for px, py in positions[:N]:
            tri_cfg.extend([px, py, 0.0])
        configs.append(np.array(tri_cfg))
        
        # Random configs
        rng = random.Random(42)
        for s in range(10):
            rand_cfg = []
            for _ in range(N):
                x = rng.uniform(-3, 3)
                y = rng.uniform(-3, 3)
                ang = rng.choice([0, 30, 60, 90, 120, 150])
                rand_cfg.extend([x, y, ang])
            configs.append(np.array(rand_cfg))
        
        return configs
    
    best_S = float('inf')
    best_x = None
    best_ocx, best_ocy, best_oangle = 0.0, 0.0, 0.0
    
    configs = make_initial_configs()
    
    for cfg_idx, x0 in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Try different outer hexagon orientations
        for oangle in [0, 30]:
            if time.time() > deadline:
                break
            
            # First, compute initial S and check validity
            S_init, valid_init = compute_S_and_check(x0, 0.0, 0.0, oangle)
            
            # If initial config is valid, we can try to optimize
            # Use penalty-based optimization
            def obj(x):
                if time.time() > deadline:
                    return 1e10
                return objective_and_constraints(x, 0.0, 0.0, oangle)
            
            # Bounds: positions in [-5, 5], angles in [0, 180)
            bounds = []
            for i in range(N):
                bounds.extend([(-5, 5), (-5, 5), (0, 180)])
            
            try:
                result = minimize(obj, x0, method='Nelder-Mead', 
                                 options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6,
                                          'adaptive': True})
                
                if time.time() > deadline:
                    break
                    
                x_opt = result.x
                S_opt, valid_opt = compute_S_and_check(x_opt, 0.0, 0.0, oangle)
                
                if valid_opt and S_opt < best_S:
                    best_S = S_opt
                    best_x = x_opt.copy()
                    best_ocx, best_ocy, best_oangle = 0.0, 0.0, oangle
                    
            except Exception:
                pass
        
        # Also try optimizing the outer center
        if best_x is not None and time.time() < deadline:
            # Fine-tune outer center for the best configuration found so far
            # Try small perturbations of outer center
            for dx in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                for dy in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                    if time.time() > deadline:
                        break
                    S_test, valid_test = compute_S_and_check(best_x, dx, dy, best_oangle)
                    if valid_test and S_test < best_S:
                        best_S = S_test
                        best_ocx, best_ocy = dx, dy
    
    # Final check: if we have a valid solution, return it
    if best_x is not None:
        inner = []
        for i in range(N):
            inner.append([best_x[3*i], best_x[3*i+1], best_x[3*i+2]])
        return inner, (best_ocx, best_ocy), best_S, best_oangle
    
    # Fallback: linear arrangement
    sp = SQRT3
    inner = []
    for k in range(N):
        inner.append([(k - N//2) * sp, 0.0, 0.0])
    return inner, (0.0, 0.0), 40.0, 0.0
