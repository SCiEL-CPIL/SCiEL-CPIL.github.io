import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def min_dist_to_edge(p, a, b):
        """Minimum distance from point p to segment ab."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def smooth_overlap_penalty(poly1, poly2):
        """
        Calculates a smooth penalty for overlap between two convex hexagons.
        Returns 0 if disjoint (with margin), >0 if overlapping.
        Uses the minimum distance from vertices of one poly to edges of the other.
        If min_dist < 0 (which is impossible for distance, but we use a proxy), 
        actually distance is always >= 0. 
        To detect overlap, we check if min_dist is 0 AND they are not just touching.
        However, for a smooth penalty, we can use:
        penalty = sum( max(0, threshold - dist)^2 ) for all vertex-edge pairs?
        That's expensive (36 pairs).
        
        Better approach:
        If the polygons are disjoint, min_dist > 0.
        If they touch, min_dist = 0.
        If they overlap, min_dist = 0 (vertex on edge or inside).
        
        We need a function that is 0 when disjoint, and increases as they overlap.
        Standard approach: 
        1. Check if any vertex of P1 is inside P2. If so, penalty = 1.
        2. Check if any vertex of P2 is inside P1. If so, penalty = 1.
        3. Check edge intersections. If any, penalty = 1.
        
        This is discontinuous. To make it smooth for SA:
        We can define the "overlap depth" roughly.
        If disjoint, depth = 0.
        If overlapping, depth > 0.
        
        Approximation:
        min_dist = min( min_dist_to_edge(v, e) for v in P1, e in P2 )
        min_dist2 = min( min_dist_to_edge(v, e) for v in P2, e in P1 )
        
        If min_dist > 0 and min_dist2 > 0, they are disjoint. Penalty = 0.
        If min_dist == 0 or min_dist2 == 0, they might be touching or overlapping.
        
        To distinguish touching from overlapping:
        If a vertex of P1 is strictly inside P2, it's an overlap.
        
        For SA, a simple binary penalty (0 or 1) often works if the step size is small enough,
        but a smooth one is better.
        
        Let's use a "soft" check:
        For each vertex v in P1, calculate distance to P2 edges.
        If dist == 0, check if v is inside P2.
        If inside, penalty += 1.
        
        Actually, let's just use a large constant penalty for any overlap.
        SA can handle discontinuities if the temperature is high enough initially.
        """
        # Fast bounding box
        min_x1, min_y1 = np.min(poly1, axis=0)
        max_x1, max_y1 = np.max(poly1, axis=0)
        min_x2, min_y2 = np.min(poly2, axis=0)
        max_x2, max_y2 = np.max(poly2, axis=0)
        
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return 0.0

        # Point in Polygon check (Ray casting)
        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check vertices of v1 in v2
        for pt in poly1:
            if point_in_poly(pt, poly2):
                return 1.0
        
        # Check vertices of v2 in v1
        for pt in poly2:
            if point_in_poly(pt, poly1):
                return 1.0
        
        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i+1)%6]
            for j in range(6):
                p3 = poly2[j]
                p4 = poly2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return 1.0
        
        return 0.0

    # Strategy:
    # 1. Initialize with a honeycomb lattice (touching).
    # 2. Run Simulated Annealing to optimize positions and angles.
    # 3. Use a penalty for overlaps.
    # 4. Coarse search over outer hexagon rotation.

    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    sqrt3 = math.sqrt(3)

    # Generate initial positions: Honeycomb lattice
    # v1 = (0, sqrt3), v2 = (1.5, sqrt3/2)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)

    # Verify initial validity
    polys = [get_hex_verts(init_positions[i][0], init_positions[i][1], 1.0, 0.0) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if smooth_overlap_penalty(polys[i], polys[j]) > 0:
                valid = False
                break
        if not valid: break

    if valid:
        score = calc_min_outer_side(init_positions, init_angles, 0.0)
        if score < best_score:
            best_score = score
            best_positions = init_positions.copy()
            best_angles = init_angles.copy()

    # Simulated Annealing
    if best_positions is not None:
        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if penalty > 1.0:
                        break
                    p = smooth_overlap_penalty(polys[i], polys[j])
                    if p > 0:
                        penalty += p
                if penalty > 1.0:
                    break
            
            if penalty > 0:
                return penalty * 100.0 + calc_min_outer_side(positions, angles, 0.0)
            return calc_min_outer_side(positions, angles, 0.0)

        # SA Parameters
        T = 0.5
        T_min = 0.001
        cooling_rate = 0.999
        current_params = np.concatenate([best_positions.flatten(), best_angles])
        current_score = objective(current_params)
        best_params_sa = current_params.copy()
        best_score_sa = current_score
        
        step_size = 0.2
        max_iter = 5000
        
        for k in range(max_iter):
            if time.time() > deadline - 10:
                break
            if T < T_min:
                break
            
            # Perturb
            new_params = current_params + np.random.normal(0, step_size, size=current_params.shape)
            
            # Keep angles in [0, 60)
            new_params[2*N:] = new_params[2*N:] % 60.0
            
            new_score = objective(new_params)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_params = new_params
                current_score = new_score
                
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_params_sa = current_params.copy()
            
            T *= cooling_rate
            # Adaptive step size?
            if k % 100 == 0:
                step_size *= 0.99

        # Update best if SA found something better
        if best_score_sa < best_score:
            # Verify validity
            new_positions = best_params_sa[:2*N].reshape(N, 2)
            new_angles = best_params_sa[2*N:]
            polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if smooth_overlap_penalty(polys[i], polys[j]) > 0:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_score = best_score_sa
                best_positions = new_positions
                best_angles = new_angles

    # Coarse search over outer hexagon rotation
    # The objective function depends on outer_angle.
    # We can try a few angles: 0, 5, 10, 15, 30 (symmetry)
    # Note: The inner hexagons' angles are fixed relative to their local frame.
    # Rotating the outer hexagon is equivalent to rotating the inner hexagons by -outer_angle.
    # But we can just re-evaluate the score for different outer angles.
    
    # Actually, the score is defined as the side length of the outer hexagon.
    # The interface asks for: inner, center, side, angle.
    # So we can output the best outer angle found.
    
    for outer_ang in [0, 5, 10, 15, 30]:
        if time.time() > deadline - 5:
            break
        s = calc_min_outer_side(best_positions, best_angles, outer_ang)
        if s < best_score:
            best_score = s
            best_outer_angle = outer_ang

    # Final refinement of the best configuration with the best outer angle fixed?
    # No, the score is just the value. The configuration is positions+angles.
    # The outer angle is just a parameter for the score calculation.
    # But wait, if we change outer angle, the "containing" hexagon rotates.
    # The inner hexagons do NOT rotate with it.
    # So we found a set of inner hexagons that fits in an outer hexagon of side S at angle A.
    # We return S and A.
    
    final_positions = best_positions
    final_angles = best_angles
    final_side = best_score
    final_outer_angle = best_outer_angle

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

