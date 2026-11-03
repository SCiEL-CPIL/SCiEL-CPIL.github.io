import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Standard orientation: vertices at 0, 60, 120...
    # "Flat-top" usually means a side is horizontal at the top/bottom.
    # Vertices at 30, 90, 150, 210, 270, 330 degrees for flat-top?
    # Let's stick to the definition in the prompt's seed:
    # Seed used: range(0, 360, 60) -> 0, 60, 120...
    # This is "pointy-top" (vertex at 0).
    # The prompt says "Unit regular hexagons... side length 1".
    # The orientation doesn't matter for the minimum side length as long as it's consistent,
    # but the outer hexagon orientation matters relative to the inner ones.
    # We will optimize the inner angles, so the base orientation is just a reference.
    
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def generate_hexagons_6fold(params):
        """
        Generate 12 hexagons with 6-fold symmetry.
        params: [x1, y1, a1, x2, y2, a2]
        Sector 0 contains Hex 1 and Hex 2.
        Other sectors are rotations of Sector 0 by 60, 120, ... 300 degrees.
        """
        x1, y1, a1, x2, y2, a2 = params
        
        # Hex 1 and Hex 2 in sector 0
        verts1 = get_hex_verts(x1, y1, 1.0, a1)
        verts2 = get_hex_verts(x2, y2, 1.0, a2)
        
        all_verts = [verts1, verts2]
        all_positions = [(x1, y1), (x2, y2)]
        all_angles = [a1, a2]
        
        for k in range(1, 6):
            angle_rad = math.radians(k * 60.0)
            cos_k = math.cos(angle_rad)
            sin_k = math.sin(angle_rad)
            rot_k = np.array([[cos_k, -sin_k], [sin_k, cos_k]])
            
            # Rotate positions and vertices
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            
            new_p1 = rot_k @ p1
            new_p2 = rot_k @ p2
            
            # Rotate angles
            new_a1 = (a1 + k * 60.0) % 360.0
            new_a2 = (a2 + k * 60.0) % 360.0
            
            new_v1 = get_hex_verts(new_p1[0], new_p1[1], 1.0, new_a1)
            new_v2 = get_hex_verts(new_p2[0], new_p2[1], 1.0, new_a2)
            
            all_verts.append(new_v1)
            all_verts.append(new_v2)
            all_positions.append((new_p1[0], new_p1[1]))
            all_angles.append(new_a1)
            all_verts.append(new_v2)
            all_positions.append((new_p2[0], new_p2[1]))
            all_angles.append(new_a2)
            
        return all_verts, all_positions, all_angles

    def objective_6fold(params):
        if time.time() > deadline - 5:
            return 1000.0
            
        verts, positions, angles = generate_hexagons_6fold(params)
        
        # Check collisions
        # Due to symmetry, we only need to check:
        # 1. (0, 1) : Hex1 vs Hex2 in Sector 0
        # 2. (0, 6) : Hex1(S0) vs Hex1(S1)
        # 3. (0, 7) : Hex1(S0) vs Hex2(S1)
        # 4. (1, 6) : Hex2(S0) vs Hex1(S1)
        # 5. (1, 7) : Hex2(S0) vs Hex2(S1)
        
        penalty = 0.0
        pairs_to_check = [(0, 1), (0, 6), (0, 7), (1, 6), (1, 7)]
        
        for i, j in pairs_to_check:
            gap = sat_gap(verts[i], verts[j])
            if gap < 0:
                penalty += 100.0 * gap**2
                # Early exit if severe overlap
                if penalty > 10.0:
                    return 1000.0 + penalty
        
        if penalty > 1e-6:
            return 1000.0 + penalty
        
        # Calculate outer side length
        side = calc_min_outer_side(np.array(positions), np.array(angles), 0.0)
        
        return side

    # Initial guesses
    sqrt3 = math.sqrt(3)
    
    # Guess 1: Regular Ring of 12
    # Radius approx 2.5?
    # Hex width is sqrt(3). Distance between centers in ring of 12.
    # Circumference approx 12 * 1.5 = 18. R = 18 / (2 * pi) ~ 2.8
    r_ring = 2.8
    init_ring = []
    for k in range(6):
        theta1 = math.radians(k * 60.0 + 15.0)
        theta2 = math.radians(k * 60.0 - 15.0)
        # Just put them at specific angles in the sector
        # Sector 0 is [0, 60]. Put them at 15 and 45?
        # Let's try placing them at 15 deg and 45 deg relative to sector start.
        x1 = r_ring * math.cos(math.radians(15))
        y1 = r_ring * math.sin(math.radians(15))
        a1 = 0.0
        
        x2 = r_ring * math.cos(math.radians(45))
        y2 = r_ring * math.sin(math.radians(45))
        a2 = 0.0
        
        init_ring.append([x1, y1, a1, x2, y2, a2])

    # Guess 2: Two Rings (Inner 6, Outer 6)
    # Inner ring radius ~ 1.5, Outer ~ 3.0
    r_inner = 1.8
    r_outer = 3.2
    init_two_rings = []
    for k in range(6):
        # Inner hex at 0 deg, Outer at 30 deg (staggered)
        x1 = r_inner * math.cos(math.radians(0))
        y1 = r_inner * math.sin(math.radians(0))
        a1 = 0.0
        
        x2 = r_outer * math.cos(math.radians(30))
        y2 = r_outer * math.sin(math.radians(30))
        a2 = 0.0
        
        init_two_rings.append([x1, y1, a1, x2, y2, a2])

    # Guess 3: Tight Cluster
    # Center-ish
    init_cluster = []
    for k in range(6):
        x1 = 1.0
        y1 = 1.0
        a1 = 0.0
        x2 = 1.0
        y2 = -1.0
        a2 = 0.0
        init_cluster.append([x1, y1, a1, x2, y2, a2])

    best_score = float('inf')
    best_params = None

    initial_guesses = [
        init_ring[0],
        init_two_rings[0],
        init_cluster[0]
    ]

    # Add some random perturbations to initial guesses for diversity
    for _ in range(10):
        base = initial_guesses[np.random.randint(len(initial_guesses))]
        noise = np.random.normal(0, 0.5, 6)
        initial_guesses.append(base + noise)

    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is robust for small non-smooth problems
            res = minimize(
                objective_6fold, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < best_score:
                # Verify validity strictly
                verts, positions, angles = generate_hexagons_6fold(res.x)
                valid = True
                pairs_to_check = [(0, 1), (0, 6), (0, 7), (1, 6), (1, 7)]
                for pi, pj in pairs_to_check:
                    if sat_gap(verts[pi], verts[pj]) < -1e-5:
                        valid = False
                        break
                
                if valid:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            pass

    # If we have a good result, try a final polish with a slightly larger initial simplex or different method?
    # Or just run a few more random restarts if time permits.
    
    restarts_done = len(initial_guesses)
    while time.time() < deadline - 20 and restarts_done < 50:
        # Random restart
        # Random radius between 1 and 4
        r1 = np.random.uniform(1.0, 3.5)
        r2 = np.random.uniform(1.0, 4.0)
        a1 = np.random.uniform(0, 60)
        a2 = np.random.uniform(0, 60)
        
        x1 = r1 * math.cos(math.radians(a1))
        y1 = r1 * math.sin(math.radians(a1))
        x2 = r2 * math.cos(math.radians(a2))
        y2 = r2 * math.sin(math.radians(a2))
        
        x0 = [x1, y1, 0.0, x2, y2, 0.0]
        
        try:
            res = minimize(
                objective_6fold, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < best_score:
                verts, positions, angles = generate_hexagons_6fold(res.x)
                valid = True
                pairs_to_check = [(0, 1), (0, 6), (0, 7), (1, 6), (1, 7)]
                for pi, pj in pairs_to_check:
                    if sat_gap(verts[pi], verts[pj]) < -1e-5:
                        valid = False
                        break
                
                if valid:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            pass
        restarts_done += 1

    # Finalize
    if best_params is not None:
        verts, final_positions, final_angles = generate_hexagons_6fold(best_params)
        final_side = calc_min_outer_side(np.array(final_positions), np.array(final_angles), 0.0)
    else:
        # Fallback: Simple ring
        final_positions = []
        final_angles = []
        r = 3.0
        for k in range(12):
            theta = math.radians(k * 30.0)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            final_positions.append([x, y])
            final_angles.append(0.0)
        final_positions = np.array(final_positions)
        final_angles = np.array(final_angles)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

