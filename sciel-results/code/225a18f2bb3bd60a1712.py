"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def is_disjoint(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def calc_outer_side_exact(hexagons):
        """
        Calculate the minimum side length of a regular hexagon containing all given hexagons.
        The outer hexagon is centered at the origin.
        We find the minimum S such that all vertices are inside.
        
        The condition for a point (x,y) to be inside a regular hexagon of side S, 
        rotated by angle alpha, centered at origin:
        The apothem is A = S * SQRT3 / 2.
        The distance from origin to the edge in direction theta is A / cos(delta), 
        where delta is the angle between theta and the normal to the edge.
        
        Actually, simpler:
        For a fixed outer rotation alpha, the required S is:
        S = (2 / SQRT3) * max_{v in vertices} max_{k=0..5} (v . n_k(alpha))
        where n_k are the outward normals of the outer hexagon edges.
        
        The function f(alpha) = max_proj(alpha) is the maximum projection of all inner vertices
        onto the 6 outward normals of the outer hexagon.
        We want to minimize f(alpha).
        
        f(alpha) is a "max of sinusoids" function. Its minimum occurs when two sinusoids intersect
        or at a peak of a single sinusoid (which is a local max, so we ignore).
        So we just need to check the intersections of the sinusoids formed by each vertex's projection
        onto the rotating normals.
        
        However, a simpler robust method:
        The critical angles for the outer hexagon orientation are when a vertex of the inner cluster
        lies on a vertex of the outer hexagon, or a vertex of the inner cluster is perpendicular to an edge.
        
        Given the complexity of exact critical angle finding for 11 hexagons (66 vertices),
        a fine grid search (step 0.1 deg) is fast enough and robust, followed by a local refinement.
        """
        # Collect all vertices
        all_verts = []
        for h in hexagons:
            all_verts.extend(h)
        
        # If no vertices, return 0
        if not all_verts:
            return 0.0
            
        # We search for the best outer rotation.
        # The function is periodic with 60 degrees.
        # We do a coarse search, then a fine search.
        
        def max_proj_for_rotation(alpha_deg):
            max_p = 0.0
            # Normals of the outer hexagon
            # Flat top hexagon normals are at 30, 90, 150, 210, 270, 330
            # Rotated by alpha
            base_angles = [30, 90, 150, 210, 270, 330]
            for a in base_angles:
                rad = math.radians(a + alpha_deg)
                nx = math.cos(rad)
                ny = math.sin(rad)
                # Project all vertices
                # Optimization: vectorize if possible, but N=11*6=66 is small
                for vx, vy in all_verts:
                    p = abs(vx * nx + vy * ny)
                    if p > max_p:
                        max_p = p
            return max_p

        best_side = float('inf')
        best_alpha = 0.0
        
        # Coarse search
        for alpha in np.arange(0, 60, 1.0):
            mp = max_proj_for_rotation(alpha)
            side = 2.0 * mp / SQRT3
            if side < best_side:
                best_side = side
                best_alpha = alpha
                
        # Fine search around best_alpha
        for alpha in np.arange(best_alpha - 1, best_alpha + 1, 0.01):
            if alpha < 0 or alpha >= 60: continue
            mp = max_proj_for_rotation(alpha)
            side = 2.0 * mp / SQRT3
            if side < best_side:
                best_side = side
                best_alpha = alpha
                
        return best_side

    # Initial configuration: Hexagonal tiling cluster
    # 1 center, 6 around, 4 more on the perimeter
    # Axial coordinates (q, r) for flat-top hexagons
    # Center (0,0)
    # Neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Perimeter: (2,0), (2,-1), (2,-2), (1,-2) ... wait, let's pick a symmetric set of 4
    # (2,0), (2,-2), (0,-2), (-2,-1) ? 
    # Let's use a standard "flower" of 7, and add 4 to the outer ring.
    # To minimize bounding box, we want a compact shape.
    # A hexagon of radius 2 has 19 cells. We need 11.
    # A hexagon of radius 1 has 7 cells.
    # Add 4 to the radius 2 ring.
    
    coords = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (2,-1), (1,-2), (-1,2) # Arbitrary 4 on the outer ring
    ]
    
    def axial_to_cartesian(q, r, spacing=SQRT3):
        # Flat top
        x = spacing * (q + r/2.0)
        y = spacing * 0.5 * r * 2 # Wait, y = 1.5 * r * size?
        # For unit hexagons, distance between centers in x is SQRT3, in y is 1.5
        # x = SQRT3 * (q + r/2)
        # y = 1.5 * r
        x = SQRT3 * (q + r/2.0)
        y = 1.5 * r
        return x, y

    initial_params = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r)
        initial_params.extend([x, y, 0.0])
        
    # Center the configuration
    cx_avg = sum(p[0] for p in initial_params[::3]) / N
    cy_avg = sum(p[1] for p in initial_params[1::3]) / N
    
    # Shift to origin
    for i in range(N):
        initial_params[i*3] -= cx_avg
        initial_params[i*3+1] -= cy_avg

    best_result = None
    best_score = float('inf')
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    min_T = 1e-4
    steps = 0
    max_steps = 50000
    
    current_params = np.array(initial_params)
    current_hexagons = [get_hex_vertices(current_params[i*3], current_params[i*3+1], current_params[i*3+2]) for i in range(N)]
    current_score = calc_outer_side_exact(current_hexagons)
    best_params = current_params.copy()
    best_score = current_score
    best_hexagons = [h.copy() for h in current_hexagons] # Deep copy
    
    # Check validity of initial
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(current_hexagons[i], current_hexagons[j]):
                valid = False
                break
        if not valid: break
    if not valid:
        # If initial is invalid, we must fix it or start with a looser config
        # For now, assume the tiling is valid (it is, with spacing SQRT3 they touch, so disjoint is false? 
        # "Disjoint" usually means interiors do not intersect. Touching is allowed?
        # The prompt says "disjoint". In geometry, disjoint sets have empty intersection.
        # If they touch, intersection is non-empty (the boundary).
        # Usually in packing problems, "non-overlapping" allows touching.
        # The SAT check `max1 < min2` returns True if strictly separated.
        # If they touch, max1 == min2, so it returns False (overlap).
        # We need to adjust the tolerance or the initial spacing.
        # Let's use spacing = SQRT3 * 1.01 to ensure strict disjointness initially.
        pass

    # Re-initialize with slight gap to ensure validity
    spacing = SQRT3 * 1.05
    initial_params = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing)
        initial_params.extend([x, y, 0.0])
    cx_avg = sum(p[0] for p in initial_params[::3]) / N
    cy_avg = sum(p[1] for p in initial_params[1::3]) / N
    for i in range(N):
        initial_params[i*3] -= cx_avg
        initial_params[i*3+1] -= cy_avg
        
    current_params = np.array(initial_params)
    current_hexagons = [get_hex_vertices(current_params[i*3], current_params[i*3+1], current_params[i*3+2]) for i in range(N)]
    
    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(current_hexagons[i], current_hexagons[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback: very loose grid
        spacing = 3.0
        initial_params = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing)
            initial_params.extend([x, y, 0.0])
        cx_avg = sum(p[0] for p in initial_params[::3]) / N
        cy_avg = sum(p[1] for p in initial_params[1::3]) / N
        for i in range(N):
            initial_params[i*3] -= cx_avg
            initial_params[i*3+1] -= cy_avg
        current_params = np.array(initial_params)
        current_hexagons = [get_hex_vertices(current_params[i*3], current_params[i*3+1], current_params[i*3+2]) for i in range(N)]

    current_score = calc_outer_side_exact(current_hexagons)
    best_params = current_params.copy()
    best_score = current_score
    best_hexagons = [h for h in current_hexagons]
    
    while T > min_T and steps < max_steps:
        if time.time() > deadline - 5.0:
            break
            
        # Perturb a random hexagon
        i = random.randint(0, N-1)
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 5.0)
        
        new_params = current_params.copy()
        new_params[i*3] += dx
        new_params[i*3+1] += dy
        new_params[i*3+2] += da
        
        # Check validity of new hexagon against all others
        new_hex = get_hex_vertices(new_params[i*3], new_params[i*3+1], new_params[i*3+2])
        valid = True
        for j in range(N):
            if i == j: continue
            if not is_disjoint(new_hex, current_hexagons[j]):
                valid = False
                break
        
        if valid:
            # Update hexagons list for this iteration (only i changes)
            new_hexagons = current_hexagons.copy()
            new_hexagons[i] = new_hex
            new_score = calc_outer_side_exact(new_hexagons)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                current_params = new_params
                current_hexagons = new_hexagons
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    best_hexagons = current_hexagons.copy()
        else:
            # Reject, but we can still "cool"
            pass
            
        T *= cooling_rate
        steps += 1

    # Final Result Construction
    inner_list = []
    for i in range(N):
        inner_list.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    
    # Recalculate final score precisely
    final_hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
    final_side = calc_outer_side_exact(final_hexagons)
    
    # Find the exact angle for the outer hexagon
    # We need to return the angle.
    # Re-run the fine search to get the angle
    all_verts = []
    for h in final_hexagons:
        all_verts.extend(h)
        
    def max_proj_for_rotation(alpha_deg):
        max_p = 0.0
        base_angles = [30, 90, 150, 210, 270, 330]
        for a in base_angles:
            rad = math.radians(a + alpha_deg)
            nx = math.cos(rad)
            ny = math.sin(rad)
            for vx, vy in all_verts:
                p = abs(vx * nx + vy * ny)
                if p > max_p:
                    max_p = p
        return max_p

    best_alpha = 0.0
    min_proj = float('inf')
    for alpha in np.arange(0, 60, 0.1):
        mp = max_proj_for_rotation(alpha)
        if mp < min_proj:
            min_proj = mp
            best_alpha = alpha
            
    # Refine
    for alpha in np.arange(best_alpha - 0.1, best_alpha + 0.1, 0.001):
        if alpha < 0 or alpha >= 60: continue
        mp = max_proj_for_rotation(alpha)
        if mp < min_proj:
            min_proj = mp
            best_alpha = alpha
            
    final_side = 2.0 * min_proj / SQRT3

    return (inner_list, (0.0, 0.0), final_side, best_alpha)

# EVOLVE-BLOCK-END
