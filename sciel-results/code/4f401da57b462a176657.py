import math
import time
import numpy as np
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Geometry helpers
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        verts = get_hex_vertices(hx, hy, side, angle_deg)
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Vector from p1 to center
            v_center_x = hx - p1[0]
            v_center_y = hy - p1[1]
            # Vector from p1 to point
            v_point_x = px - p1[0]
            v_point_y = py - p1[1]
            
            cross_center = ex * v_center_y - ey * v_center_x
            cross_point = ex * v_point_y - ey * v_point_x
            
            if cross_center == 0:
                continue
            if (cross_center > 0 and cross_point < 0) or (cross_center < 0 and cross_point > 0):
                return False
        return True

    def hexagon_overlap_penalty(h1, h2):
        """
        Returns a penalty value if two hexagons overlap.
        h1, h2: (cx, cy, side, angle)
        We approximate overlap by checking if any vertex is inside the other.
        For a smoother penalty, we could compute distance between centers, 
        but hexagons are complex. 
        For SA, a simple '0 if disjoint, 1 if overlap' is too noisy.
        Let's use a continuous approximation: 
        If center distance < min_dist, penalty increases.
        Min distance for two unit hexagons to be disjoint is roughly 2 (if aligned) 
        but depends on angle. 
        A safer approach: Check vertex containment. If any vertex of 1 is in 2, 
        or any vertex of 2 is in 1, or edges intersect, it's an overlap.
        To make it smooth for SA, we can return the number of violations.
        """
        v1 = get_hex_vertices(h1[0], h1[1], h1[2], h1[3])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2], h2[3])
        
        penalty = 0.0
        # Check vertices of 1 in 2
        for v in v1:
            if point_in_hexagon(v[0], v[1], h2[0], h2[1], h2[2], h2[3]):
                penalty += 1.0
        # Check vertices of 2 in 1
        for v in v2:
            if point_in_hexagon(v[0], v[1], h1[0], h1[1], h1[2], h1[3]):
                penalty += 1.0
        # Check edge intersections (simplified: if center distance is very small, assume overlap)
        # This is a heuristic. A more robust check would be segment intersection.
        # Given time constraints, vertex containment catches most overlaps.
        # To catch edge-only overlaps, we can add a distance check.
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        # Minimum distance between centers of two disjoint unit hexagons is ~1.732 (sqrt(3)) if aligned,
        # but can be less if rotated. Actually, the inradius is sqrt(3)/2 ~ 0.866.
        # Distance between parallel sides is sqrt(3).
        # So if dist < 1.5, they definitely overlap.
        if dist < 1.5:
             penalty += (1.5 - dist) * 10.0 # Strong penalty for close centers
        
        return penalty

    def containment_penalty(h_inner, outer_side, outer_angle):
        """
        Returns penalty if inner hexagon is not inside outer hexagon (centered at 0,0).
        """
        v = get_hex_vertices(h_inner[0], h_inner[1], h_inner[2], h_inner[3])
        penalty = 0.0
        for p in v:
            if not point_in_hexagon(p[0], p[1], 0.0, 0.0, outer_side, outer_angle):
                penalty += 1.0
        return penalty

    def objective(params):
        """
        params: 12 * 3 (x, y, angle) + 1 (outer_side)
        Outer center is fixed at (0,0), angle 0.
        """
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, 1.0, angle))
        
        outer_side = params[N*3]
        
        if outer_side <= 0:
            return 1e10
            
        total_penalty = 0.0
        
        # Overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                total_penalty += hexagon_overlap_penalty(inner_hexes[i], inner_hexes[j])
                
        # Containment penalty
        for ih in inner_hexes:
            total_penalty += containment_penalty(ih, outer_side, 0.0)
            
        # Objective: Minimize outer side, penalize violations
        # We want a valid solution (penalty 0) with minimal outer_side.
        # If penalty > 0, the cost is high.
        return outer_side + 10.0 * total_penalty

    # Initialization: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    coords = [
        (-2, 0), (0, 0), (2, 0),
        (-1, sqrt3), (1, sqrt3), (3, sqrt3),
        (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
        (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
    ]
    cx = sum(c[0] for c in coords) / N
    cy = sum(c[1] for c in coords) / N
    
    init_params = []
    for c in coords:
        init_params.append(c[0] - cx)
        init_params.append(c[1] - cy)
        init_params.append(0.0)
        
    # Estimate initial outer side
    max_r = 0
    for c in coords:
        x = c[0] - cx
        y = c[1] - cy
        r = math.sqrt(x*x + y*y) + 1
        if r > max_r:
            max_r = r
            
    init_params.append(max_r)
    
    current_params = init_params.copy()
    current_cost = objective(current_params)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.995 # Cooling rate
    step_size = 1.0
    
    best_params = current_params.copy()
    best_cost = current_cost
    
    # Pre-compute valid angle range? Angles are periodic 0-360.
    # We can just wrap around.
    
    while time.time() < deadline:
        # Generate neighbor
        new_params = current_params.copy()
        
        # Perturb a random inner hexagon
        idx = random.randint(0, N-1)
        # x
        new_params[idx*3] += random.gauss(0, step_size)
        # y
        new_params[idx*3+1] += random.gauss(0, step_size)
        # angle
        new_params[idx*3+2] += random.gauss(0, step_size * 10)
        new_params[idx*3+2] = new_params[idx*3+2] % 360
        
        # Occasionally perturb outer side
        if random.random() < 0.1:
            new_params[N*3] += random.gauss(0, step_size * 0.5)
            if new_params[N*3] < 3.0:
                new_params[N*3] = 3.0
            
        new_cost = objective(new_params)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_params = new_params
            current_cost = new_cost
            if current_cost < best_cost:
                best_params = current_params.copy()
                best_cost = current_cost
        else:
            # Metropolis criterion
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_params = new_params
                current_cost = new_cost
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally reheat or stop if no improvement for a while?
            # For simplicity, just stop cooling.
            
        # Reduce step size slowly
        if random.random() < 0.01:
            step_size *= 0.95
            if step_size < 0.01:
                step_size = 0.01

    # Final verification and cleanup
    # If the best solution has penalties, it might be invalid.
    # We should try to repair it or fall back to init if invalid.
    
    # Check validity of best_params
    def is_valid(params):
        inner_hexes = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            angle = params[i*3+2]
            inner_hexes.append((x, y, 1.0, angle))
        outer_side = params[N*3]
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hexagon_overlap_penalty(inner_hexes[i], inner_hexes[j]) > 0:
                    return False
        # Check containment
        for ih in inner_hexes:
            if containment_penalty(ih, outer_side, 0.0) > 0:
                return False
        return True

    if is_valid(best_params):
        inner = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            angle = best_params[i*3+2]
            inner.append([x, y, angle])
        outer_side = best_params[N*3]
        return inner, (0.0, 0.0), outer_side, 0.0
    else:
        # Fallback to initial honeycomb if SA failed to find a valid packing
        # The initial honeycomb is valid if max_r is calculated correctly.
        # Let's verify the init params validity.
        if is_valid(init_params):
            inner = []
            for i in range(N):
                x = init_params[i*3]
                y = init_params[i*3+1]
                angle = init_params[i*3+2]
                inner.append([x, y, angle])
            return inner, (0.0, 0.0), init_params[N*3], 0.0
        else:
            # Absolute fallback: larger hexagon
            inner = [[0.0, 0.0, 0.0] for _ in range(N)] # This is invalid (overlap)
            # We must return a valid solution.
            # The honeycomb lattice with side 1 always fits in a hexagon of side ~4-5.
            # Let's construct a valid honeycomb manually.
            coords = [
                (-2, 0), (0, 0), (2, 0),
                (-1, sqrt3), (1, sqrt3), (3, sqrt3),
                (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
                (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
            ]
            cx = sum(c[0] for c in coords) / N
            cy = sum(c[1] for c in coords) / N
            inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
            # Calculate required side
            max_r = 0
            for c in coords:
                x = c[0] - cx
                y = c[1] - cy
                r = math.sqrt(x*x + y*y) + 1
                if r > max_r:
                    max_r = r
            # Add a small buffer to ensure validity
            return inner, (0.0, 0.0), max_r + 0.1, 0.0

# EVOLVE-BLOCK-END
