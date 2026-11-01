import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute constants
    SQRT3 = np.sqrt(3.0)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_vertices(positions):
        """
        positions: (N, 3) array of [x, y, angle_deg]
        Returns: (N, 6, 2) array of vertices
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            x, y, ang = positions[i]
            verts[i] = hexagon_vertices(x, y, 1.0, ang)
        return verts

    def min_distance_polygons(v1, v2):
        """
        Calculate minimum distance between two convex polygons.
        If they overlap, returns negative value proportional to penetration depth (approx).
        If disjoint, returns positive distance.
        Uses SAT to find separating axis. If no separating axis, they overlap.
        For overlap depth, we approximate by finding the minimum translation vector magnitude 
        or simply return a fixed penalty if SAT fails, but for smooth optimization, 
        a better metric is needed. 
        However, for Simulated Annealing, a simple binary penalty + distance to boundary 
        or a smooth approximation of the "separation" is often sufficient.
        
        A robust smooth-ish metric:
        1. Check SAT. If separating axis found, return distance along that axis (positive).
        2. If no separating axis (overlap), return -max_penetration_depth.
        Estimating penetration depth is tricky. 
        Alternative: Use the "minimum distance" defined as:
        If disjoint: min distance between edges.
        If overlapping: - (max overlap depth).
        
        For simplicity and speed in SA, we can use a penalty based on the 
        "violation" of the separating axis conditions.
        Let d_axis be the gap along axis n. If d_axis > 0, gap is d_axis.
        If d_axis < 0 for all axes, they overlap.
        We can define the "separation" as the minimum d_axis over all candidate axes.
        If min_d > 0, they are separated by min_d.
        If min_d < 0, they are overlapping by -min_d (this is an approximation of penetration depth, 
        actually it's the minimum translation along that specific axis, which might not be the 
        true MTD, but it's a valid smooth-ish proxy for local optimization).
        
        Let's use: separation = min( d_axis for all axes )
        If separation > 0: disjoint, distance = separation.
        If separation < 0: overlapping, "distance" = separation (negative).
        This creates a continuous function (mostly) that goes to 0 at contact.
        """
        min_sep = np.inf
        
        # Axes from v1
        for i in range(6):
            j = (i + 1) % 6
            edge = v1[j] - v1[i]
            norm = np.linalg.norm(edge)
            if norm < 1e-10: continue
            normal = np.array([-edge[1], edge[0]]) / norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap: how much they are separated along this axis
            # If max1 < min2, gap = min2 - max1
            # If max2 < min1, gap = min1 - max2
            # If overlapping, gap is negative
            gap1 = min2 - max1
            gap2 = min1 - max2
            
            # We want the minimum gap. If any gap is positive, they are separated.
            # The "separation" along this axis is max(gap1, gap2) if disjoint, else min(gap1, gap2)??
            # Actually, if they overlap, both gap1 and gap2 are negative.
            # The penetration depth along this axis is max(gap1, gap2) (closer to 0).
            # Wait, if they overlap, the "gap" is negative.
            # Let's define sep_axis = max(min2 - max1, min1 - max2).
            # If sep_axis > 0, they are separated by sep_axis.
            # If sep_axis < 0, they are overlapping by -sep_axis (approx).
            sep_axis = max(min2 - max1, min1 - max2)
            
            if sep_axis < min_sep:
                min_sep = sep_axis
        
        # Axes from v2
        for i in range(6):
            j = (i + 1) % 6
            edge = v2[j] - v2[i]
            norm = np.linalg.norm(edge)
            if norm < 1e-10: continue
            normal = np.array([-edge[1], edge[0]]) / norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            sep_axis = max(min2 - max1, min1 - max2)
            
            if sep_axis < min_sep:
                min_sep = sep_axis
                
        return min_sep

    def calculate_cost(positions):
        """
        Calculate total cost: Outer Side Length + Penalty for overlaps.
        """
        verts = get_vertices(positions)
        
        # Calculate Outer Side Length
        # We need the minimal regular hexagon containing all vertices.
        # To allow rotation of the outer hexagon, we would need to optimize over angle too.
        # For speed in SA, let's fix outer hexagon angle to 0 initially, or optimize it separately.
        # Actually, the task allows outer hexagon rotation. 
        # Let's assume outer hexagon is centered at (0,0) and we can rotate it.
        # However, rotating the outer hexagon is equivalent to rotating the inner configuration.
        # So we can fix outer angle to 0 and just find the best S.
        # But wait, the inner hexagons can rotate too.
        
        # Let's compute S for angle 0 outer hexagon.
        # S = max over all vertices v of max(|v_x|, |v_y|*2/SQRT3, |v_x| + |v_y|/SQRT3)
        # This is for a "flat-top" hexagon (vertices at 0, 60, ...).
        # The formula derived earlier:
        # S >= |x|
        # S >= |y| * 2 / SQRT3
        # S >= |x| + |y| / SQRT3  (This was for specific orientation)
        
        # Let's re-verify the "hexagonal radius" formula.
        # For a regular hexagon with side S, centered at origin, flat-top (vertex at (S,0)):
        # The boundary is defined by 3 pairs of parallel lines.
        # Line 1: x = S
        # Line 2: x = -S
        # Line 3: SQRT3*y + x = SQRT3*S  (Normal (1, SQRT3)) -> (1, SQRT3).dot(p) = SQRT3*S
        # Line 4: -SQRT3*y - x = SQRT3*S
        # Line 5: SQRT3*y - x = SQRT3*S
        # Line 6: -SQRT3*y + x = SQRT3*S
        
        # Wait, vertex at (S,0). Next vertex at (S/2, S*SQRT3/2).
        # Edge between (S,0) and (S/2, S*SQRT3/2).
        # Vector: (-S/2, S*SQRT3/2). Normal: (S*SQRT3/2, S/2) -> (SQRT3, 1).
        # Dot product with (S,0) is SQRT3*S.
        # So constraint: SQRT3*x + y <= SQRT3*S.
        
        # The 6 constraints for flat-top hexagon (vertex at 0 deg):
        # 1. x <= S
        # 2. -x <= S
        # 3. SQRT3*x + y <= SQRT3*S  => x + y/SQRT3 <= S
        # 4. -SQRT3*x - y <= SQRT3*S => -x - y/SQRT3 <= S
        # 5. -SQRT3*x + y <= SQRT3*S => -x + y/SQRT3 <= S
        # 6. SQRT3*x - y <= SQRT3*S  => x - y/SQRT3 <= S
        
        # So S must be >= max( |x|, |x + y/SQRT3|, |x - y/SQRT3| )
        
        max_s = 0.0
        for i in range(N):
            for vx, vy in verts[i]:
                s1 = abs(vx)
                s2 = abs(vx + vy / SQRT3)
                s3 = abs(vx - vy / SQRT3)
                s_req = max(s1, s2, s3)
                if s_req > max_s:
                    max_s = s_req
                    
        # Calculate Overlap Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = min_distance_polygons(verts[i], verts[j])
                if sep < 0:
                    # Overlap. Penalty should be large and smooth.
                    # Use quadratic penalty to encourage resolution
                    penalty += (sep ** 2) * 100.0
                    # Or just linear with large weight
                    # penalty += -sep * 10.0
                    
        return max_s + penalty

    def step(positions, temperature):
        """
        Perform a random move.
        Returns new_positions, delta_cost
        """
        new_positions = positions.copy()
        
        # Pick a random hexagon
        i = random.randint(0, N - 1)
        
        # Random move: translate and rotate
        dx = random.gauss(0, temperature * 0.5)
        dy = random.gauss(0, temperature * 0.5)
        d_angle = random.gauss(0, temperature * 10.0) # degrees
        
        new_positions[i, 0] += dx
        new_positions[i, 1] += dy
        new_positions[i, 2] += d_angle
        
        return new_positions

    # Initialize
    # Start with a valid hexagonal packing to ensure we are in a good basin
    # Hex grid coordinates
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = np.zeros((N, 3))
    for i, (q, r) in enumerate(coords):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_positions[i, 0] = x
        init_positions[i, 1] = y
        init_positions[i, 2] = 0.0
        
    current_positions = init_positions
    current_cost = calculate_cost(current_positions)
    
    best_positions = current_positions.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    T = T_start
    iterations = 0
    
    while time.time() < deadline:
        if T < T_end:
            break
            
        new_positions = step(current_positions, T)
        new_cost = calculate_cost(new_positions)
        
        delta_cost = new_cost - current_cost
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current_positions = new_positions
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = current_positions.copy()
                
        T *= cooling_rate
        iterations += 1
        
        # Check time periodically
        if iterations % 100 == 0 and time.time() > deadline:
            break

    # Final Polish: Local minimization using Nelder-Mead on the best solution found
    # This helps fine-tune the positions to reduce the outer side length further.
    
    def polish_objective(params):
        pos = params.reshape((N, 3))
        # Ensure angles are within reasonable bounds? No, they wrap.
        # But for NM, it's better to keep them continuous.
        return calculate_cost(pos)

    try:
        from scipy.optimize import minimize
        
        # Initial guess is best_positions
        # Bounds are not strictly necessary for NM, but we can clip if needed.
        result = minimize(
            polish_objective,
            best_positions.flatten(),
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 2000}
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_positions = result.x.reshape((N, 3))
            
    except Exception:
        pass

    # Construct return value
    inner = []
    for i in range(N):
        x, y, ang = best_positions[i]
        inner.append([float(x), float(y), float(ang)])
        
    # Outer hexagon
    # Center (0,0)
    # Side length is best_cost (if penalty is 0). 
    # Wait, best_cost includes penalty. If penalty > 0, the solution is invalid.
    # We need to extract the actual side length from the best valid configuration.
    # If best_cost > 0 and the penalty part is non-zero, we should return the side length 
    # calculated from the vertices, but the evaluator will mark it invalid if they overlap.
    # To be safe, if penalty is significant, we should return the initial valid configuration 
    # or the best valid one found during SA.
    
    # Let's re-evaluate the best_positions to get the pure side length and check validity.
    verts = get_vertices(best_positions)
    max_s = 0.0
    for i in range(N):
        for vx, vy in verts[i]:
            s1 = abs(vx)
            s2 = abs(vx + vy / SQRT3)
            s3 = abs(vx - vy / SQRT3)
            s_req = max(s1, s2, s3)
            if s_req > max_s:
                max_s = s_req
                
    # Check validity of best_positions
    is_valid = True
    for i in range(N):
        for j in range(i + 1, N):
            sep = min_distance_polygons(verts[i], verts[j])
            if sep < -1e-5: # Allow tiny numerical errors
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback to initial valid configuration
        # Recalculate side length for init
        init_verts = get_vertices(init_positions)
        init_max_s = 0.0
        for i in range(N):
            for vx, vy in init_verts[i]:
                s1 = abs(vx)
                s2 = abs(vx + vy / SQRT3)
                s3 = abs(vx - vy / SQRT3)
                s_req = max(s1, s2, s3)
                if s_req > init_max_s:
                    init_max_s = s_req
                    
        inner = []
        for i in range(N):
            x, y, ang = init_positions[i]
            inner.append([float(x), float(y), float(ang)])
        side = float(init_max_s)
    else:
        side = float(max_s)

    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
