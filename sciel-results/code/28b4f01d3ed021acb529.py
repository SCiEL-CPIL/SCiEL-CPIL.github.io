<<<<<<< SEARCH
def local_search(positions, angles, outer_center, outer_side, outer_angle, max_iters=1000):
    """Improve the packing via local search: try small perturbations."""
    positions = [p[:] for p in positions]
    angles = angles[:]
    
    current_cost = 0.0
    # Cost: sum of overlaps (should be 0 for feasible)
    # We want to minimize overlap
    
    def compute_overlap_penalty():
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = math.sqrt(3)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        # Also penalize going outside
        for i in range(n):
            verts = hexagon_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    penalty += 10.0
        return penalty
    
    current_penalty = compute_overlap_penalty()
    
    for it in range(max_iters):
        if current_penalty < 1e-6:
            break
        
        # Try perturbing a random hexagon
        idx = random.randint(0, len(positions)-1)
        old_pos = positions[idx][:]
        old_angle = angles[idx]
        
        # Random perturbation
        positions[idx][0] += random.gauss(0, 0.1)
        positions[idx][1] += random.gauss(0, 0.1)
        angles[idx] = (old_angle + random.gauss(0, 5.0)) % 360
        
        new_penalty = compute_overlap_penalty()
        
        if new_penalty < current_penalty:
            current_penalty = new_penalty
        else:
            # Revert
            positions[idx] = old_pos
            angles[idx] = old_angle
    
    if current_penalty < 1e-6:
        return positions, angles, True
    
    return positions, angles, False
=======
def local_search(positions, angles, outer_center, outer_side, outer_angle, max_iters=1000):
    """Improve the packing via local search: try small perturbations."""
    positions = [p[:] for p in positions]
    angles = angles[:]
    
    def compute_overlap_penalty():
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = math.sqrt(3)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        # Also penalize going outside
        for i in range(n):
            verts = hexagon_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    penalty += 10.0
        return penalty
    
    current_penalty = compute_overlap_penalty()
    
    for it in range(max_iters):
        if current_penalty < 1e-6:
            break
        
        # Try perturbing a random hexagon
        idx = random.randint(0, len(positions)-1)
        old_pos = positions[idx][:]
        old_angle = angles[idx]
        
        # Random perturbation - only translate, keep angle fixed
        positions[idx][0] += random.gauss(0, 0.05)
        positions[idx][1] += random.gauss(0, 0.05)
        
        new_penalty = compute_overlap_penalty()
        
        if new_penalty < current_penalty:
            current_penalty = new_penalty
        else:
            # Revert
            positions[idx] = old_pos
    
    if current_penalty < 1e-6:
        return positions, angles, True
    
    return positions, angles, False

def optimize_side_length(positions, angles, outer_center, outer_angle, initial_side, max_iters=50):
    """
    Given a configuration of hexagons, find the minimal outer side length
    that contains all of them.
    
    For a set of points, the minimal enclosing hexagon (with given orientation)
    can be found by checking the support function in 6 directions.
    """
    # The outer hexagon must contain all inner hexagons.
    # For each inner hexagon, compute its support function values in the 6 directions
    # of the outer hexagon's normals.
    
    # The outer hexagon with center (cx, cy), side s, and angle theta has the property that
    # a point (px, py) is inside if and only if for all 6 normals:
    # nx_i * (px - cx) + ny_i * (py - cy) <= s * sqrt(3)/2
    
    # For an inner hexagon centered at (xi, yi) with angle 0, its vertices are at
    # (xi + cos(k*pi/3), yi + sin(k*pi/3)) for k=0..5
    
    # The maximum support value for the inner hexagon in direction i is:
    # max over vertices v of (nx_i * (vx - cx) + ny_i * (vy - cy))
    # = nx_i * (xi - cx) + ny_i * (yi - cy) + max over vertices v of (nx_i * vx' + ny_i * vy')
    # where v' = v - (xi, yi)
    
    # For a unit hexagon with angle 0, the support function in direction (nx, ny) is:
    # max over k of (nx * cos(k*pi/3) + ny * sin(k*pi/3))
    
    # Compute the 6 normal directions for the outer hexagon
    normals = []
    for i in range(6):
        theta = math.radians(outer_angle) + (i + 0.5) * math.pi / 3
        normals.append((math.cos(theta), math.sin(theta)))
    
    # For each inner hexagon, compute its support values in the 6 directions
    max_supports = [0.0] * 6
    
    for pos, ang in zip(positions, angles):
        # Compute the support function of this hexagon in each normal direction
        for i, (nx, ny) in enumerate(normals):
            # The support value of a hexagon centered at (px, py) with angle ang in direction (nx, ny)
            # is: nx*px + ny*py + support_unit_hex(nx, ny, ang)
            # where support_unit_hex is the support function of a unit hexagon at origin with angle ang
            
            # For a unit hexagon with angle ang, the vertices are at:
            # (cos(ang + k*pi/3), sin(ang + k*pi/3)) for k=0..5
            max_dot = 0.0
            for k in range(6):
                vx = math.cos(math.radians(ang) + k * math.pi / 3)
                vy = math.sin(math.radians(ang) + k * math.pi / 3)
                dot = nx * vx + ny * vy
                if dot > max_dot:
                    max_dot = dot
            
            support_val = nx * pos[0] + ny * pos[1] + max_dot
            if support_val > max_supports[i]:
                max_supports[i] = support_val
    
    # The outer hexagon must satisfy:
    # nx_i * (0 - cx) + ny_i * (0 - cy) + s * sqrt(3)/2 >= max_supports[i]
    # for all i, where (cx, cy) = outer_center
    
    # If outer_center is (0, 0), this simplifies to:
    # s * sqrt(3)/2 >= max_supports[i] for all i
    # s >= 2 * max(max_supports[i]) / sqrt(3)
    
    if outer_center[0] == 0.0 and outer_center[1] == 0.0:
        required_side = 2.0 * max(max_supports) / math.sqrt(3)
    else:
        # General case: solve for s such that all constraints are satisfied
        # nx_i * (0 - cx) + ny_i * (0 - cy) + s * sqrt(3)/2 >= max_supports[i]
        # s * sqrt(3)/2 >= max_supports[i] + nx_i * cx + ny_i * cy
        required_side = 0.0
        for i, (nx, ny) in enumerate(normals):
            rhs = max_supports[i] + nx * outer_center[0] + ny * outer_center[1]
            s_i = 2.0 * rhs / math.sqrt(3)
            if s_i > required_side:
                required_side = s_i
    
    return required_side
>>>>>>> REPLACE

