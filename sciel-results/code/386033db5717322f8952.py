import numpy as np
import time

N = 11

def get_hex_vertices(cx, cy, angle):
    # Returns 6 vertices of a unit hexagon centered at (cx, cy) with rotation angle
    # Unit hexagon vertices relative to center (flat-top, angle 0)
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + np.deg2rad(angle)
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def calc_outer_side(positions, angles):
    # positions: (N, 2)
    # angles: (N,)
    # Calculate minimal side length of a regular hexagon (centered at origin, angle 0) 
    # that contains all vertices of the inner hexagons.
    # We assume the outer hexagon is fixed at angle 0 for calculation, 
    # but we can rotate the whole system later if needed. 
    # Actually, the problem allows the outer hexagon to rotate. 
    # However, due to symmetry, we can fix the outer hexagon orientation and 
    # optimize the inner positions relative to it, or allow outer rotation.
    # Given the symmetry of the hexagon, fixing outer angle to 0 and allowing 
    # inner rotations is sufficient if we also allow the "center" of the outer 
    # hexagon to be optimized (which we do by centering the config).
    
    all_vertices = []
    for i in range(N):
        verts = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
        all_vertices.append(verts)
    
    if not all_vertices:
        return 0.0
        
    all_vertices = np.vstack(all_vertices)
    
    # Center the configuration at the origin to minimize the bounding box
    centroid = np.mean(all_vertices, axis=0)
    all_vertices -= centroid
    
    # The outer hexagon has normals at 30, 90, 150, 210, 270, 330 degrees
    # Normal vectors for a flat-top hexagon (vertices at 0, 60, ...)
    # Sides are perpendicular to these normals.
    # Distance from center to side is r = S * sqrt(3) / 2.
    # Condition: v . n <= r
    # So S >= 2/sqrt(3) * max(v . n)
    
    normals = []
    for ang in [30, 90, 150, 210, 270, 330]:
        rad = np.deg2rad(ang)
        normals.append([np.cos(rad), np.sin(rad)])
    normals = np.array(normals) # (6, 2)
    
    # Project vertices onto normals
    # all_vertices is (M, 2), normals is (6, 2)
    # We need max over all vertices and all normals of (v . n)
    # Note: v . n can be negative. We need the max absolute value?
    # No, the hexagon is centered at origin. The boundary is at r in direction n.
    # So we need max(v . n) <= r.
    # Since the hexagon is symmetric, if v . n <= r for all n, it is inside.
    # We should check both n and -n? 
    # The normals defined above cover all 6 sides.
    # v . n gives the signed distance.
    # We need max(v . n) <= r.
    
    projections = all_vertices @ normals.T # (M, 6)
    max_proj = np.max(projections)
    
    S = (2.0 / np.sqrt(3.0)) * max_proj
    
    return S, centroid

def check_overlaps(positions):
    # Check if any two hexagons overlap.
    # Distance between centers must be >= 2.
    # This is a necessary condition. Is it sufficient for non-overlapping hexagons?
    # Yes, for regular hexagons of side 1, the minimum distance between centers 
    # for non-overlap is 2.
    # Proof: The "radius" to the side is sqrt(3)/2. 
    # If two hexagons are aligned, the gap is 0 when distance is 2.
    # If rotated, the effective radius can be larger (up to 1).
    # Wait. If one hexagon is rotated 30 degrees, its vertex points towards the other.
    # Distance 2 might not be enough if orientations are different.
    # However, in optimal packings of hexagons, they are usually aligned or 
    # the rotation is such that they fit in the "valleys".
    # For simplicity and robustness, we will penalize any distance < 2.
    # We can also add a penalty for specific close approaches if needed, 
    # but distance < 2 is the primary constraint.
    
    violations = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 2.0:
                violations += (2.0 - d)**2
    return violations

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial configuration: Hexagonal cluster
    # 1 center, 6 around, 4 more
    # Let's generate a lattice of points and pick 11 closest to origin
    # Lattice vectors for hex centers (distance 2):
    # a1 = (2, 0)
    # a2 = (1, sqrt(3))
    
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = 2*i + j
            y = np.sqrt(3)*j
            pts.append([x, y])
    pts = np.array(pts)
    
    # Sort by distance to origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    init_pos = pts[idx[:N]]
    
    # Randomize angles slightly? Or keep 0.
    init_ang = np.zeros(N)
    
    best_S = float('inf')
    best_pos = init_pos.copy()
    best_ang = init_ang.copy()
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    current_pos = init_pos.copy()
    current_ang = init_ang.copy()
    
    # Initial cost
    S_curr, _ = calc_outer_side(current_pos, current_ang)
    overlap_curr = check_overlaps(current_pos)
    cost_curr = S_curr + 10.0 * overlap_curr
    
    if cost_curr < best_S: # Note: best_S should track the valid S, or the cost?
         # We want to minimize S among valid configurations.
         # If invalid, we track the best valid one.
         # Let's track best_valid_S.
         pass
    
    best_valid_S = float('inf')
    best_valid_pos = None
    
    if overlap_curr == 0:
        best_valid_S = S_curr
        best_valid_pos = current_pos.copy()
    
    iterations = 0
    while time.time() - start_time < deadline:
        iterations += 1
        if T < T_min:
            break
            
        # Pick random index
        i = np.random.randint(N)
        
        # Perturb position
        step = 0.5 * T
        dx = np.random.normal(0, step)
        dy = np.random.normal(0, step)
        
        new_pos = current_pos.copy()
        new_pos[i] = [current_pos[i][0] + dx, current_pos[i][1] + dy]
        
        # Optionally perturb angle
        # For hexagons, rotation might help, but let's stick to 0 for now 
        # or small perturbations.
        # Let's try small angle perturbations.
        new_ang = current_ang.copy()
        if np.random.random() < 0.1:
            new_ang[i] = current_ang[i] + np.random.normal(0, 10 * np.pi/180)
        
        S_new, _ = calc_outer_side(new_pos, new_ang)
        overlap_new = check_overlaps(new_pos)
        cost_new = S_new + 10.0 * overlap_new
        
        delta_cost = cost_new - cost_curr
        
        if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
            current_pos = new_pos
            current_ang = new_ang
            cost_curr = cost_new
            S_curr = S_new
            overlap_curr = overlap_new
            
            if overlap_curr == 0:
                if S_curr < best_valid_S:
                    best_valid_S = S_curr
                    best_valid_pos = current_pos.copy()
                    best_valid_ang = current_ang.copy()
        
        T *= cooling
        
    # If no valid solution found, return a large S
    if best_valid_pos is None:
        # Fallback: return a large box
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 50.0, 0.0
    
    # Center the best solution
    S_final, centroid = calc_outer_side(best_valid_pos, best_valid_ang)
    
    # The outer hexagon center should be the centroid of the configuration?
    # The problem asks for (inner, center, side, angle).
    # 'center' is the outer hexagon center.
    # We calculated S assuming the outer hexagon is at the origin.
    # But we shifted the inner vertices to center at origin.
    # So the outer hexagon center is at the negative of the centroid of the 
    # original (unshifted) best_valid_pos?
    # No, calc_outer_side shifted the vertices to origin.
    # So the outer hexagon center is at the position that makes the 
    # centered configuration valid.
    # If we shift the inner hexagons by -centroid, they fit in a hexagon at (0,0).
    # So if we place the outer hexagon at (0,0), the inner hexagons are at 
    # best_valid_pos - centroid.
    # But we can also just place the outer hexagon at 'centroid' and keep 
    # inner hexagons at 'best_valid_pos'.
    # Let's output:
    # inner: best_valid_pos (with angles)
    # center: centroid
    # side: S_final
    # angle: 0.0
    
    inner_list = []
    for i in range(N):
        inner_list.append([best_valid_pos[i][0], best_valid_pos[i][1], np.rad2deg(best_valid_ang[i])])
        
    return inner_list, (centroid[0], centroid[1]), S_final, 0.0

# EVOLVE-BLOCK-START
def construct():
    # The code above is the implementation.
    # We need to wrap it in the function construct.
    # I will paste the logic inside construct.
    
    import numpy as np
    import time
    
    N = 11
    
    def get_hex_vertices(cx, cy, angle):
        angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + np.deg2rad(angle)
        x = cx + np.cos(angles)
        y = cy + np.sin(angles)
        return np.column_stack((x, y))

    def calc_outer_side(positions, angles):
        all_vertices = []
        for i in range(N):
            verts = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
            all_vertices.append(verts)
        
        if not all_vertices:
            return 0.0, np.array([0.0, 0.0])
            
        all_vertices = np.vstack(all_vertices)
        
        centroid = np.mean(all_vertices, axis=0)
        all_vertices -= centroid
        
        normals = []
        for ang in [30, 90, 150, 210, 270, 330]:
            rad = np.deg2rad(ang)
            normals.append([np.cos(rad), np.sin(rad)])
        normals = np.array(normals)
        
        projections = all_vertices @ normals.T
        max_proj = np.max(projections)
        
        S = (2.0 / np.sqrt(3.0)) * max_proj
        
        return S, centroid

    def check_overlaps(positions):
        violations = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 2.0:
                    violations += (2.0 - d)**2
        return violations

    start_time = time.time()
    deadline = 110.0
    
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = 2*i + j
            y = np.sqrt(3)*j
            pts.append([x, y])
    pts = np.array(pts)
    
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    init_pos = pts[idx[:N]]
    
    init_ang = np.zeros(N)
    
    best_valid_S = float('inf')
    best_valid_pos = None
    best_valid_ang = None
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    current_pos = init_pos.copy()
    current_ang = init_ang.copy()
    
    S_curr, _ = calc_outer_side(current_pos, current_ang)
    overlap_curr = check_overlaps(current_pos)
    cost_curr = S_curr + 10.0 * overlap_curr
    
    if overlap_curr == 0:
        best_valid_S = S_curr
        best_valid_pos = current_pos.copy()
        best_valid_ang = current_ang.copy()
    
    while time.time() - start_time < deadline:
        if T < T_min:
            break
            
        i = np.random.randint(N)
        
        step = 0.5 * T
        dx = np.random.normal(0, step)
        dy = np.random.normal(0, step)
        
        new_pos = current_pos.copy()
        new_pos[i] = [current_pos[i][0] + dx, current_pos[i][1] + dy]
        
        new_ang = current_ang.copy()
        if np.random.random() < 0.1:
            new_ang[i] = current_ang[i] + np.random.normal(0, 10 * np.pi/180)
        
        S_new, _ = calc_outer_side(new_pos, new_ang)
        overlap_new = check_overlaps(new_pos)
        cost_new = S_new + 10.0 * overlap_new
        
        delta_cost = cost_new - cost_curr
        
        if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
            current_pos = new_pos
            current_ang = new_ang
            cost_curr = cost_new
            S_curr = S_new
            overlap_curr = overlap_new
            
            if overlap_curr == 0:
                if S_curr < best_valid_S:
                    best_valid_S = S_curr
                    best_valid_pos = current_pos.copy()
                    best_valid_ang = current_ang.copy()
        
        T *= cooling
    
    if best_valid_pos is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 50.0, 0.0
    
    S_final, centroid = calc_outer_side(best_valid_pos, best_valid_ang)
    
    inner_list = []
    for i in range(N):
        inner_list.append([best_valid_pos[i][0], best_valid_pos[i][1], np.rad2deg(best_valid_ang[i])])
        
    return inner_list, (centroid[0], centroid[1]), S_final, 0.0
# EVOLVE-BLOCK-END
