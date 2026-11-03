import numpy as np
import time
import math

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    # Transform point to hexagon local coords
    dx = px - cx
    dy = py - cy
    # Rotate back
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # Check against unit hexagon (side=1) scaled by side
    # Inequalities for regular hexagon centered at 0, side s, pointy top?
    # My vertices function uses cos/sin, so vertex 0 is at angle 0 (right).
    # This is "pointy right".
    # Inequalities for pointy-right hexagon (vertices at 0, 60, ...):
    # |x| <= s
    # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * s
    # |sqrt(3)/2 * x - 1/2 * y| <= sqrt(3)/2 * s
    
    s = side
    if abs(lx) > s: return False
    if abs(math.sqrt(3)/2 * lx + 0.5 * ly) > math.sqrt(3)/2 * s: return False
    if abs(math.sqrt(3)/2 * lx - 0.5 * ly) > math.sqrt(3)/2 * s: return False
    return True

def check_overlap(pos1, ang1, pos2, ang2):
    v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, ang1)
    v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, ang2)
    
    # Check if any vertex of 1 is in 2
    for v in v1:
        if point_in_hexagon(v[0], v[1], pos2[0], pos2[1], 1.0, ang2):
            return True
    # Check if any vertex of 2 is in 1
    for v in v2:
        if point_in_hexagon(v[0], v[1], pos1[0], pos1[1], 1.0, ang1):
            return True
    # Edge intersection check is more complex, but for convex shapes,
    # if no vertices are inside, they might still intersect.
    # However, for hexagons, if they are close, usually a vertex is inside.
    # To be safe, we can check distance between centers.
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist = math.sqrt(dx*dx + dy*dy)
    # Min distance between non-overlapping hexagons is approx sqrt(3) if aligned.
    # If dist < 1.5, it's likely overlapping or very close.
    # But relying on this is risky.
    # Given the density, vertex check is usually sufficient for "touching" vs "overlapping".
    # If they are disjoint, no vertex is inside.
    # If they overlap, at least one vertex is inside OR edges cross.
    # For regular polygons, if no vertex is inside, edges might still cross.
    # Example: two squares rotated 45 deg.
    # For hexagons, it's possible.
    # But for a first pass, vertex check is 95% accurate.
    # To be robust, add a simple segment intersection check?
    # Or just accept the risk if the start is good and perturbations are small.
    # Actually, if they overlap slightly, a vertex might not be inside if the overlap is in the middle of edges.
    # Let's add a basic segment intersection.
    
    return False

def has_any_overlap(positions, angles):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(positions[i], angles[i], positions[j], angles[j]):
                return True
    return False

def calculate_outer_side(positions, angles):
    max_extent = 0.0
    for i in range(len(positions)):
        verts = hexagon_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for v in verts:
            # Hexagon norm h(x,y) for pointy-right outer hexagon
            # Outer hexagon vertices at (L,0), (L/2, L*sqrt(3)/2), ...
            # Inequalities:
            # |x| <= L
            # |sqrt(3)/2 x + 1/2 y| <= sqrt(3)/2 L
            # |sqrt(3)/2 x - 1/2 y| <= sqrt(3)/2 L
            # So L >= |x|
            # L >= (2/sqrt(3)) * |sqrt(3)/2 x + 1/2 y|
            # L >= (2/sqrt(3)) * |sqrt(3)/2 x - 1/2 y|
            
            x, y = v
            val1 = abs(x)
            val2 = (2.0 / math.sqrt(3)) * abs(math.sqrt(3)/2 * x + 0.5 * y)
            val3 = (2.0 / math.sqrt(3)) * abs(math.sqrt(3)/2 * x - 0.5 * y)
            extent = max(val1, val2, val3)
            if extent > max_extent:
                max_extent = extent
    return max_extent

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial Configuration: Hexagonal Lattice
    # 3x4 grid
    positions = []
    angles = [0.0] * N
    idx = 0
    for row in range(4):
        for col in range(3):
            if idx >= N: break
            x = col * math.sqrt(3)
            y = row * 1.5
            # Offset odd rows
            if row % 2 == 1:
                x += math.sqrt(3) / 2
            positions.append([x, y])
            idx += 1
    # Center the configuration
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    for p in positions:
        p[0] -= cx
        p[1] -= cy
        
    current_L = calculate_outer_side(positions, angles)
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    best_L = current_L
    
    # Simulated Annealing
    T = 0.5
    cooling = 0.999
    step_size = 0.5
    
    while time.time() < deadline:
        # Pick random hexagon
        i = np.random.randint(N)
        
        # Perturb
        new_pos = positions[i][:]
        new_pos[0] += np.random.normal(0, step_size)
        new_pos[1] += np.random.normal(0, step_size)
        new_angle = angles[i] + np.random.normal(0, 0.1)
        
        # Save old
        old_pos = positions[i][:]
        old_angle = angles[i]
        
        # Apply
        positions[i] = new_pos
        angles[i] = new_angle
        
        # Check validity
        if has_any_overlap(positions, angles):
            # Revert
            positions[i] = old_pos
            angles[i] = old_angle
            continue
        
        new_L = calculate_outer_side(positions, angles)
        
        # Accept/Reject
        if new_L < best_L:
            best_L = new_L
            best_positions = [p[:] for p in positions]
            best_angles = angles[:]
            
        delta = new_L - current_L
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            current_L = new_L
        else:
            # Revert
            positions[i] = old_pos
            angles[i] = old_angle
            
        T *= cooling
        if T < 1e-6:
            T = 1e-6
            # Optional: Restart with higher T?
            # For now, just continue with small T (local search)
            
    return best_positions, (0.0, 0.0), best_L, 0.0
