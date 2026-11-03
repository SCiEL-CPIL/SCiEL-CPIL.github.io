import numpy as np
import time
import itertools

N = 11

def hex_vertices(x, y, angle_deg):
    """
    Generate vertices of a unit regular hexagon centered at (x, y)
    with rotation angle in degrees.
    Vertices are in counter-clockwise order.
    """
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    
    # Base vertices for unit hexagon (flat-top, side 1)
    # Angles: 0, 60, 120, 180, 240, 300
    base_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    base_x = np.cos(base_angles)
    base_y = np.sin(base_angles)
    
    # Rotate
    x_rot = c * base_x - s * base_y
    y_rot = s * base_x + c * base_y
    
    # Translate
    x_final = x_rot + x
    y_final = y_rot + y
    
    return np.column_stack((x_final, y_final))

def point_in_hexagon(px, py, cx, cy, angle_deg):
    """
    Check if point (px, py) is inside the hexagon centered at (cx, cy) with rotation.
    """
    verts = hex_vertices(cx, cy, angle_deg)
    # Use cross product method
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Vector from vertex i to vertex i+1
        ex = x2 - x1
        ey = y2 - y1
        # Vector from vertex i to point
        px_rel = px - x1
        py_rel = py - y1
        cross = ex * py_rel - ey * px_rel
        # For CCW polygon, point is inside if cross product is positive (or zero) for all edges
        # Assuming CCW winding
        if cross < -1e-9:
            return False
    return True

def check_packing(centers, angles, eps=1e-7):
    """
    Check if the given packing of hexagons has no overlaps.
    Returns True if valid (no overlaps).
    """
    for i in range(N):
        verts_i = hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        for j in range(i + 1, N):
            # Check if any vertex of i is inside j
            for v in verts_i:
                if point_in_hexagon(v[0], v[1], centers[j, 0], centers[j, 1], angles[j]):
                    return False
            # Check if any vertex of j is inside i
            verts_j = hex_vertices(centers[j, 0], centers[j, 1], angles[j])
            for v in verts_j:
                if point_in_hexagon(v[0], v[1], centers[i, 0], centers[i, 1], angles[i]):
                    return False
    return True

def calc_side(centers, angles):
    """
    Calculate the side length of the minimal regular hexagon
    (centered at origin, rotation 0) that contains all inner hexagons.
    """
    # Normals of the outer hexagon (flat-top)
    # Faces are perpendicular to vectors at 0, 60, 120, 180, 240, 300 degrees
    normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] 
        for a in [0, 60, 120, 180, 240, 300]
    ])
    
    max_apothem = 0.0
    for i in range(N):
        verts = hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        for v in verts:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_apothem:
                    max_apothem = proj
                    
    # Side S = 2 * apothem / sqrt(3)
    return 2.0 * max_apothem / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Initial configuration: Hexagonal lattice
    # Lattice vectors for flat-top hexagons
    d = np.sqrt(3)
    u = np.array([d, 0])
    v = np.array([d/2, 1.5])
    
    # Generate 11 positions
    # Center
    pos = [(0, 0)]
    # 6 neighbors
    pos.append(tuple(u))
    pos.append(tuple(v))
    pos.append(tuple(u + v))
    pos.append(tuple(-u))
    pos.append(tuple(-v))
    pos.append(tuple(-u - v))
    # 4 more from second ring
    pos.append(tuple(2*u))
    pos.append(tuple(2*v))
    pos.append(tuple(u + 2*v))
    pos.append(tuple(-2*u))
    
    centers = np.array(pos[:N])
    angles = np.zeros(N)
    
    best_side = calc_side(centers, angles)
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    # Local search
    # We want to minimize the side length.
    # We will use a simple stochastic local search with simulated annealing-like acceptance.
    
    current_centers = centers.copy()
    current_angles = angles.copy()
    current_side = best_side
    
    step_size = 0.1
    temp = 0.5
    
    for it in range(100000):
        if time.time() > deadline:
            break
            
        # Pick a random hexagon to move
        idx = np.random.randint(N)
        
        # Random perturbation
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        dangle = np.random.normal(0, step_size * 10)
        
        new_centers = current_centers.copy()
        new_angles = current_angles.copy()
        new_centers[idx, 0] += dx
        new_centers[idx, 1] += dy
        new_angles[idx] += dangle
        
        # Check validity (no overlaps)
        # To save time, we can first check a cheaper metric: distance between centers
        # Min distance between centers of non-overlapping hexagons is sqrt(3)
        # If any pair is closer, it's likely overlapping.
        valid = True
        for j in range(N):
            if j == idx:
                continue
            dist_sq = (new_centers[idx, 0] - new_centers[j, 0])**2 + (new_centers[idx, 1] - new_centers[j, 1])**2
            if dist_sq < (d - 1e-6)**2:
                # Likely overlap, but not certain if rotated.
                # For simplicity in search, reject if too close.
                # If we want to be more accurate, we'd do full check, but it's slow.
                # Let's do a full check only if it passes the distance check.
                valid = False
                break
        
        if valid:
            # Full overlap check
            if not check_packing(new_centers, new_angles):
                valid = False
        
        if valid:
            new_side = calc_side(new_centers, new_angles)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_side < current_side:
                current_centers = new_centers
                current_angles = new_angles
                current_side = new_side
                
                if current_side < best_side:
                    best_side = current_side
                    best_centers = current_centers.copy()
                    best_angles = current_angles.copy()
            else:
                # Simulated annealing acceptance
                delta = new_side - current_side
                if np.random.rand() < np.exp(-delta / temp):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_side = new_side
        
        # Decay temperature and step size
        temp *= 0.999
        step_size *= 0.9999
        if step_size < 0.001:
            step_size = 0.001

    # Final refinement
    # If we have time, do a few more iterations with smaller steps
    if time.time() < deadline - 5:
        step_size = 0.01
        temp = 0.1
        for it in range(50000):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            dangle = np.random.normal(0, step_size * 10)
            
            new_centers = current_centers.copy()
            new_angles = current_angles.copy()
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            new_angles[idx] += dangle
            
            valid = True
            for j in range(N):
                if j == idx:
                    continue
                dist_sq = (new_centers[idx, 0] - new_centers[j, 0])**2 + (new_centers[idx, 1] - new_centers[j, 1])**2
                if dist_sq < (d - 1e-6)**2:
                    valid = False
                    break
            
            if valid and check_packing(new_centers, new_angles):
                new_side = calc_side(new_centers, new_angles)
                if new_side < current_side:
                    current_centers = new_centers
                    current_angles = new_angles
                    current_side = new_side
                    if current_side < best_side:
                        best_side = current_side
                        best_centers = current_centers.copy()
                        best_angles = current_angles.copy()
            step_size *= 0.9999

    # Construct output
    inner = []
    for i in range(N):
        x, y = best_centers[i, 0], best_centers[i, 1]
        ang = best_angles[i]
        inner.append([float(x), float(y), float(ang)])
    
    # Outer hexagon center is (0,0), side is best_side, rotation is 0
    return inner, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    # Quick test
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
    print(f"Inner hexagons: {inner[:3]}...")
