import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    """Generate vertices of a unit hexagon centered at (x,y) with given rotation."""
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def point_in_hexagon(p, center, angle_deg):
    """Check if point p is inside a unit hexagon centered at 'center' with rotation 'angle_deg'."""
    p_local = p - center
    angle_rad = -np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    p_local = p_local @ R.T
    
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    projs = p_local @ normals.T
    apothem = np.sqrt(3)/2
    return np.all(projs <= apothem + 1e-9)

def calc_side_and_penalty(params):
    """Calculate outer hexagon side length and overlap penalty."""
    normals = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    max_apothem = 0.0
    for k in range(N):
        x, y, ang = params[k*3], params[k*3+1], params[k*3+2]
        verts = hex_vertices(x, y, ang)
        for v in verts:
            for n in normals:
                proj = np.dot(v, n)
                if proj > max_apothem:
                    max_apothem = proj
        
    side = 2.0 * max_apothem / np.sqrt(3)
    
    penalty = 0.0
    for i in range(N):
        x_i, y_i, ang_i = params[i*3], params[i*3+1], params[i*3+2]
        verts_i = hex_vertices(x_i, y_i, ang_i)
        for j in range(i+1, N):
            x_j, y_j, ang_j = params[j*3], params[j*3+1], params[j*3+2]
            verts_j = hex_vertices(x_j, y_j, ang_j)
            
            for v in verts_i:
                if point_in_hexagon(v, np.array([x_j, y_j]), ang_j):
                    penalty += 10.0
            for v in verts_j:
                if point_in_hexagon(v, np.array([x_i, y_i]), ang_i):
                    penalty += 10.0
    return side, penalty

def objective(params):
    side, penalty = calc_side_and_penalty(params)
    return side + penalty

def construct():
    start_time = time.time()
    deadline = 145.0

    d = np.sqrt(3)
    sqrt3 = np.sqrt(3)
    
    # Define the optimal hexagonal lattice positions for 11 unit hexagons
    # Flower pattern: 1 center + 6 ring + 4 outer
    # Using exact coordinates for a tight hexagonal packing
    lattice_positions = np.array([
        [0.0, 0.0],           # center
        [d, 0.0],             # right
        [d/2, 1.5],           # upper-right
        [-d/2, 1.5],          # upper-left
        [-d, 0.0],            # left
        [-d/2, -1.5],         # lower-left
        [d/2, -1.5],          # lower-right
        [0.0, 3.0],           # top
        [2.6, 1.5],           # far upper-right
        [2.6, -1.5],          # far lower-right
        [0.0, -3.0],          # bottom
    ])
    
    # Precompute all vertices of all 11 hexagons (66 points total)
    # in the local frame (no rotation applied yet)
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # All 66 vertices in local frame
    all_local_verts = np.array([
        lattice_positions[i] + base_verts[j] 
        for i in range(N) 
        for j in range(6)
    ])
    
    # Outer hexagon normals (unit hexagon, rotation 0)
    outer_normals = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    def outer_side_for_rotation(angle_rad):
        """Compute the minimum outer hexagon side length for a given global rotation."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        rotated_verts = all_local_verts @ R.T
        
        # Project onto all 6 outer normals
        projections = rotated_verts @ outer_normals.T  # shape (66, 6)
        
        # For each normal direction, find the max projection
        # The outer hexagon must contain all points, so for each facet normal,
        # the apothem must be at least the max projection in that direction
        max_projs = np.max(projections, axis=0)  # shape (6,)
        
        # The outer hexagon's apothem must be at least max(max_projs)
        # But we also need to consider the negative direction (symmetric)
        # Actually, for a centered hexagon, we need |proj| <= apothem for all
        # So the required apothem is max(|proj|) over all points and normals
        abs_max_projs = np.max(np.abs(projections), axis=0)
        required_apothem = np.max(abs_max_projs)
        
        # Side length = 2 * apothem / sqrt(3)
        return 2.0 * required_apothem / sqrt3
    
    # Perform golden-section search over rotation angle [0, pi/3)
    # The problem has 6-fold symmetry, so we only need [0, pi/3)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    inv_phi = 1 / phi
    
    a, b = 0.0, np.pi/3
    
    # Initial golden-section points
    c_point = b - inv_phi * (b - a)
    d_point = a + inv_phi * (b - a)
    
    fc = outer_side_for_rotation(c_point)
    fd = outer_side_for_rotation(d_point)
    
    best_angle = c_point if fc < fd else d_point
    best_side = min(fc, fd)
    
    # Golden-section iterations
    max_iter = 200
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        if fc < fd:
            b = d_point
            d_point = c_point
            fd = fc
            c_point = b - inv_phi * (b - a)
            fc = outer_side_for_rotation(c_point)
        else:
            a = c_point
            c_point = d_point
            fc = fd
            d_point = a + inv_phi * (b - a)
            fd = outer_side_for_rotation(d_point)
        
        current_best = min(fc, fd)
        if current_best < best_side:
            best_side = current_best
            best_angle = c_point if fc < fd else d_point
    
    # High-precision polish with scipy minimize_scalar
    if time.time() < deadline - 10:
        try:
            res = scipy.optimize.minimize_scalar(
                outer_side_for_rotation,
                bounds=(0, np.pi/3),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 1000}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_angle = res.x
        except Exception:
            pass
        
        # Even tighter polish around the best point
        if time.time() < deadline - 5:
            try:
                delta = 0.01
                res2 = scipy.optimize.minimize_scalar(
                    outer_side_for_rotation,
                    bounds=(max(0, best_angle - delta), min(np.pi/3, best_angle + delta)),
                    method='bounded',
                    options={'xatol': 1e-14, 'maxiter': 2000}
                )
                if res2.success and res2.fun < best_side:
                    best_side = res2.fun
                    best_angle = res2.x
            except Exception:
                pass
    
    # Construct output: rotate all hexagon centers by best_angle
    c, s = np.cos(best_angle), np.sin(best_angle)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for i in range(N):
        center = lattice_positions[i] @ R.T
        inner.append([float(center[0]), float(center[1]), float(np.deg2rad(best_angle) * 180 / np.pi)])
    
    # Verify no overlaps (sanity check)
    penalty = 0.0
    for i in range(N):
        x_i, y_i, ang_i = inner[i][0], inner[i][1], inner[i][2]
        verts_i = hex_vertices(x_i, y_i, ang_i)
        for j in range(i+1, N):
            x_j, y_j, ang_j = inner[j][0], inner[j][1], inner[j][2]
            verts_j = hex_vertices(x_j, y_j, ang_j)
            for v in verts_i:
                if point_in_hexagon(v, np.array([x_j, y_j]), ang_j):
                    penalty += 10.0
            for v in verts_j:
                if point_in_hexagon(v, np.array([x_i, y_i]), ang_i):
                    penalty += 10.0
    
    if penalty > 1e-6:
        # Fallback to widely spaced config if overlaps detected
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (0.0, 0.0), float(best_side), 0.0
