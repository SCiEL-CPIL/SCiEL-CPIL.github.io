import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _get_normals(verts):
    """Compute outward normals for a convex polygon."""
    edges = np.roll(verts, -1, axis=0) - verts
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    normals = normals / norms
    # Ensure normals point outward
    center = np.mean(verts, axis=0, keepdims=True)
    mids = (verts + np.roll(verts, -1, axis=0)) / 2.0
    dots = np.sum(normals * (center - mids), axis=1, keepdims=True)
    normals = np.where(dots < 0, -normals, normals)
    return normals

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons v1, v2.
    Returns 0 if no overlap, positive value if overlapping.
    """
    n1 = _get_normals(v1)
    n2 = _get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    # Project vertices onto all normals
    proj1 = v1 @ all_normals.T
    proj2 = v2 @ all_normals.T
    
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)
    min2 = proj2.min(axis=0)
    max2 = proj2.max(axis=0)
    
    # Calculate overlap in each normal direction
    overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
    
    # If any direction has no overlap, polygons are separated
    if np.any(overlaps <= 1e-9):
        return 0.0
        
    return np.min(overlaps)

def _min_enclosing_hexagon_side(positions, angles, outer_angle_deg, outer_center):
    """
    Compute the minimum side length of a regular hexagon with the given outer angle and center
    that contains all the inner hexagons.
    """
    outer_angle_rad = np.radians(outer_angle_deg)
    # Normals to the sides of the outer hexagon
    normal_angles = outer_angle_rad + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    cos_normals = np.cos(normal_angles)
    sin_normals = np.sin(normal_angles)
    
    max_proj = 0.0
    oc = np.array(outer_center)
    
    for i in range(N):
        v = _hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
        v_rel = v - oc
        projs = v_rel[:, 0] * cos_normals + v_rel[:, 1] * sin_normals
        col_max = np.max(projs, axis=0)
        global_max = np.max(col_max)
        if global_max > max_proj:
            max_proj = global_max
            
    # Side length s = 2 * apothem / sqrt(3)
    side = max_proj * 2.0 / np.sqrt(3.0)
    return side

def _objective(params):
    """
    Objective function: Minimize outer side length + overlap penalty.
    """
    positions = [(params[i*3], params[i*3+1]) for i in range(N)]
    angles = [params[i*3+2] for i in range(N)]
    outer_angle = params[N*3]
    outer_cx = params[N*3+1]
    outer_cy = params[N*3+2]
    
    side = _min_enclosing_hexagon_side(positions, angles, outer_angle, (outer_cx, outer_cy))
    
    penalty = 0.0
    # Check overlaps
    for i in range(N):
        v1 = _hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
        for j in range(i+1, N):
            v2 = _hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
            # Quick check: distance between centers
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist_sq = dx*dx + dy*dy
            # Max distance for touching hexagons is sqrt(3) (center to center)
            # If dist_sq > (2*sqrt(3))^2, they definitely don't overlap
            # Actually, max radius is 1. So if dist > 2, no overlap.
            # But hexagons are elongated. Max distance between centers for ANY overlap is 2*sqrt(3) approx.
            # Let's use a safe bound: 2.5
            if dist_sq < 6.25: 
                depth = _sat_overlap_depth(v1, v2)
                if depth > 0:
                    penalty += depth * depth * 500.0
                    
    return side + penalty

def _try_pack(initial_params, time_budget):
    """
    Run optimizer on a specific initial configuration.
    """
    start_time = time.time()
    
    def _obj_with_timeout(params):
        if time.time() - start_time > time_budget:
            return 1e10
        return _objective(params)

    try:
        res = minimize(
            _obj_with_timeout,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
        )
        
        if time.time() - start_time > time_budget:
            return None, float('inf')
            
        # Check validity
        positions = [(res.x[i*3], res.x[i*3+1]) for i in range(N)]
        angles = [res.x[i*3+2] for i in range(N)]
        outer_angle = res.x[N*3]
        outer_cx = res.x[N*3+1]
        outer_cy = res.x[N*3+2]
        
        has_overlap = False
        for i in range(N):
            v1 = _hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            for j in range(i+1, N):
                v2 = _hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
                depth = _sat_overlap_depth(v1, v2)
                if depth > 1e-5:
                    has_overlap = True
                    break
            if has_overlap:
                break
                
        if has_overlap:
            return None, float('inf')
            
        side = _min_enclosing_hexagon_side(positions, angles, outer_angle, (outer_cx, outer_cy))
        return res.x, side
        
    except Exception:
        return None, float('inf')

def _get_lattice_init(rows, cols, offset_x=0.0, offset_y=0.0):
    """Generate a hexagonal lattice configuration."""
    params = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2.0
            y = r * 1.5
            # Center the configuration
            x -= (cols - 1) * np.sqrt(3) / 2.0
            y -= (rows - 1) * 1.5 / 2.0
            params.extend([x + offset_x, y + offset_y, 0.0])
            count += 1
            
    while len(params) < N*3:
        params.extend([np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0.0])
        
    params.extend([0.0, 0.0, 0.0]) # Outer angle, cx, cy
    return np.array(params[:N*3+3])

def _get_random_init():
    params = []
    for _ in range(N):
        params.extend([np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(0, 60)])
    params.extend([np.random.uniform(0, 60), 0.0, 0.0])
    return np.array(params)

def construct():
    start_time = time.time()
    best_params = None
    best_side = float('inf')
    
    # Initial configurations
    inits = [
        _get_lattice_init(3, 4),
        _get_lattice_init(4, 3),
        _get_lattice_init(2, 6),
        _get_lattice_init(6, 2),
        _get_random_init(),
        _get_random_init()
    ]
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        params, side = _try_pack(init, time_budget=10.0)
        
        if params is not None and side < best_side:
            best_side = side
            best_params = params
            
            # If we find a very good solution, we can stop or refine
            if best_side < 4.0:
                break
                
    # If no valid packing found, return a fallback
    if best_params is None:
        # Fallback: Large hexagon
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.1 + (r % 2) * 1.05
                y = r * 1.8
                x -= 3 * 2.1 / 2
                y -= 2 * 1.8 / 2
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 8.0, 0.0
        
    # Extract final result
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    outer_angle = best_params[N*3]
    outer_cx = best_params[N*3+1]
    outer_cy = best_params[N*3+2]
    
    # Recompute side for safety
    positions = [(inner[i][0], inner[i][1]) for i in range(N)]
    angles = [inner[i][2] for i in range(N)]
    final_side = _min_enclosing_hexagon_side(positions, angles, outer_angle, (outer_cx, outer_cy))
    
    return inner, (outer_cx, outer_cy), final_side, outer_angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
