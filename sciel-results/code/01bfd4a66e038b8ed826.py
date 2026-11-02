"""
Packing 12 unit hexagons in a regular hexagon - optimized approach.
Uses a hexagonal lattice structure with 3-fold symmetry, then local optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 145.0

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def sat_min_separation(v1, v2):
    """Compute minimum separation between two convex polygons using SAT."""
    axes = []
    for v in [v1, v2]:
        for i in range(len(v)):
            edge = v[(i + 1) % len(v)] - v[i]
            elen = np.linalg.norm(edge)
            if elen > 1e-12:
                nrm = np.array([-edge[1], edge[0]]) / elen
                axes.append(nrm)
    
    min_sep = float('inf')
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
    return min_sep

def required_outer_side(centers, angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    max_extent = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = hexagon_vertices(cx, cy, 1.0, angles[i])
        for v in verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            # For a regular hexagon with apothem a, the constraint is:
            # max over 6 directions of |dot(rel, n_k)| <= a
            for k in range(6):
                theta = np.radians(60 * k)
                d = abs(rx * np.cos(theta) + ry * np.sin(theta))
                if d > max_extent:
                    max_extent = d
    
    return max_extent / (np.sqrt(3) / 2)

def check_all_valid(centers, angles, outer_side, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if packing is valid: no overlaps and all inside outer hexagon."""
    # Check outer containment
    actual_side = required_outer_side(centers, angles, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    # Check pairwise separations
    verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts_list.append(hexagon_vertices(cx, cy, 1.0, angles[i]))
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep = sat_min_separation(verts_list[i], verts_list[j])
            if sep < -1e-9:
                return False
    return True

def build_hexagonal_lattice(r1, r2, angle1=0.0, angle2=0.0):
    """
    Build a hexagonal lattice packing with 3-fold symmetry.
    Ring 1: 6 hexagons at radius r1, angle angle1
    Ring 2: 6 hexagons at radius r2, angle angle2
    All hexagons have the same orientation (0 degrees for flat-top).
    """
    centers = []
    angles = []
    
    # Ring 1: 6 hexagons at 60-degree intervals
    for i in range(6):
        theta = angle1 + np.radians(60 * i)
        x = r1 * np.cos(theta)
        y = r1 * np.sin(theta)
        centers.append((x, y))
        angles.append(0.0)
    
    # Ring 2: 6 hexagons at 60-degree intervals, offset by 30 degrees
    for i in range(6):
        theta = angle2 + np.radians(60 * i)
        x = r2 * np.cos(theta)
        y = r2 * np.sin(theta)
        centers.append((x, y))
        angles.append(0.0)
    
    return centers, angles

def objective_lattice(params):
    """
    Objective for lattice optimization.
    params = [r1, r2, angle1, angle2]
    Returns the required outer side length (to minimize).
    """
    r1, r2, angle1, angle2 = params
    
    if r1 < 0 or r2 < 0:
        return 1e6
    
    centers, angles = build_hexagonal_lattice(r1, r2, angle1, angle2)
    
    # Check for overlaps
    verts_list = []
    for i in range(N):
        cx, cy = centers[i]
        verts_list.append(hexagon_vertices(cx, cy, 1.0, angles[i]))
    
    min_sep = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_min_separation(verts_list[i], verts_list[j])
            if sep < min_sep:
                min_sep = sep
    
    if min_sep < -1e-6:
        # Penalize overlaps heavily
        return 1e6 + abs(min_sep) * 1000
    
    side = required_outer_side(centers, angles)
    return side

def optimize_lattice(r1_init, r2_init, angle1_init=0.0, angle2_init=0.0, time_limit=10.0):
    """Optimize the lattice parameters using L-BFGS-B."""
    start_time = time.time()
    
    x0 = np.array([r1_init, r2_init, angle1_init, angle2_init])
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        return objective_lattice(p)
    
    result = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10})
    
    return result.x

def optimize_full(centers_init, angles_init, outer_side, time_limit=15.0):
    """
    Full optimization of all positions and angles.
    params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    """
    start_time = time.time()
    
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers_init[i][0]
        params[3*i+1] = centers_init[i][1]
        params[3*i+2] = angles_init[i]
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[3*i], p[3*i+1]))
            angles.append(p[3*i+2])
        
        # Check overlaps
        verts_list = []
        for i in range(N):
            cx, cy = centers[i]
            verts_list.append(hexagon_vertices(cx, cy, 1.0, angles[i]))
        
        min_sep = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                sep = sat_min_separation(verts_list[i], verts_list[j])
                if sep < min_sep:
                    min_sep = sep
        
        if min_sep < -1e-6:
            return 1e6 + abs(min_sep) * 1000
        
        side = required_outer_side(centers, angles)
        return side
    
    result = minimize(obj, params, method='Nelder-Mead',
                      options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10})
    
    centers = []
    angles = []
    for i in range(N):
        centers.append((result.x[3*i], result.x[3*i+1]))
        angles.append(result.x[3*i+2])
    
    return centers, angles

def local_search(centers, angles, outer_side, n_iter=500, time_limit=10.0):
    """
    Local search / simulated annealing on the full configuration.
    """
    start_time = time.time()
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_side = required_outer_side(centers, angles)
    
    current_centers = [list(c) for c in centers]
    current_angles = list(angles)
    current_side = required_outer_side(current_centers, current_angles)
    
    T = 0.5
    T_end = 0.01
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random hexagon to move
        i = np.random.randint(N)
        new_centers = [list(c) for c in current_centers]
        new_angles = list(current_angles)
        
        step = T * 0.2
        new_centers[i][0] += np.random.normal(0, step)
        new_centers[i][1] += np.random.normal(0, step)
        new_angles[i] = (new_angles[i] + np.random.normal(0, step * 30)) % 60
        
        # Check validity
        verts_list = []
        for j in range(N):
            cx, cy = new_centers[j]
            verts_list.append(hexagon_vertices(cx, cy, 1.0, new_angles[j]))
        
        min_sep = float('inf')
        for j in range(N):
            for k in range(j + 1, N):
                sep = sat_min_separation(verts_list[j], verts_list[k])
                if sep < min_sep:
                    min_sep = sep
        
        if min_sep < -1e-6:
            # Reject if overlapping
            T = max(T_end, T * 0.99)
            continue
        
        new_side = required_outer_side(new_centers, new_angles)
        
        delta = new_side - current_side
        if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-10)):
            current_centers = new_centers
            current_angles = new_angles
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in current_centers]
                best_angles = list(current_angles)
        
        T = max(T_end, T * 0.99)
    
    return best_centers, best_angles, best_side

def construct():
    start_time = time.time()
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Hexagonal lattice with 3-fold symmetry
    # Initial guess: ring 1 at r~2, ring 2 at r~4
    initial_params = [
        (2.0, 4.0, 0.0, 30.0),
        (2.1, 4.2, 0.0, 30.0),
        (1.9, 3.8, 0.0, 30.0),
        (2.0, 4.0, 0.0, 0.0),
        (2.2, 4.4, 0.0, 30.0),
    ]
    
    for r1, r2, a1, a2 in initial_params:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
        
        # Optimize lattice parameters
        try:
            opt_params = optimize_lattice(r1, r2, a1, a2, time_limit=8.0)
            centers, angles = build_hexagonal_lattice(opt_params[0], opt_params[1], 
                                                       opt_params[2], opt_params[3])
        except Exception:
            centers, angles = build_hexagonal_lattice(r1, r2, a1, a2)
        
        # Check validity
        verts_list = []
        for i in range(N):
            cx, cy = centers[i]
            verts_list.append(hexagon_vertices(cx, cy, 1.0, angles[i]))
        
        min_sep = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                sep = sat_min_separation(verts_list[i], verts_list[j])
                if sep < min_sep:
                    min_sep = sep
        
        if min_sep < -1e-6:
            continue
        
        side = required_outer_side(centers, angles)
        
        # Local search to improve
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 15:
            try:
                lc, la, ls = local_search(centers, angles, side, n_iter=300, time_limit=10.0)
                if ls < side:
                    centers, angles, side = lc, la, ls
            except Exception:
                pass
        
        if side < best_score:
            best_score = side
            best_result = (centers, angles, side)
    
    # Strategy 2: Full optimization from lattice starting points
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10 and best_result is not None:
        centers, angles, side = best_result
        try:
            oc, oa = optimize_full(centers, angles, side, time_limit=15.0)
            # Verify
            verts_list = []
            for i in range(N):
                cx, cy = oc[i]
                verts_list.append(hexagon_vertices(cx, cy, 1.0, oa[i]))
            
            min_sep = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    sep = sat_min_separation(verts_list[i], verts_list[j])
                    if sep < min_sep:
                        min_sep = sep
            
            if min_sep >= -1e-6:
                new_side = required_outer_side(oc, oa)
                if new_side < best_score:
                    best_score = new_side
                    best_result = (oc, oa, new_side)
        except Exception:
            pass
    
    if best_result is not None:
        centers, angles, side = best_result
        # Format output: inner rows (x, y, angle_degrees)
        inner = []
        for i in range(N):
            inner.append([centers[i][0], centers[i][1], angles[i]])
        return inner, (0.0, 0.0), side, 0.0
    
    # Fallback: simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(
        [(p[0], p[1]) for p in inner_default], 
        [p[2] for p in inner_default]
    )
    return inner_default, (0.0, 0.0), required_side, 0.0
