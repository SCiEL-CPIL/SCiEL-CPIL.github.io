"""Rigid-body cluster optimization for hexagon packing."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon at (cx, cy)."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    c1, c2: (x, y, angle_deg)
    Returns (min_sep, best_axis). Positive if separated, negative if overlapping.
    """
    v1 = hexagon_vertices(c1[0], c1[1], 1.0, c1[2])
    v2 = hexagon_vertices(c2[0], c2[1], 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    best_axis = None
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
    
    return min_sep, best_axis

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    outer_normals = np.column_stack([np.cos(np.radians(60 * np.arange(6))), 
                                      np.sin(np.radians(60 * np.arange(6)))])
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices(px, py, 1.0, pa)
        for v in inner_verts:
            rx = v[0] - outer_center[0]
            ry = v[1] - outer_center[1]
            rx_local = rx * cos_a - ry * sin_a
            ry_local = rx * sin_a + ry * cos_a
            d = np.abs(rx_local * outer_normals[:, 0] + ry_local * outer_normals[:, 1])
            max_extent = max(max_extent, np.max(d))
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-9:
                return False
    return True

def generate_hex_lattice_config(scale, rotation_deg, outer_angle=0.0):
    """
    Generate a hexagonal lattice configuration of 12 unit hexagons.
    scale: scaling factor for lattice spacing
    rotation_deg: rotation of the lattice
    Returns list of (x, y, angle_deg) for each hexagon.
    """
    # Hexagonal lattice: 1 center + 6 around + 5 more
    # Standard hex packing: center at origin, 6 neighbors, then outer ring
    base_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2),
        (0.5, -np.sqrt(3)/2), (-0.5, -np.sqrt(3)/2),
        (1.5, np.sqrt(3)/2), (-1.5, np.sqrt(3)/2),
        (1.5, -np.sqrt(3)/2), (-1.5, -np.sqrt(3)/2),
        (0, np.sqrt(3)), (0, -np.sqrt(3))
    ]
    
    rot_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    
    centers = []
    for (bx, by) in base_positions:
        x = scale * (bx * cos_r - by * sin_r)
        y = scale * (bx * sin_r + by * cos_r)
        centers.append([x, y, rotation_deg])
    
    return centers

def compute_packing_score(centers, outer_center, outer_angle):
    """Compute score: outer side length + penalty for overlaps."""
    side = required_outer_side(centers, outer_center, outer_angle)
    
    overlap_pen = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < 0:
                overlap_pen += (-sep) ** 2
    
    return side + 10.0 * overlap_pen

def optimize_lattice_params(scale, rotation_deg, outer_angle, time_limit=20.0):
    """
    Optimize lattice parameters using Nelder-Mead.
    Returns (best_centers, best_outer_center, best_side, best_outer_angle).
    """
    start_time = time.time()
    
    def objective(params):
        s, rot, oa = params
        centers = generate_hex_lattice_config(s, rot, oa)
        return compute_packing_score(centers, (0.0, 0.0), oa)
    
    x0 = np.array([scale, rotation_deg, outer_angle])
    
    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False}
    )
    
    if time.time() - start_time > time_limit:
        return None
    
    best_params = result.x
    best_centers = generate_hex_lattice_config(best_params[0], best_params[1], best_params[2])
    best_side = required_outer_side(best_centers, (0.0, 0.0), best_params[2])
    
    # Verify validity
    valid = True
    for i in range(len(best_centers)):
        for j in range(i + 1, len(best_centers)):
            sep, _ = sat_min_separation(best_centers[i], best_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return (best_centers, (0.0, 0.0), best_side, best_params[2])

def differential_evolution_search(time_limit=30.0):
    """
    Use differential evolution to search over lattice parameters.
    """
    start_time = time.time()
    
    def objective(params):
        s, rot, oa = params
        centers = generate_hex_lattice_config(s, rot, oa)
        return compute_packing_score(centers, (0.0, 0.0), oa)
    
    bounds = [(1.8, 3.5), (0, 60), (0, 60)]
    
    result = differential_evolution(
        objective, bounds,
        maxiter=50, popsize=15, seed=42,
        tol=1e-6, polish=True
    )
    
    if time.time() - start_time > time_limit:
        return None
    
    best_params = result.x
    best_centers = generate_hex_lattice_config(best_params[0], best_params[1], best_params[2])
    best_side = required_outer_side(best_centers, (0.0, 0.0), best_params[2])
    
    valid = True
    for i in range(len(best_centers)):
        for j in range(i + 1, len(best_centers)):
            sep, _ = sat_min_separation(best_centers[i], best_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return (best_centers, (0.0, 0.0), best_side, best_params[2])

def individual_optimization(centers, outer_center, outer_angle, time_limit=20.0):
    """
    Individually optimize each hexagon's position and angle using Nelder-Mead.
    """
    start_time = time.time()
    n = len(centers)
    
    # First, fix all except one and optimize each in turn
    for target_idx in range(n):
        if time.time() - start_time > time_limit:
            break
        
        def objective(params):
            new_centers = [list(c) for c in centers]
            new_centers[target_idx] = [params[0], params[1], params[2] % 60]
            return compute_packing_score(new_centers, outer_center, outer_angle)
        
        x0 = np.array(centers[target_idx])
        
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7, 'disp': False}
        )
        
        centers[target_idx] = [result.x[0], result.x[1], result.x[2] % 60]
    
    # Optimize outer angle
    if time.time() - start_time < time_limit - 5:
        def outer_angle_obj(oa):
            return required_outer_side(centers, outer_center, oa)
        
        result = minimize(
            outer_angle_obj, outer_angle,
            method='Nelder-Mead',
            options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-7, 'disp': False}
        )
        outer_angle = result.x[0]
    
    side = required_outer_side(centers, outer_center, outer_angle)
    
    # Verify validity
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return (centers, outer_center, side, outer_angle)

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Lattice-based search with differential evolution
    try:
        result = differential_evolution_search(time_limit=min(30, DEADLINE - 30))
        if result is not None:
            centers, outer_center, side, outer_angle = result
            if check_validity(centers, outer_center, side, outer_angle):
                if side < best_score:
                    best_score = side
                    best_result = (centers, outer_center, side, outer_angle)
    except Exception:
        pass
    
    # Phase 2: Individual optimization on best result
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < DEADLINE - 25:
        try:
            centers, outer_center, side, outer_angle = best_result
            result = individual_optimization(centers, outer_center, outer_angle, 
                                            time_limit=min(25, DEADLINE - elapsed - 10))
            if result is not None:
                c, oc, s, oa = result
                if check_validity(c, oc, s, oa) and s < best_score:
                    best_score = s
                    best_result = (c, oc, s, oa)
        except Exception:
            pass
    
    # Phase 3: Try different lattice rotations and scales
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        for rot in [0, 30, 15, 45]:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
            try:
                result = optimize_lattice_params(2.5, rot, 0, time_limit=min(10, DEADLINE - elapsed - 5))
                if result is not None:
                    centers, outer_center, side, outer_angle = result
                    if check_validity(centers, outer_center, side, outer_angle):
                        if side < best_score:
                            best_score = side
                            best_result = (centers, outer_center, side, outer_angle)
            except Exception:
                pass
    
    # Fallback: return best found or a valid default
    if best_result is not None:
        return best_result
    
    # Default valid packing
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
# EVOLVE-BLOCK-END
