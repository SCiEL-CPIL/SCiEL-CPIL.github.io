"""
Packing 11 unit hexagons inside a regular hexagon.
Strategy: Use known compact topologies, optimize with a fast vectorized smooth
objective, then polish with Nelder-Mead. Focus on the hexagonal cluster topology
which is near-optimal.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute unit hexagon vertices for various angles
def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def get_hex_vertices_array(cx, cy, side, angle_rad):
    """Return 6x2 array of vertices."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    # Precompute normal directions for the outer hexagon
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Fast vectorized computation of required side length.
    inner: list of [x, y, angle_deg]
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    # Collect all vertices
    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices_array(ix, iy, 1.0, math.radians(iangle))
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
    
    all_verts = np.vstack(all_verts)  # (6*N, 2)
    
    # Transform to outer frame
    dx = all_verts[:, 0] - ox
    dy = all_verts[:, 1] - oy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a

    # Precompute normals
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)

    # Compute all projections: (6*N, 6)
    proj_lx = lx[:, np.newaxis] * normals[0, :]
    proj_ly = ly[:, np.newaxis] * normals[1, :]
    projections = proj_lx + proj_ly  # (6*N, 6)

    max_projection = np.max(projections)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temp=0.01):
    """Smooth approximation using log-sum-exp."""
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices_array(ix, iy, 1.0, math.radians(iangle))
        all_verts.append(verts)
    
    if not all_verts:
        return 0.0
    
    all_verts = np.vstack(all_verts)
    
    dx = all_verts[:, 0] - ox
    dy = all_verts[:, 1] - oy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a

    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    proj_lx = lx[:, np.newaxis] * normals[0, :]
    proj_ly = ly[:, np.newaxis] * normals[1, :]
    projections = proj_lx + proj_ly  # (6*N, 6)

    # Log-sum-exp over all projections
    amax = np.max(projections)
    if not np.isfinite(amax):
        return 0.0
    smooth_max = amax + temp * math.log(np.sum(np.exp((projections - amax) / temp)))
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True

def axial_to_xy(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian for hex grid."""
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y

def get_candidate_topologies():
    """
    Generate candidate discrete topologies for 11 hexagons.
    Focus on compact hexagonal clusters.
    """
    topologies = []
    
    # The standard hexagonal cluster: center + 6 neighbors + 4 more
    # This is a radius-2 hexagon minus 2 corners
    base_cluster = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    topologies.append(base_cluster)
    
    # Variant 1: different corner removal
    variant1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (0, -2)
    ]
    topologies.append(variant1)
    
    # Variant 2: another corner configuration
    variant2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ]
    topologies.append(variant2)
    
    # Variant 3: more spread
    variant3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (2, -1)  # This has 12, fix
    ]
    # Remove one to get 11
    variant3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    topologies.append(variant3)
    
    # Variant 4: asymmetric
    variant4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (2, -1)
    ]
    topologies.append(variant4)
    
    # Variant 5: linear-ish
    variant5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 0)
    ]
    topologies.append(variant5)
    
    # Variant 6: two rows with offset
    variant6 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    topologies.append(variant6)
    
    return topologies

def optimize_topology(topology, time_budget=20.0):
    """
    Optimize a given discrete topology by adjusting positions, rotations, and container.
    Returns (inner, center, side, angle) or None if invalid.
    """
    start = time.time()
    n = len(topology)
    if n != N:
        return None
    
    # Initial positions from axial coordinates with slight expansion
    scale = 1.0 + 1e-5
    initial_positions = []
    for q, r in topology:
        x, y = axial_to_xy(q, r, scale)
        initial_positions.append([x, y, 0.0])
    
    # Compute centroid for initial outer center
    cx = sum(p[0] for p in initial_positions) / n
    cy = sum(p[1] for p in initial_positions) / n
    
    # Parameters: 3*n for inner (x, y, angle) + 3 for outer (cx, cy, angle)
    params = []
    for x, y, a in initial_positions:
        params.extend([x, y, a])
    params.extend([cx, cy, 0.0])
    
    def objective(p):
        inner = []
        for i in range(n):
            inner.append([p[i*3], p[i*3+1], p[i*3+2]])
        ocx = p[n*3]
        ocy = p[n*3+1]
        oang = p[n*3+2]
        
        side = compute_required_side_smooth(inner, (ocx, ocy), oang, temp=0.01)
        
        # Overlap penalty using center distances
        penalty = 0.0
        min_dist = SQRT3 - 0.02
        for i in range(n):
            for j in range(i+1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        return side + 500.0 * penalty
    
    # Simulated Annealing
    T0 = 0.3
    T_end = 0.0005
    total_steps = 12000
    
    np.random.seed(hash(tuple(topology)) % 10000)
    
    current = params[:]
    current_obj = objective(current)
    best_params = current[:]
    best_obj = current_obj
    best_valid = False
    best_inner = None
    best_center = None
    best_angle = None
    best_side = float('inf')
    
    # Check initial validity
    inner_init = []
    for i in range(n):
        inner_init.append([current[i*3], current[i*3+1], current[i*3+2]])
    if check_all_disjoint(inner_init):
        side_init = compute_required_side(inner_init, (current[n*3], current[n*3+1]), current[n*3+2])
        if check_inside_outer(inner_init, (current[n*3], current[n*3+1]), side_init, current[n*3+2]):
            best_valid = True
            best_inner = inner_init
            best_center = (current[n*3], current[n*3+1])
            best_angle = current[n*3+2]
            best_side = side_init
            best_obj = side_init
    
    for step in range(total_steps):
        if time.time() - start > time_budget:
            break
        
        T = T0 * (T_end / T0) ** (step / total_steps)
        new_p = current[:]
        move_type = np.random.randint(0, 4)
        
        if move_type < 2:
            idx = np.random.randint(0, n)
            base = idx * 3
            new_p[base] += np.random.uniform(-0.08, 0.08)
            new_p[base+1] += np.random.uniform(-0.08, 0.08)
        elif move_type == 2:
            idx = np.random.randint(0, n)
            new_p[idx*3+2] += np.random.uniform(-8, 8)
        else:
            if np.random.random() < 0.7:
                new_p[n*3] += np.random.uniform(-0.08, 0.08)
                new_p[n*3+1] += np.random.uniform(-0.08, 0.08)
            else:
                new_p[n*3+2] += np.random.uniform(-8, 8)
        
        new_obj = objective(new_p)
        
        if new_obj < current_obj or np.random.random() < math.exp(-(new_obj - current_obj) / max(T, 1e-10)):
            current = new_p
            current_obj = new_obj
            
            if step % 300 == 0:
                inner_l = []
                for i in range(n):
                    inner_l.append([current[i*3], current[i*3+1], current[i*3+2]])
                if check_all_disjoint(inner_l):
                    side_l = compute_required_side(inner_l, (current[n*3], current[n*3+1]), current[n*3+2])
                    if check_inside_outer(inner_l, (current[n*3], current[n*3+1]), side_l, current[n*3+2]):
                        if side_l < best_side:
                            best_valid = True
                            best_inner = inner_l
                            best_center = (current[n*3], current[n*3+1])
                            best_angle = current[n*3+2]
                            best_side = side_l
                            best_obj = side_l
                            best_params = current[:]
    
    # Nelder-Mead polish
    if time.time() - start < time_budget - 2:
        try:
            def polish_obj(p):
                inner_l = []
                for i in range(n):
                    inner_l.append([p[i*3], p[i*3+1], p[i*3+2]])
                side = compute_required_side_smooth(inner_l, (p[n*3], p[n*3+1]), p[n*3+2], temp=0.005)
                pen = 0.0
                min_d = SQRT3 - 0.01
                for i in range(n):
                    for j in range(i+1, n):
                        d = math.sqrt((inner_l[i][0]-inner_l[j][0])**2 + (inner_l[i][1]-inner_l[j][1])**2)
                        if d < min_d:
                            pen += (min_d - d)**2
                return side + 1000.0 * pen
            
            res = minimize(polish_obj, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000})
            
            inner_l = []
            for i in range(n):
                inner_l.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            cx_p = res.x[n*3]
            cy_p = res.x[n*3+1]
            ang_p = res.x[n*3+2]
            
            if check_all_disjoint(inner_l):
                side_p = compute_required_side(inner_l, (cx_p, cy_p), ang_p)
                if check_inside_outer(inner_l, (cx_p, cy_p), side_p, ang_p):
                    if side_p < best_side:
                        best_inner = inner_l
                        best_center = (cx_p, cy_p)
                        best_angle = ang_p
                        best_side = side_p
        except Exception:
            pass
    
    if best_valid:
        return best_inner, best_center, best_side, best_angle
    return None

def construct():
    start_time = time.time()
    
    # Generate candidate topologies
    topologies = get_candidate_topologies()
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sort topologies by compactness heuristic
    def topology_compactness(topo):
        xs = [t[0] for t in topo]
        ys = [t[1] for t in topo]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))
    
    topologies.sort(key=topology_compactness)
    
    # Optimize each topology with time budget
    time_per_topology = max(10.0, (DEADLINE - 25 - (time.time() - start_time)) / max(1, len(topologies)))
    
    for topo in topologies:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        topo_list = [tuple(t) for t in topo]
        
        result = optimize_topology(topo_list, time_budget=time_per_topology)
        
        if result is not None:
            inner, center, side, angle = result
            if side < best_side:
                best_side = side
                best_inner = inner
                best_center = center
                best_angle = angle
    
    # Fallback if no valid solution found
    if best_inner is None:
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x, y = axial_to_xy(q, r, scale_fallback)
            fallback_centers.append((x, y))
        
        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
    
    # Final verification
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_center, best_side, best_angle):
        # Force a valid fallback
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        scale_fallback = 1.1
        fallback_centers = []
        for q, r in axial_coords:
            x, y = axial_to_xy(q, r, scale_fallback)
            fallback_centers.append((x, y))
        
        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
    
    return best_inner, best_center, best_side, best_angle
