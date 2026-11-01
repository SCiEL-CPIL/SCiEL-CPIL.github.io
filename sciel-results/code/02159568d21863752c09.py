"""Improved packing of 11 unit hexagons in a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def get_hex_vertices_local(angle_rad):
    """Return 6 vertices of a unit hexagon centered at origin with given angle."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = math.cos(theta)
        vy = math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Outward normals for a hexagon with vertices at 0, 60, 120, 180, 240, 300 degrees
    # are at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_projection = 0.0
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    all_projections = []
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                all_projections.append(proj)
    
    if not all_projections:
        return 0.0
    
    max_proj = max(all_projections)
    if max_proj < -1000:
        return 0.0
    
    exp_sum = 0.0
    for p in all_projections:
        exp_sum += math.exp(beta * (p - max_proj))
    soft_max = max_proj + math.log(exp_sum) / beta
    
    return 2.0 * soft_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap.
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
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
    verts_list = []
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        verts_list.append(vi)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on center distances.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    dists = np.sqrt(dists_sq_upper)
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def build_hex_cluster(n_hex):
    """
    Build a compact hexagonal cluster using axial coordinates.
    For n=11, use center + 6 neighbors + 4 from second ring.
    """
    if n_hex == 11:
        # Center + 6 ring + 4 from second ring for compactness
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-1, 2), (-2, 1)
        ]
    else:
        # Generic: fill rings
        axial_coords = [(0, 0)]
        ring = 1
        while len(axial_coords) < n_hex:
            for q in range(-ring, ring + 1):
                for r in range(-ring, ring + 1):
                    if max(abs(q), abs(r), abs(-q-r)) == ring:
                        axial_coords.append((q, r))
                        if len(axial_coords) >= n_hex:
                            break
                if len(axial_coords) >= n_hex:
                    break
            ring += 1
    
    return axial_coords[:n_hex]

def axial_to_cartesian(q, r, spacing=1.0):
    """Convert axial coordinates to Cartesian for a hex grid."""
    x = 1.5 * spacing * q
    y = SQRT3 * spacing * (r + q / 2.0)
    return x, y

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Build initial hexagonal cluster
    axial_coords = build_hex_cluster(N)
    
    # Use tight spacing to encourage compact packing
    spacing = 1.0 + 1e-7
    
    centers = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r, spacing)
        centers.append((x, y))
    
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner[:]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Compute centroid for initial outer center
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # 2. Optimize outer hexagon parameters (center + angle) with fixed inner
    def objective_outer(params):
        cx, cy, angle_deg = params
        return compute_required_side_fast(initial_inner, (cx, cy), angle_deg)
    
    best_val = float('inf')
    best_result = None
    
    # Try multiple initial angles
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 50:
            break
        
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        best_outer_center = (float(best_result[0]), float(best_result[1]))
        best_outer_angle = float(best_result[2])
        best_side = best_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(initial_inner, best_outer_center, best_outer_angle)
    
    # 3. Joint optimization: allow inner hexagons to move and rotate
    # Use a smooth surrogate for the outer constraint
    
    def objective_joint(params, beta=100.0):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 500.0 * penalty
    
    # Initial params
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params, dtype=np.float64)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Simulated annealing style local search
    np.random.seed(42)
    step_size = 0.1
    min_step = 0.001
    accept_prob = 0.1
    
    for iteration in range(5000):
        if time.time() > deadline - 30:
            break
        
        num_perturb = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15])
        indices = np.random.choice(len(current_params), num_perturb, replace=False)
        
        neighbor = current_params.copy()
        for idx in indices:
            if idx % 3 == 2:  # angle
                delta = np.random.normal(0, step_size * 0.3)
            else:  # position
                delta = np.random.normal(0, step_size)
            neighbor[idx] += delta
        
        neighbor_val = objective_joint(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        else:
            # Accept worse solutions with some probability
            delta_e = neighbor_val - current_val
            if delta_e < 0.1 and np.random.random() < accept_prob * math.exp(-delta_e / (0.01 + step_size)):
                current_params = neighbor
                current_val = neighbor_val
        
        # Decay step size
        step_size = max(min_step, step_size * 0.997)
        accept_prob = max(0.01, accept_prob * 0.999)
    
    # Reconstruct best joint solution
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    # Verify disjointness
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    else:
        # Try to repair: nudge overlapping hexagons apart
        repaired = best_inner_joint[:]
        for _ in range(50):
            if check_all_disjoint(repaired, eps=1e-5):
                break
            # Find an overlapping pair and separate them
            verts = [get_hex_vertices(r[i][0], r[i][1], 1.0, math.radians(r[i][2])) for i in range(N)]
            found = False
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap_check(verts[i], verts[j], 1e-5):
                        # Push apart
                        dx = repaired[j][0] - repaired[i][0]
                        dy = repaired[j][1] - repaired[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        push = 0.05
                        repaired[i][0] -= push * dx / dist
                        repaired[i][1] -= push * dy / dist
                        repaired[j][0] += push * dx / dist
                        repaired[j][1] += push * dy / dist
                        found = True
                        break
                if found:
                    break
            # Recompute vertices
            verts = [get_hex_vertices(r[i][0], r[i][1], 1.0, math.radians(r[i][2])) for i in range(N)]
        
        if check_all_disjoint(repaired, eps=1e-5):
            best_inner = repaired
            best_outer_center = (best_ocx, best_ocy)
            best_outer_angle = best_oangle
            best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        else:
            # Fallback to rigid body result
            best_inner = initial_inner[:]
            best_side = best_val
    
    # 4. Final polish: optimize outer container tightly around the fixed inner layout
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side_fast(best_inner, (cx, cy), angle_deg)
    
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 5],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 5],
        [best_outer_center[0] + 0.05, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.05, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.05, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.05, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 20000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    best_outer_center = (float(best_polish_params[0]), float(best_polish_params[1]))
    best_outer_angle = float(best_polish_params[2])
    best_side = best_polish_val
    
    # Final verification
    if not check_all_disjoint(best_inner, eps=1e-6):
        # Fallback to the initial lattice configuration
        best_inner = initial_inner[:]
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
