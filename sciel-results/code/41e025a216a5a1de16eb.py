"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

N = 12
DEADLINE = time.time() + 140  # safety margin before 150s kill
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    verts_x = cx + side * np.cos(angles)
    verts_y = cy + side * np.sin(angles)
    return np.column_stack((verts_x, verts_y))

def point_in_convex_polygon(pt, poly):
    """Check if point is inside or on boundary of convex polygon."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross < -1e-10:
            return False
    return True

def polygons_overlap_sat(p1, p2):
    """Check if two convex polygons overlap using SAT."""
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        ax = p1[i]
        bx = p1[(i + 1) % n1]
        ex = bx - ax
        nx, ny = -ex[1], ex[0]
        proj1 = p1 @ np.array([nx, ny])
        proj2 = p2 @ np.array([nx, ny])
        if proj1.max() < proj2.min() - 1e-10 or proj2.max() < proj1.min() - 1e-10:
            return False
    for i in range(n2):
        ax = p2[i]
        bx = p2[(i + 1) % n2]
        ex = bx - ax
        nx, ny = -ex[1], ex[0]
        proj1 = p1 @ np.array([nx, ny])
        proj2 = p2 @ np.array([nx, ny])
        if proj1.max() < proj2.min() - 1e-10 or proj2.max() < proj1.min() - 1e-10:
            return False
    return True

def min_translation_to_separate(p1, p2):
    """Find minimum translation vector to separate p1 from p2 using SAT."""
    n1, n2 = len(p1), len(p2)
    min_overlap = float('inf')
    best_normal = None
    
    axes = []
    for i in range(n1):
        ax = p1[i]
        bx = p1[(i + 1) % n1]
        ex = bx - ax
        axes.append(np.array([-ex[1], ex[0]]))
    for i in range(n2):
        ax = p2[i]
        bx = p2[(i + 1) % n2]
        ex = bx - ax
        axes.append(np.array([-ex[1], ex[0]]))
    
    for axis in axes:
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            continue
        axis = axis / axis_norm
        proj1 = p1 @ axis
        proj2 = p2 @ axis
        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        if overlap < -1e-10:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
            best_normal = axis
    
    if best_normal is None:
        return None
    
    c1 = p1.mean(axis=0)
    c2 = p2.mean(axis=0)
    if np.dot(c1 - c2, best_normal) < 0:
        best_normal = -best_normal
    
    return best_normal * min_overlap

def compute_min_outer_side(inner_polys, max_iter=50):
    """Binary search for minimum outer hex side (centered at origin, angle 0)."""
    max_r = 0.0
    for poly in inner_polys:
        for v in poly:
            r = np.linalg.norm(v)
            if r > max_r:
                max_r = r
    lo = max(1.0, max_r * 0.9)
    hi = max_r * 1.5 + 2.0
    
    for _ in range(max_iter):
        if hi - lo < 1e-7:
            break
        mid = (lo + hi) / 2
        outer_verts = hex_vertices(0, 0, 0, mid)
        ok = True
        for poly in inner_polys:
            for v in poly:
                if not point_in_convex_polygon(v, outer_verts):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def validate_config(centers, angles, outer_side):
    """Check if configuration is valid."""
    inner_polys = [hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_overlap_sat(inner_polys[i], inner_polys[j]):
                return False, inner_polys
    outer_verts = hex_vertices(0, 0, 0, outer_side)
    for poly in inner_polys:
        for v in poly:
            if not point_in_convex_polygon(v, outer_verts):
                return False, inner_polys
    return True, inner_polys

def pbd_simulate(centers, angles, outer_side, n_iterations=150, time_limit=None):
    """PBD simulation to resolve overlaps and keep inside boundary."""
    start_time = time.time()
    centers = centers.copy()
    angles = angles.copy()
    
    for it in range(n_iterations):
        if time_limit is not None and time.time() > time_limit:
            break
        
        inner_polys = [hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_overlap_sat(inner_polys[i], inner_polys[j]):
                    trans = min_translation_to_separate(inner_polys[i], inner_polys[j])
                    if trans is not None:
                        centers[i, 0] += trans[0] * 0.5
                        centers[i, 1] += trans[1] * 0.5
                        centers[j, 0] -= trans[0] * 0.5
                        centers[j, 1] -= trans[1] * 0.5
                        inner_polys[i] = hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
                        inner_polys[j] = hex_vertices(centers[j, 0], centers[j, 1], angles[j], 1.0)
        
        outer_verts = hex_vertices(0, 0, 0, outer_side)
        for i in range(N):
            for v in inner_polys[i]:
                if not point_in_convex_polygon(v, outer_verts):
                    dist_from_origin = np.linalg.norm(centers[i])
                    if dist_from_origin > 0.01:
                        centers[i, 0] -= centers[i, 0] * 0.3
                        centers[i, 1] -= centers[i, 1] * 0.3
                    break
            inner_polys[i] = hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0)
    
    return centers, angles

def hexagonal_lattice_init():
    """Initialize with hexagonal lattice pattern."""
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    spacing = 2.1
    
    positions = [
        (0, 0), (spacing, 0), (2*spacing, 0),
        (spacing/2, spacing*SQRT3), (1.5*spacing, spacing*SQRT3), (2.5*spacing, spacing*SQRT3), (3.5*spacing, spacing*SQRT3),
        (0, 2*spacing*SQRT3), (spacing, 2*spacing*SQRT3), (2*spacing, 2*spacing*SQRT3), (3*spacing, 2*spacing*SQRT3),
        (spacing/2, 3*spacing*SQRT3)
    ]
    
    for i in range(N):
        centers[i, 0] = positions[i][0]
        centers[i, 1] = positions[i][1]
    
    center = centers.mean(axis=0)
    centers -= center
    return centers, angles

def construct():
    start_time = time.time()
    best_side = 1e10
    best_centers = None
    best_angles = None
    best_valid = False
    
    # Strategy 1: PBD simulation from hexagonal lattice init
    try:
        centers, angles = hexagonal_lattice_init()
        inner_polys = [hex_vertices(centers[i, 0], centers[i, 1], angles[i], 1.0) for i in range(N)]
        max_r = max(np.linalg.norm(v) for poly in inner_polys for v in poly)
        current_side = max_r + 0.5
        
        for shrink_iter in range(8):
            if time.time() > DEADLINE - 20:
                break
            centers, angles = pbd_simulate(centers, angles, current_side, n_iterations=120, time_limit=DEADLINE - 15)
            valid, inner_polys = validate_config(centers, angles, current_side)
            if valid:
                new_side = compute_min_outer_side(inner_polys)
                if new_side < best_side:
                    best_side = new_side
                    best_centers = centers.copy()
                    best_angles = angles.copy()
                    best_valid = True
                current_side = max(new_side + 0.05, best_side * 0.96)
            else:
                current_side *= 0.97
        
        if best_valid and time.time() < DEADLINE - 10:
            def obj(p):
                c = p[:2*N].reshape(N, 2)
                a = p[2*N:]
                valid, _ = validate_config(c, a, best_side)
                if not valid:
                    return best_side + 10.0
                return compute_min_outer_side([hex_vertices(c[i,0], c[i,1], a[i], 1.0) for i in range(N)])
            
            x0 = np.concatenate([best_centers.flatten(), best_angles])
            local_result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if local_result.fun < best_side:
                new_c = local_result.x[:2*N].reshape(N, 2)
                new_a = local_result.x[2*N:]
                valid2, _ = validate_config(new_c, new_a, local_result.fun)
                if valid2:
                    best_side = local_result.fun
                    best_centers = new_c
                    best_angles = new_a
    except Exception:
        pass
    
    # Strategy 2: Differential evolution fallback
    if (not best_valid or best_side > 5.0) and time.time() < DEADLINE - 30:
        try:
            def de_obj(params):
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                inner_polys = [hex_vertices(c[i,0], c[i,1], a[i], 1.0) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        if polygons_overlap_sat(inner_polys[i], inner_polys[j]):
                            return 1e10
                return compute_min_outer_side(inner_polys)
            
            bounds = [(-4, 4)] * 2*N + [(0, 360)] * N
            result = differential_evolution(de_obj, bounds, maxiter=100, tol=0.01, popsize=12, seed=42, polish=False, workers=1)
            if result.fun < best_side:
                best_side = result.fun
                best_centers = result.x[:2*N].reshape(N, 2)
                best_angles = result.x[2*N:]
                best_valid = True
        except Exception:
            pass
    
    # Fallback
    if not best_valid or best_side >= 1e9:
        best_centers = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for i in range(N):
            best_centers[i, 0] = (i - N/2 + 0.5) * 2.5
            best_centers[i, 1] = 0
        best_side = compute_min_outer_side([hex_vertices(best_centers[i,0], best_centers[i,1], 0, 1.0) for i in range(N)])
        best_valid = True
    
    valid, _ = validate_config(best_centers, best_angles, best_side)
    if not valid:
        best_centers = np.zeros((N, 2))
        best_angles = np.zeros(N)
        for i in range(N):
            best_centers[i, 0] = (i - N/2 + 0.5) * 3.0
            best_centers[i, 1] = 0
        best_side = compute_min_outer_side([hex_vertices(best_centers[i,0], best_centers[i,1], 0, 1.0) for i in range(N)])
    
    inner = [[best_centers[i,0], best_centers[i,1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
