"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _is_valid_aligned(cx_arr, cy_arr):
    """
    Check if a set of aligned (angle=0) unit hexagons are disjoint.
    For aligned hexagons, the minimum distance to avoid overlap is sqrt(3) (approx 1.732).
    However, this is the minimum distance for *touching*. 
    If dist < sqrt(3), they overlap.
    """
    min_dist_sq = (math.sqrt(3) - 1e-7) ** 2
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                return False
    return True

def _min_outer_side(centers, angles):
    """
    Finds the minimum side length of an outer regular hexagon
    that contains all inner hexagons defined by centers and angles.
    Uses grid search for outer angle and center.
    Optimized for speed.
    """
    # Precompute vertices for all inner hexagons
    all_verts = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        all_verts.append(verts)
    
    best_side = float('inf')
    best_center = (0,0)
    best_angle = 0.0
    
    # Symmetry of hexagon: period 60 degrees.
    # Search range 0 to 60.
    n_angles = 30
    for i in range(n_angles):
        outer_angle = i * (60.0 / n_angles)
        
        # Normals for this outer angle
        a_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            na = a_rad + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        # Calculate centroid of all vertices for center initialization
        all_v_flat = [v for verts in all_verts for v in verts]
        cx_mean = sum(v[0] for v in all_v_flat) / len(all_v_flat)
        cy_mean = sum(v[1] for v in all_v_flat) / len(all_v_flat)
        
        # Grid search for center around centroid
        # Range: +/- 1.5
        # Steps: 15x15
        best_local_side = float('inf')
        best_local_cx = cx_mean
        best_local_cy = cy_mean
        
        for dcx in range(-15, 16):
            for dcy in range(-15, 16):
                cx = cx_mean + dcx * 0.1
                cy = cy_mean + dcy * 0.1
                
                max_proj = 0.0
                for verts in all_verts:
                    for (vx, vy) in verts:
                        for (nx, ny) in normals:
                            proj = nx * (vx - cx) + ny * (vy - cy)
                            if proj > max_proj:
                                max_proj = proj
                # Side length S
                side = (2.0 * max_proj) / math.sqrt(3)
                if side < best_local_side:
                    best_local_side = side
                    best_local_cx = cx
                    best_local_cy = cy
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_center = (best_local_cx, best_local_cy)
            best_angle = outer_angle
            
    return best_side, best_center, best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_triangular_lattice():
        points = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = i * math.sqrt(3) + (j % 2) * math.sqrt(3) / 2
                y = j * 1.5
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        centers = [[p[0], p[1], 0.0] for p in points[:12]]
        return centers

    def make_rectangular_grid():
        centers = []
        dx = 2.0
        dy = math.sqrt(3)
        for r in range(3):
            for c in range(4):
                x = c * dx - 1.5 * dx
                y = r * dy - dy
                centers.append([x, y, 0.0])
        return centers

    def make_compact_cluster():
        centers = [[0.0, 0.0, 0.0]]
        r = math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            centers.append([x, y, 0.0])
        r2 = r * 2
        for deg in [30, 90, 150, 210, 270]:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            centers.append([x, y, 0.0])
        return centers

    initializers = [make_triangular_lattice, make_rectangular_grid, make_compact_cluster]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0,0)
    best_angle = 0.0

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        cfg = init_func()
        # Ensure validity
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        if _is_valid_aligned(cx_arr, cy_arr):
            centers = np.array([p[:2] for p in cfg])
            angles = np.array([p[2] for p in cfg])
            side, center, oangle = _min_outer_side(centers, angles)
            if side < best_side:
                best_side = side
                best_cfg = cfg
                best_center = center
                best_angle = oangle

    if best_cfg is None:
        cfg = make_triangular_lattice()
        best_cfg = cfg
        centers = np.array([p[:2] for p in cfg])
        angles = np.array([p[2] for p in cfg])
        best_side, best_center, best_angle = _min_outer_side(centers, angles)

    current_cfg = [row[:] for row in best_cfg]
    current_side = best_side
    current_center = best_center
    current_angle = best_angle
    
    T = 0.1
    alpha = 0.95
    
    while time.time() - t0 < TIME_LIMIT - 10:
        improved = False
        # Limit iterations per outer loop to prevent getting stuck
        for _ in range(50):
            if time.time() - t0 > TIME_LIMIT - 10:
                break
            
            idx = random.randrange(N)
            old_x, old_y, old_a = current_cfg[idx]
            
            current_cfg[idx][0] += random.gauss(0, T)
            current_cfg[idx][1] += random.gauss(0, T)
            
            cx_arr = np.array([p[0] for p in current_cfg])
            cy_arr = np.array([p[1] for p in current_cfg])
            
            if _is_valid_aligned(cx_arr, cy_arr):
                centers = np.array([p[:2] for p in current_cfg])
                angles = np.array([p[2] for p in current_cfg])
                new_side, new_center, new_angle = _min_outer_side(centers, angles)
                
                if new_side < current_side:
                    current_side = new_side
                    current_center = new_center
                    current_angle = new_angle
                    improved = True
                else:
                    current_cfg[idx][0] = old_x
                    current_cfg[idx][1] = old_y
                    current_cfg[idx][2] = old_a
            else:
                current_cfg[idx][0] = old_x
                current_cfg[idx][1] = old_y
                current_cfg[idx][2] = old_a
        
        if not improved:
            T *= alpha
            if T < 0.001:
                break

    if current_side < best_side:
        best_side = current_side
        best_cfg = current_cfg
        best_center = current_center
        best_angle = current_angle

    return best_cfg, best_center, best_side, best_angle
