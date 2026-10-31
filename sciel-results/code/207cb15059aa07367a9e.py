"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array."""
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    outer_center: (ox, oy)
    outer_angle_deg: rotation of outer hexagon
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
    
    max_projection = 0.0
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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
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
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
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
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Smooth penalty for overlaps based on center distances."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    return penalty

def generate_polyhexes_bfs(n_cells, max_cells=11):
    """
    Generate all polyhexes with n_cells cells using BFS on axial grid.
    Returns list of sets of axial coordinates.
    """
    # Axial directions
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Start with single cell
    polyhexes = [frozenset([(0, 0)])]
    
    for size in range(1, max_cells):
        new_polyhexes = set()
        for ph in polyhexes:
            for cell in ph:
                for d in dirs:
                    new_cell = (cell[0] + d[0], cell[1] + d[1])
                    if new_cell not in ph:
                        new_ph = frozenset(ph | {new_cell})
                        # Canonical form: translate so min cell is (0,0)
                        min_q = min(c[0] for c in new_ph)
                        min_r = min(c[1] for c in new_ph)
                        canonical = frozenset([(c[0] - min_q, c[1] - min_r) for c in new_ph])
                        new_polyhexes.add(canonical)
        if not new_polyhexes:
            break
        polyhexes = list(new_polyhexes)
    
    return polyhexes

def axial_to_cartesian(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian."""
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return x, y

def polyhex_to_inner_list(polyhex, scale=1.0):
    """Convert a polyhex (set of axial coords) to list of (x, y, angle) for inner hexagons."""
    inner = []
    for q, r in polyhex:
        x, y = axial_to_cartesian(q, r, scale)
        inner.append([x, y, 0.0])
    return inner

def construct():
    """Main entry point: find optimal packing."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Try polyhex-based approach (discrete combinatorial search)
    # Generate all 11-cell polyhexes and evaluate each
    try:
        if time.time() < deadline - 60:
            polyhexes = generate_polyhexes_bfs(N, max_cells=N)
            if polyhexes:
                # For each polyhex, try different scales and outer orientations
                for polyhex in polyhexes:
                    if time.time() > deadline - 30:
                        break
                    
                    # Try a few scales
                    for scale in [1.0, 1.01, 1.02]:
                        inner = polyhex_to_inner_list(polyhex, scale)
                        
                        # Compute centroid
                        cx_avg = sum(c[0] for c in inner) / len(inner)
                        cy_avg = sum(c[1] for c in inner) / len(inner)
                        
                        # Try several outer angles
                        for outer_angle in [0, 30, 60, 90, 120, 150]:
                            S = compute_required_side(inner, (cx_avg, cy_avg), outer_angle)
                            if S < best_score:
                                # Verify validity
                                if check_all_disjoint(inner, eps=1e-6):
                                    best_score = S
                                    best_result = (inner, (cx_avg, cy_avg), S, outer_angle)
    except Exception:
        pass
    
    # Strategy 2: Continuous optimization with smooth objective
    # Initialize from hexagonal lattice
    hex_positions = [(0, 0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = SQRT3 * np.cos(angle)
        y = SQRT3 * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * SQRT3 * np.cos(angle)
        y = 2 * SQRT3 * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    hex_positions = hex_positions[:N]
    
    initial_inner = [[px, py, 0.0] for px, py in hex_positions]
    cx_avg = sum(c[0] for c in initial_inner) / len(initial_inner)
    cy_avg = sum(c[1] for c in initial_inner) / len(initial_inner)
    
    # Phase 1: Optimize outer hexagon with smooth surrogate
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    best_outer = None
    best_outer_val = float('inf')
    
    for init_angle in init_angles:
        if time.time() > deadline - 40:
            break
        
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # Smooth pass for basin finding
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                  options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 800})
            
            # Exact refinement from smooth result
            res_exact = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead',
                                 options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1200})
            
            if res_exact.fun < best_outer_val:
                best_outer_val = res_exact.fun
                best_outer = res_exact.x
        except Exception:
            pass
    
    if best_outer is not None:
        outer_center = (best_outer[0], best_outer[1])
        outer_angle = best_outer[2]
        S = compute_required_side(initial_inner, outer_center, outer_angle)
        if check_all_disjoint(initial_inner, eps=1e-6):
            if S < best_score:
                best_score = S
                best_result = (initial_inner, outer_center, S, outer_angle)
    
    # Phase 2: Joint optimization with smooth objective
    if time.time() < deadline - 20:
        def objective_joint_smooth(params, beta=60.0):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append((x, y, a))
            ox = params[3*N]
            oy = params[3*N+1]
            outer_angle = params[3*N+2]
            S = compute_required_side_smooth(inner_list, (ox, oy), outer_angle, beta=beta)
            penalty = compute_overlap_penalty(inner_list)
            return S + penalty * 100
        
        def objective_joint_exact(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append((x, y, a))
            ox = params[3*N]
            oy = params[3*N+1]
            outer_angle = params[3*N+2]
            S = compute_required_side(inner_list, (ox, oy), outer_angle)
            penalty = compute_overlap_penalty(inner_list)
            return S + penalty * 100
        
        # Initialize from best result or initial
        if best_result is not None:
            inner_init = best_result[0]
            ox_init = best_result[1][0]
            oy_init = best_result[1][1]
            oa_init = best_result[3]
        else:
            inner_init = initial_inner
            ox_init = cx_avg
            oy_init = cy_avg
            oa_init = 0.0
        
        params_init = np.zeros(3*N + 3)
        for i, (x, y, a) in enumerate(inner_init):
            params_init[i*3] = x
            params_init[i*3+1] = y
            params_init[i*3+2] = a
        params_init[3*N] = ox_init
        params_init[3*N+1] = oy_init
        params_init[3*N+2] = oa_init
        
        # Try multiple random restarts
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                p_init = params_init.copy()
            else:
                p_init = params_init + np.random.randn(3*N + 3) * 0.1
            
            try:
                # Smooth phase
                res_smooth = minimize(objective_joint_smooth, p_init, method='L-BFGS-B',
                                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                
                # Exact phase
                res_exact = minimize(objective_joint_exact, res_smooth.x, method='L-BFGS-B',
                                     options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res_exact.fun < best_score:
                    params = res_exact.x
                    inner = []
                    for i in range(N):
                        cx = params[i*3]
                        cy = params[i*3+1]
                        ang = params[i*3+2]
                        inner.append([cx, cy, ang])
                    ox = params[3*N]
                    oy = params[3*N+1]
                    outer_angle = params[3*N+2]
                    S = compute_required_side(inner, (ox, oy), outer_angle)
                    if check_all_disjoint(inner, eps=1e-6):
                        if S < best_score:
                            best_score = S
                            best_result = (inner, (ox, oy), S, outer_angle)
            except Exception:
                pass
    
    # Fallback if nothing valid found
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in hex_positions]
        ox, oy = cx_avg, cy_avg
        outer_angle = 0.0
        S = compute_required_side(inner, (ox, oy), outer_angle)
        S += 0.1
        best_result = (inner, (ox, oy), S, outer_angle)
    
    return best_result

# EVOLVE-BLOCK-END
