import numpy as np
import time
import math
from scipy.optimize import linprog, minimize

N = 11
DEADLINE = 140.0
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    normals = _hex_edge_normals(outer_angle_deg)
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
                            for i in range(len(centers))])
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    c_obj = np.array([0.0, 0.0, 1.0])
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
    cx, cy, r = res.x
    side = 2 * r / SQRT3
    return side, np.array([cx, cy])

def _fast_optimize_outer(centers, inner_angles):
    """Coarse sweep + golden-section refinement for each candidate angle."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # Coarse sweep: 24 angles
    coarse_angles = np.linspace(0, 60, 25)
    for outer_angle in coarse_angles:
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # Fine refinement: golden section around best coarse angle
    # The side length as a function of angle is piecewise smooth; try nearby angles
    if best_side < float('inf'):
        # Try 12 fine angles around the best
        fine_angles = best_angle + np.linspace(-2.0, 2.0, 13)
        for outer_angle in fine_angles:
            if outer_angle < 0 or outer_angle > 60:
                continue
            side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _try_arrangement(centers, inner_angles=None):
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    side, center, angle = _fast_optimize_outer(centers, inner_angles)
    
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _get_valid_fallback():
    centers = []
    angles = [0.0] * N
    x = 0
    y = 0
    for i in range(N):
        centers.append((x, y))
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0
    side, center, angle = _fast_optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _smooth_max(values, beta=50.0):
    """Soft max using log-sum-exp."""
    m = np.max(values)
    if not np.isfinite(m):
        return m
    return m + math.log(np.sum(np.exp(beta * (values - m)))) / beta

def _polish_slsqp(centers, inner_angles, max_time=15.0):
    """
    High-precision polish using SLSQP with smooth epigraph formulation.
    Variables: x = [cx_1, cy_1, ..., cx_N, cy_N, a_1, ..., a_N, ocx, ocy, side]
    Objective: minimize side
    Constraints: 
      - For each outer normal direction i: max over all inner vertices of projection <= inradius
        Approximated smoothly: soft_max of projections - inradius <= 0
      - Non-overlap: for each pair, minimum separation >= 0 (approximated by distance constraint)
    """
    start_time = time.time()
    centers_arr = np.array(centers, dtype=float)
    angles_arr = np.array(inner_angles, dtype=float)
    
    # Initial outer
    side0, center0, angle0 = _fast_optimize_outer(centers, inner_angles)
    if not np.isfinite(side0):
        return centers, inner_angles, float('inf'), None, None, None
    
    # Build initial x
    x0 = np.concatenate([centers_arr.flatten(), angles_arr, [center0[0], center0[1], side0]])
    n_vars = 2 * N + N + 3
    
    def objective(x):
        if time.time() - start_time > max_time:
            return float('inf')
        return x[-1]  # side
    
    def constraints(x):
        if time.time() - start_time > max_time:
            return []
        
        c = x[:2*N].reshape(N, 2)
        a = x[2*N:3*N]
        ocx, ocy, side = x[3*N], x[3*N+1], x[3*N+2]
        
        # Outer angle: use angle0 (fixed for this polish)
        outer_normals = _hex_edge_normals(angle0)
        inradius = side * SQRT3 / 2
        
        cons = []
        
        # Containment constraints (smoothed)
        for i in range(6):
            nx, ny = outer_normals[i]
            # Compute max projection of all inner hex vertices in direction (nx, ny)
            # For each inner hex, the max projection of its vertices
            max_proj = -1e10
            for k in range(N):
                verts = _hex_vertices(c[k, 0], c[k, 1], 1.0, a[k])
                projs = (verts[:, 0] - ocx) * nx + (verts[:, 1] - ocy) * ny
                max_proj = max(max_proj, projs.max())
            # Constraint: max_proj - inradius <= 0
            cons.append(inradius - max_proj)
        
        # Non-overlap constraints (smooth approximation using center distance)
        # For hexagons of side 1, minimum center distance depends on relative angle
        # Use a conservative bound: distance >= 2*inradius_inner = sqrt(3)
        # But this is too conservative; use actual SAT-based check as a soft constraint
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[j, 0] - c[i, 0]
                dy = c[j, 1] - c[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance for non-overlapping unit hexagons is sqrt(3) when aligned
                # Use a slightly relaxed constraint
                min_dist = SQRT3 * 0.99  # small relaxation
                cons.append(dist - min_dist)
        
        return cons
    
    # Convert to scipy constraint format
    def make_constraint(idx):
        def func(x):
            cs = constraints(x)
            if idx < len(cs):
                return cs[idx]
            return 0.0
        return func
    
    n_constraints = 6 + N*(N-1)//2
    constraint_list = [{'type': 'ineq', 'fun': make_constraint(i)} for i in range(n_constraints)]
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-20, 20))  # cx
        bounds.append((-20, 20))  # cy
    for i in range(N):
        bounds.append((-180, 180))  # angle
    bounds.append((-20, 20))  # ocx
    bounds.append((-20, 20))  # ocy
    bounds.append((1.0, 50.0))  # side
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraint_list,
                       options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        
        if res.success and np.isfinite(res.fun):
            c = res.x[:2*N].reshape(N, 2)
            a = res.x[2*N:3*N]
            ocx, ocy, side = res.x[3*N], res.x[3*N+1], res.x[3*N+2]
            
            # Verify
            centers_list = [list(c[i]) for i in range(N)]
            angles_list = list(a)
            if _check_validity(centers_list, angles_list, np.array([ocx, ocy]), side, angle0):
                return centers_list, angles_list, side, np.array([ocx, ocy]), angle0
    except Exception:
        pass
    
    return centers, inner_angles, float('inf'), None, None, None

def _generate_initial_configurations():
    configs = []
    
    # Hexagonal lattice
    d1 = 2 * SQRT3
    d2 = 4 * SQRT3
    first_ring = [(d1 * math.cos(math.pi/3*i), d1 * math.sin(math.pi/3*i)) for i in range(6)]
    second_ring = []
    for i in range(6):
        a = math.pi/3*i
        second_ring.append((d2 * math.cos(a), d2 * math.sin(a)))
        a2 = math.pi/3*i + math.pi/6
        second_ring.append((d2 * math.cos(a2), d2 * math.sin(a2)))
    
    subsets = [
        [0, 2, 4, 6],
        [0, 3, 6, 9],
        [1, 3, 5, 7],
        [0, 1, 2, 3],
        [0, 2, 5, 7],
        [0, 4, 8, 10],
    ]
    for subset in subsets:
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        configs.append((centers, [0.0] * N))
    
    # Linear zigzag
    for spacing in [3.5, 3.8, 4.0]:
        centers = []
        for i in range(N):
            x = (i - N//2) * spacing
            y = 0.5 * spacing * (1 if i % 2 == 0 else -1)
            centers.append((x, y))
        configs.append((centers, [0.0] * N))
    
    # Circular
    for radius in [2.5, 3.0, 3.5]:
        centers = [(0.0, 0.0)]
        for i in range(N - 1):
            angle = 2 * math.pi * i / (N - 1)
            centers.append((radius * math.cos(angle), radius * math.sin(angle)))
        configs.append((centers, [0.0] * N))
    
    # Random
    np.random.seed(123)
    for _ in range(5):
        centers = [(0.0, 0.0)]
        for i in range(N - 1):
            angle = np.random.uniform(0, 2 * math.pi)
            r = np.random.uniform(1.5, 4.0)
            centers.append((r * math.cos(angle), r * math.sin(angle)))
        configs.append((centers, [0.0] * N))
    
    return configs

def _simulate_annealing(initial_centers, initial_angles, max_time=60.0):
    start_time = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    
    current_side, current_center, current_angle = _fast_optimize_outer(centers, angles)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None, None
        
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center
    best_angle = current_angle
    
    T = 0.5
    T_min = 0.005
    cooling_rate = 0.9995
    
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Perturb 1-2 random hexagons
        n_perturb = np.random.randint(1, 3)
        for _ in range(n_perturb):
            idx = np.random.randint(N)
            new_centers[idx][0] += np.random.normal(0, 0.15)
            new_centers[idx][1] += np.random.normal(0, 0.15)
            if np.random.rand() < 0.4:
                new_angles[idx] += np.random.normal(0, 5.0)
        
        # Occasionally perturb all slightly
        if np.random.rand() < 0.08:
            dx = np.random.normal(0, 0.03)
            dy = np.random.normal(0, 0.03)
            for i in range(N):
                new_centers[i][0] += dx
                new_centers[i][1] += dy
        
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            T *= cooling_rate
            continue
            
        new_side, new_center, new_angle = _fast_optimize_outer(new_centers, new_angles)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            centers = new_centers
            angles = new_angles
            current_side = new_side
            current_center = new_center
            current_angle = new_angle
            
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in centers]
                best_angles = list(angles)
                best_center = current_center
                best_angle = current_angle
        else:
            prob = math.exp(-delta / T)
            if np.random.rand() < prob:
                centers = new_centers
                angles = new_angles
                current_side = new_side
                current_center = new_center
                current_angle = new_angle
                
        T *= cooling_rate
        
    return best_centers, best_angles, best_side, best_center, best_angle

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate and evaluate initial configurations
    initial_configs = _generate_initial_configurations()
    
    best_init_centers = None
    best_init_angles = None
    best_init_side = float('inf')
    
    for centers, angles in initial_configs:
        if time.time() - start_time > 15:
            break
        side, result = _try_arrangement(centers, angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result

    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        best_init_centers = fb_centers
        best_init_angles = fb_angles
        best_init_side = fb_side

    # Run multiple SA restarts
    sa_time_budget = min(80.0, DEADLINE - (time.time() - start_time) - 20)
    if sa_time_budget > 10:
        # Split time among restarts
        n_restarts = min(3, max(1, int(sa_time_budget / 25)))
        time_per_restart = sa_time_budget / n_restarts
        
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            if restart == 0:
                init_c = best_init_centers
                init_a = best_init_angles
            else:
                # Generate a new random initial config
                np.random.seed(100 + restart * 7)
                centers = [(0.0, 0.0)]
                for i in range(N - 1):
                    angle = np.random.uniform(0, 2 * math.pi)
                    r = np.random.uniform(1.8, 4.0)
                    centers.append((r * math.cos(angle), r * math.sin(angle)))
                init_c = centers
                init_a = [0.0] * N
            
            sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
                init_c, init_a, max_time=time_per_restart
            )
            
            if sa_side < best_side and sa_side < float('inf'):
                if _check_validity(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
                    best_side = sa_side
                    best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)

    # High-precision polish
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        centers, inner_angles, outer_center, outer_side, outer_angle = best_result
        polished_centers, polished_angles, polished_side, polished_center, polished_angle = _polish_slsqp(
            centers, inner_angles, max_time=min(15.0, DEADLINE - (time.time() - start_time) - 5)
        )
        if polished_side < best_side and np.isfinite(polished_side):
            best_side = polished_side
            best_result = (polished_centers, polished_angles, polished_center, polished_side, polished_angle)

    # Final verification
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
