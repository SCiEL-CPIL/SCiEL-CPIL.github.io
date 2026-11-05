import numpy as np
import time
import math
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 140.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_hex_edge_normals(rot_deg):
    """Pre-compute the 6 edge normals for a unit hexagon with given rotation."""
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + rot_deg)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def sat_overlap_fast(v1, v2, normals1, normals2):
    """
    Fast SAT overlap check using pre-computed normals.
    Returns positive separation if disjoint, negative penetration if overlapping.
    """
    axes = np.vstack((normals1, normals2))
    min_overlap = float('inf')
    
    for k in range(12):
        n = axes[k]
        proj1 = v1 @ n
        min1, max1 = proj1.min(), proj1.max()
        proj2 = v2 @ n
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    
    return -min_overlap

def outer_hex_side_fast(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    """
    Find the optimal rotation of the outer hexagon to minimize side length.
    """
    best_L = float('inf')
    best_rot = init_rot
    
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side_fast(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    local_rots = np.linspace(best_rot - 2.0, best_rot + 2.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side_fast(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    return best_L, best_rot

def compute_L_and_penalties(params):
    """
    Compute outer hexagon side length and total overlap penalty.
    Returns (L, penalty) where penalty is sum of squared penetrations.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    normals_list = [get_hex_edge_normals(r) for x, y, r in xys_rots]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap_fast(vertices_list[i], vertices_list[j], normals_list[i], normals_list[j])
            if d < 0:
                penalty += d * d
    
    all_points = np.vstack(vertices_list)
    L = outer_hex_side_fast(all_points, outer_rot)
    
    return L, penalty

def smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m < -50.0 / beta:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def objective_combined(params, beta=50.0, penalty_weight=100.0):
    """
    Combined objective: smooth outer side length + smooth penalty for overlaps.
    Uses per-face smooth max for better numerical stability.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    normals_list = [get_hex_edge_normals(r) for x, y, r in xys_rots]
    
    # Compute smooth overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap_fast(vertices_list[i], vertices_list[j], normals_list[i], normals_list[j])
            if d < 0:
                penalty += d * d
    
    # Compute smooth outer side length using per-face smooth max
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T  # shape (72, 6)
    
    # For each face, compute smooth max of absolute projections
    abs_proj = np.abs(proj)
    face_maxes = np.array([smooth_max(abs_proj[:, k], beta=beta) for k in range(6)])
    max_proj_smooth = smooth_max(face_maxes, beta=beta)
    L_smooth = (2.0 / np.sqrt(3.0)) * max_proj_smooth
    
    return L_smooth + penalty_weight * penalty

def is_valid(params, tol=1e-8):
    """Check if configuration has no overlaps."""
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    normals_list = [get_hex_edge_normals(r) for x, y, r in xys_rots]
    
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap_fast(vertices_list[i], vertices_list[j], normals_list[i], normals_list[j])
            if d < -tol:
                return False
    return True

def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    """Create a hex grid of unit hexagons with optional gap."""
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = sqrt3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def make_ring(gap=0.0):
    """1 center + 6 ring + 5 outer with gap."""
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def simulated_annealing(params, start_temp=1.0, end_temp=0.01, n_iter=2000, time_limit=30.0, start_time=None):
    """
    Simulated annealing to find valid configurations with low L.
    """
    if start_time is None:
        start_time = time.time()
    
    current_params = params.copy()
    current_L, current_penalty = compute_L_and_penalties(current_params)
    current_obj = current_L + 100.0 * current_penalty
    
    best_params = current_params.copy()
    best_L = current_L
    best_obj = current_obj
    
    temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_params = current_params.copy()
        n_perturb = random.choice([1, 1, 2])
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            new_params[idx*3] += random.gauss(0, 0.3)
            new_params[idx*3+1] += random.gauss(0, 0.3)
            if random.random() < 0.3:
                new_params[idx*3+2] += random.gauss(0, 10.0)
        if random.random() < 0.1:
            new_params[3*N] += random.gauss(0, 5.0)
        
        new_L, new_penalty = compute_L_and_penalties(new_params)
        new_obj = new_L + 100.0 * new_penalty
        
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / max(temp, 1e-10)):
            current_params = new_params
            current_L = new_L
            current_penalty = new_penalty
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_params = current_params.copy()
                best_L = current_L
                best_obj = current_obj
        
        temp *= cooling_rate
    
    return best_params, best_L

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0]
    
    # Phase 1: Differential evolution for global exploration
    if time_left() > 80:
        try:
            def de_objective(v):
                return objective_combined(v, beta=40.0, penalty_weight=50.0)
            
            bounds = [(-5, 5)] * (2 * N) + [(-180, 180)] * N + [(-30, 30)]
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=30,
                tol=1e-4,
                seed=42,
                popsize=15,
                polish=False,
                updating='deferred',
                init='sobol',
                time_left=time_left
            )
            
            if de_result.success or is_valid(de_result.x):
                if is_valid(de_result.x):
                    xys_rots = de_result.x[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side_fast(all_points, de_result.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = de_result.x.copy()
                        
        except Exception:
            pass
    
    # Phase 2: Evaluate initial configs and run simulated annealing
    for cfg in configs:
        if time_left() < 30:
            break
        
        for orot in outer_rots:
            if time_left() < 30:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            if is_valid(init_params):
                xys_rots = init_params[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side_fast(all_points, orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            if time_left() < 35:
                continue
            
            try:
                sa_params, sa_L = simulated_annealing(
                    init_params, 
                    start_temp=1.5, 
                    end_temp=0.05,
                    n_iter=1500,
                    time_limit=10.0,
                    start_time=start_time
                )
                
                if is_valid(sa_params):
                    xys_rots = sa_params[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side_fast(all_points, sa_params[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = sa_params.copy()
                        
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish from best result
    if best_params is not None and time_left() > 40:
        try:
            res_nm = minimize(
                lambda p: objective_combined(p, beta=30.0, penalty_weight=100.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            if is_valid(res_nm.x):
                xys_rots = res_nm.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side_fast(all_points, res_nm.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
                    
        except Exception:
            pass
    
    # Phase 4: Coordinate descent refinement
    if best_params is not None and time_left() > 25:
        try:
            current = best_params.copy()
            for iteration in range(80):
                if time_left() < 20:
                    break
                
                improved = False
                for i in range(N):
                    for axis in [0, 1, 2]:
                        for delta in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01]:
                            trial = current.copy()
                            trial[i*3 + axis] += delta
                            if is_valid(trial):
                                L_trial, _ = compute_L_and_penalties(trial)
                                if L_trial < best_L - 1e-9:
                                    best_L = L_trial
                                    best_params = trial.copy()
                                    current = trial.copy()
                                    improved = True
                
                if not improved:
                    break
                    
        except Exception:
            pass
    
    # Phase 5: Optimize outer rotation independently
    if best_params is not None and time_left() > 15:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=1440)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Phase 6: Final high-precision polish
    if best_params is not None and time_left() > 10:
        try:
            res_final = minimize(
                lambda p: objective_combined(p, beta=80.0, penalty_weight=200.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                xys_rots = res_final.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side_fast(all_points, res_final.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
    
    # Phase 7: Fine outer rotation search
    if best_params is not None and time_left() > 5:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Phase 8: Micro-polish outer rotation
    if best_params is not None and time_left() > 2:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            cur_rot = best_params[3*N]
            fine_rots = np.linspace(cur_rot - 0.1, cur_rot + 0.1, 500)
            best_local_L = best_L
            best_local_rot = cur_rot
            for rot in fine_rots:
                L = outer_hex_side_fast(all_points, rot)
                if L < best_local_L:
                    best_local_L = L
                    best_local_rot = rot
            
            if best_local_L < best_L:
                best_L = best_local_L
                best_params[3*N] = best_local_rot
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side_fast(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
