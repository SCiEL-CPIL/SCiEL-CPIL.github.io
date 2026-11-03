import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 145.0  # Safety margin for 150s limit
SQRT3 = math.sqrt(3.0)

def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon."""
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def _outer_side(theta, centers, theta_outer):
    """Minimal outer side length containing all inner hexagon vertices."""
    verts_local = _hex_vertices_local(theta)
    # verts shape (N, 6, 2)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem = proj.max()
    return apothem * 2.0 / SQRT3

def _is_valid(centers, theta, theta_outer, tol=1e-6):
    """Check non-overlap and containment."""
    # Check pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    
    # Check containment
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    s = _outer_side(theta, centers, theta_outer)
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True

def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """
    Use LP to find the optimal outer hexagon center and inradius
    for given inner hexagons and outer angle.
    Returns (cx, cy, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle_deg in inner_hexes:
        iangle = math.radians(iangle_deg)
        for k in range(6):
            a = iangle + math.pi * k / 3
            all_vertices.append((ix + math.cos(a), iy + math.sin(a)))

    n_pts = len(all_vertices)
    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)

        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _find_best_outer_lp(inner_hexes, time_limit=5.0):
    """
    Find best outer hexagon angle and center via LP.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse search
    coarse = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_angle >= 0 and time.time() < start + time_limit:
        # Fine search
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.01)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
                    
    return best_center[0], best_center[1], best_side, best_angle

def _get_inner_hexes(centers, theta):
    return [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]

def _objective(centers, theta, theta_outer):
    """
    Objective for inner optimization given fixed outer angle.
    Minimizes side length + penalty for overlap.
    """
    s = _outer_side(theta, centers, theta_outer)
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    pen = np.sum(viol * viol)
    return s + 1000.0 * pen

def _optimize_inner(centers_init, theta_init, theta_outer, maxiter=200):
    """
    Optimize inner positions and rotation for fixed outer angle.
    """
    def obj(params):
        c = params[:N * 2].reshape(N, 2)
        t = params[N * 2]
        return _objective(c, t, theta_outer)
        
    params_init = np.concatenate([centers_init.reshape(-1), [theta_init]])
    
    try:
        res = minimize(obj, params_init, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success or res.status in [1, 2]:
            c_opt = res.x[:N * 2].reshape(N, 2)
            t_opt = res.x[N * 2]
            if _is_valid(c_opt, t_opt, theta_outer):
                return c_opt, t_opt
    except Exception:
        pass
    return None

def _basin_hopping_step(centers, theta, step_size):
    """
    Perform a random perturbation and re-optimize outer hexagon.
    Returns new centers, theta, side, or None if invalid.
    """
    # Perturb centers
    rng = np.random.RandomState()
    new_centers = centers + rng.uniform(-step_size, step_size, size=(N, 2))
    new_theta = theta + rng.uniform(-0.1, 0.1)
    
    # Check validity of perturbed state roughly (optional, LP will handle containment, but overlap is hard)
    # We rely on the optimizer to fix overlaps, but if they are too bad, it might fail.
    
    # Optimize outer hexagon for this configuration
    inner_hexes = _get_inner_hexes(new_centers, new_theta)
    lp_res = _find_best_outer_lp(inner_hexes, time_limit=2.0)
    
    if lp_res is None:
        return None
    
    cx, cy, side, angle_deg = lp_res
    angle_rad = math.radians(angle_deg)
    
    # Check if the configuration is valid with this outer hexagon
    # Note: _is_valid checks if the *given* outer hexagon contains them.
    # The LP found the *minimal* outer hexagon for this angle.
    # So if _is_valid passes, it means the minimal hexagon contains them and they don't overlap.
    if _is_valid(new_centers, new_theta, angle_rad):
        return new_centers, new_theta, side, angle_rad
    
    return None

def construct():
    best_score = float('inf')
    best_result = None # (centers, theta, theta_outer, side)
    
    # Initial configurations
    inits = []
    
    # 1. 7-cluster + 4 at valleys (dist 3.46)
    c1 = np.zeros((N, 2))
    c1[0] = [0, 0]
    for i in range(6):
        ang = i * math.pi / 3.0
        c1[i+1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    # 4 extra at 30, 150, 270, 390
    extra_angles = [math.pi/6, 5*math.pi/6, 3*math.pi/2, 13*math.pi/6]
    for i, ang in enumerate(extra_angles):
        c1[i+7] = [3.46 * math.cos(ang), 3.46 * math.sin(ang)]
    inits.append((c1, 0.0))
    
    # 2. 7-cluster + 4 at corners (dist 4) - might be too big but good start
    c2 = c1.copy()
    for i in range(4):
        ang = i * math.pi / 2.0
        c2[i+7] = [4.0 * math.cos(ang), 4.0 * math.sin(ang)]
    inits.append((c2, 0.0))
    
    # 3. Random valid-ish start
    c3 = np.zeros((N, 2))
    c3[0] = [0,0]
    for i in range(6):
        ang = i * math.pi / 3.0
        c3[i+1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    # Place 4 randomly in a ring
    rng = np.random.RandomState(42)
    for i in range(4):
        ang = rng.uniform(0, 2*math.pi)
        r = rng.uniform(3.0, 4.0)
        c3[i+7] = [r * math.cos(ang), r * math.sin(ang)]
    inits.append((c3, 0.0))
    
    # 4. Optimizer start from 7-cluster + random 4
    c4 = c1.copy()
    rng = np.random.RandomState(123)
    for i in range(4):
        c4[i+7] = rng.uniform(-4, 4, size=2)
    inits.append((c4, 0.0))

    for init_centers, init_theta in inits:
        if time.time() > DEADLINE - 20:
            break
            
        # Step 1: Optimize inner for a few fixed outer angles
        for theta_o in [0.0, math.pi/6, math.pi/12]:
            if time.time() > DEADLINE - 10:
                break
            opt_res = _optimize_inner(init_centers, init_theta, theta_o)
            if opt_res:
                c_opt, t_opt = opt_res
                # Find best outer for this optimized inner
                inner_hexes = _get_inner_hexes(c_opt, t_opt)
                lp_res = _find_best_outer_lp(inner_hexes, time_limit=3.0)
                if lp_res:
                    cx, cy, side, angle_deg = lp_res
                    angle_rad = math.radians(angle_deg)
                    if _is_valid(c_opt, t_opt, angle_rad):
                        if side < best_score:
                            best_score = side
                            best_result = (c_opt.copy(), t_opt, angle_rad, side)
        
        # Step 2: Basin Hopping from the best found so far (or init if none)
        if best_result:
            cur_c, cur_t, cur_to, cur_s = best_result
        else:
            cur_c, cur_t = init_centers, init_theta
            cur_to = 0.0
            cur_s = float('inf')
            # Get a valid state first
            inner_hexes = _get_inner_hexes(cur_c, cur_t)
            lp_res = _find_best_outer_lp(inner_hexes, time_limit=2.0)
            if lp_res and _is_valid(cur_c, cur_t, math.radians(lp_res[3])):
                cur_s = lp_res[2]
                cur_to = math.radians(lp_res[3])
            else:
                continue

        # Basin Hopping Loop
        T = 0.5
        step_size = 0.5
        for step in range(20):
            if time.time() > DEADLINE - 5:
                break
            if T < 0.01:
                break
                
            res = _basin_hopping_step(cur_c, cur_t, step_size)
            if res:
                new_c, new_t, new_s, new_to = res
                delta = new_s - cur_s
                if delta < 0 or np.random.random() < math.exp(-delta / T):
                    cur_c, cur_t, cur_s, cur_to = new_c, new_t, new_s, new_to
                    if cur_s < best_score:
                        best_score = cur_s
                        best_result = (cur_c.copy(), cur_t, cur_to, cur_s)
            T *= 0.8
            step_size *= 0.9

    if best_result is None:
        # Fallback: Return a valid, albeit large, configuration
        centers = np.zeros((N, 2))
        centers[0] = [0,0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i+1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = i * math.pi / 2.0
            centers[i+7] = [4.0 * math.cos(ang), 4.0 * math.sin(ang)]
        # Find outer
        inner_hexes = _get_inner_hexes(centers, 0.0)
        lp_res = _find_best_outer_lp(inner_hexes, time_limit=2.0)
        if lp_res:
            side = lp_res[2]
            angle = lp_res[3]
        else:
            side = 10.0
            angle = 0.0
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(side), float(angle)

    centers, theta, theta_o, side = best_result
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(math.degrees(theta_o))
