import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 115.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS

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

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    # Vectorized distance check for fast rejection
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            overlap = True
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            
            if overlap:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            
            if overlap:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _quick_score(cx_arr, cy_arr, angle_arr):
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_need = 0.0
    for i in range(N):
        a = math.radians(angle_arr[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _sdf_hexagon_point(px, py, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    apothem = math.sqrt(3) / 2.0
    
    max_dist = 0.0
    for k in range(6):
        na = math.radians(60 * k + 30)
        nx, ny = math.cos(na), math.sin(na)
        dist = nx * lx + ny * ly - apothem
        if dist > max_dist:
            max_dist = dist
    
    if max_dist < 0:
        min_dist_to_edge = float('inf')
        for k in range(6):
            na = math.radians(60 * k + 30)
            nx, ny = math.cos(na), math.sin(na)
            dist = apothem - (nx * lx + ny * ly)
            if dist > 0 and dist < min_dist_to_edge:
                min_dist_to_edge = dist
        return -min_dist_to_edge
    else:
        return max_dist

def _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2):
    verts1 = _hex_vertices(cx1, cy1, a1)
    verts2 = _hex_vertices(cx2, cy2, a2)
    
    min_sdf = float('inf')
    for (vx, vy) in verts1:
        d = _sdf_hexagon_point(vx, vy, cx2, cy2, a2)
        if d < min_sdf:
            min_sdf = d
    for (vx, vy) in verts2:
        d = _sdf_hexagon_point(vx, vy, cx1, cy1, a1)
        if d < min_sdf:
            min_sdf = d
    
    d_center = math.hypot(cx1 - cx2, cy1 - cy2)
    if d_center > 2 * math.sqrt(3):
        return d_center - 2 * math.sqrt(3)
    
    return min_sdf

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]
    
    def make_lattice_initial():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
        return pts

    def make_ring_initial():
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # 1. Initialize
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # 2. Differential Evolution (Global Search)
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    pop_size = 15
    population = []
    population.append([cx_arr.copy(), cy_arr.copy(), angle_arr.copy(), best_side])
    
    for _ in range(pop_size - 1):
        p = best_cfg[:]
        for i in range(N):
            p[i][0] += random.gauss(0, 0.5)
            p[i][1] += random.gauss(0, 0.5)
            p[i][2] += random.gauss(0, 10.0)
        cx_p = np.array([q[0] for q in p])
        cy_p = np.array([q[1] for q in p])
        a_p = np.array([q[2] for q in p])
        if _is_valid_fast(cx_p, cy_p, a_p):
            score = _quick_score(cx_p, cy_p, a_p)
            population.append([cx_p, cy_p, a_p, score])
        else:
            population.append([cx_p, cy_p, a_p, float('inf')])
    
    # Sort population
    population.sort(key=lambda x: x[3])
    
    # DE loop
    F = 0.8
    CR = 0.9
    gen = 0
    while time.time() - t0 < TIME_LIMIT - 60:
        gen += 1
        if len(population) < 3:
            break
        new_pop = [population[0][:]]
        for i in range(1, pop_size):
            r1, r2, r3 = random.sample(range(pop_size), 3)
            cx_v = population[r1][0].copy()
            cy_v = population[r1][1].copy()
            a_v = population[r1][2].copy()
            
            cx_c = population[r2][0].copy()
            cy_c = population[r2][1].copy()
            a_c = population[r2][2].copy()
            
            cx_t = population[i][0].copy()
            cy_t = population[i][1].copy()
            a_t = population[i][2].copy()
            
            for k in range(N):
                if random.random() < CR:
                    cx_t[k] = cx_v[k] + F * (cx_c[k] - population[r3][0][k])
                    cy_t[k] = cy_v[k] + F * (cy_c[k] - population[r3][1][k])
                    a_t[k] = a_v[k] + F * (a_c[k] - population[r3][2][k])
                else:
                    cx_t[k] = population[i][0][k]
                    cy_t[k] = population[i][1][k]
                    a_t[k] = population[i][2][k]
            
            if _is_valid_fast(cx_t, cy_t, a_t):
                score = _quick_score(cx_t, cy_t, a_t)
            else:
                score = float('inf')
            
            if score < population[i][3]:
                new_pop.append([cx_t, cy_t, a_t, score])
                if score < best_side:
                    best_side = score
                    best_cfg = [[cx_t[k], cy_t[k], a_t[k]] for k in range(N)]
            else:
                new_pop.append(population[i][:])
        
        population = new_pop
        population.sort(key=lambda x: x[3])
        
        # Elitism: keep best
        if population[0][3] < best_side:
            best_side = population[0][3]
            best_cfg = [[population[0][0][k], population[0][1][k], population[0][2][k]] for k in range(N)]
    
    # 3. Physics-based relaxation (Symplectic Euler)
    cfg = [row[:] for row in best_cfg]
    xs = np.array([p[0] for p in cfg])
    ys = np.array([p[1] for p in cfg])
    angles = np.array([p[2] for p in cfg])
    vx = np.zeros(N)
    vy = np.zeros(N)
    va = np.zeros(N)
    
    dt = 0.02
    damping = 0.95
    repulse_k = 5.0
    attract_k = 0.5
    max_steps = 300
    
    for step in range(max_steps):
        if time.time() - t0 > TIME_LIMIT - 40:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        fa = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.hypot(dx, dy)
                
                if dist > 2 * math.sqrt(3):
                    continue
                
                sdf = _sdf_hexagon_hexagon(xs[i], ys[i], angles[i], xs[j], ys[j], angles[j])
                
                if sdf < 0:
                    force_mag = repulse_k * (-sdf)
                    if dist > 1e-6:
                        fx[i] += force_mag * dx / dist
                        fy[i] += force_mag * dy / dist
                        fx[j] -= force_mag * dx / dist
                        fy[j] -= force_mag * dy / dist
                        fa[i] += force_mag * 0.1 * random.uniform(-1, 1)
                        fa[j] -= force_mag * 0.1 * random.uniform(-1, 1)
        
        for i in range(N):
            r = math.hypot(xs[i], ys[i])
            if r > 5.0:
                fx[i] -= attract_k * (r - 5.0) * xs[i] / r
                fy[i] -= attract_k * (r - 5.0) * ys[i] / r
        
        for i in range(N):
            vx[i] = (vx[i] + dt * fx[i]) * damping
            vy[i] = (vy[i] + dt * fy[i]) * damping
            va[i] = (va[i] + dt * fa[i]) * damping
            
            xs[i] += dt * vx[i]
            ys[i] += dt * vy[i]
            angles[i] += dt * va[i]
        
        if _is_valid_fast(xs, ys, angles):
            side = _quick_score(xs, ys, angles)
            if side < best_side:
                best_side = side
                best_cfg = [[xs[i], ys[i], angles[i]] for i in range(N)]
    
    # 4. Simulated Annealing
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.3
    T_end = 0.001
    alpha = 0.995
    
    while time.time() - t0 < TIME_LIMIT - 15:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        idx2 = -1
        old_x2, old_y2, old_a2 = 0, 0, 0
        
        if r < 0.7:
            cx_arr[idx] += random.gauss(0, 0.12)
            cy_arr[idx] += random.gauss(0, 0.12)
            angle_arr[idx] += random.gauss(0, 2.5)
        elif r < 0.85:
            angle_arr[idx] += random.gauss(0, 6.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if idx2 != -1:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # 5. L-BFGS-B refinement
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    def _joint_objective_and_grad(params):
        n_params = 3 * N + 2
        if len(params) != n_params:
            return 0, np.zeros(n_params)
        
        xs = params[0:N]
        ys = params[N:2*N]
        angles = params[2*N:3*N]
        cx_out = params[3*N]
        cy_out = params[3*N + 1]
        
        a_out = 0.0
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a_out + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        max_need = 0.0
        for i in range(N):
            verts = _hex_vertices(xs[i], ys[i], angles[i])
            for (vx, vy) in verts:
                for (nx, ny) in normals_out:
                    val = (nx * (vx - cx_out) + ny * (vy - cy_out)) / apothem_unit
                    if val > max_need:
                        max_need = val
        
        penalty = 0.0
        grad = np.zeros(n_params)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq > (2 * math.sqrt(3)) ** 2:
                    continue
                
                min_dist = 2.0
                if dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    overlap = min_dist - dist
                    if overlap > 0:
                        penalty += overlap * overlap
                        if dist > 1e-6:
                            gx = dx / dist
                            gy = dy / dist
                            grad[i] += 2 * overlap * gx
                            grad[N + i] += 2 * overlap * gy
                            grad[j] -= 2 * overlap * gx
                            grad[N + j] -= 2 * overlap * gy
        
        penalty_weight = 10.0
        obj = max_need + penalty_weight * penalty
        
        return obj, grad
    
    init_params = np.zeros(3 * N + 2)
    for i in range(N):
        init_params[i] = cx_best[i]
        init_params[N + i] = cy_best[i]
        init_params[2 * N + i] = angle_best[i]
    init_params[3 * N] = 0.0
    init_params[3 * N + 1] = 0.0
    
    if time.time() - t0 < TIME_LIMIT - 5:
        try:
            result = minimize(
                _joint_objective_and_grad,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < best_side:
                opt_params = result.x
                xs_opt = opt_params[0:N]
                ys_opt = opt_params[N:2*N]
                angles_opt = opt_params[2*N:3*N]
                cx_out_opt = opt_params[3*N]
                cy_out_opt = opt_params[3*N + 1]
                
                if _is_valid_fast(xs_opt, ys_opt, angles_opt):
                    side_opt = _outer_hex_side(xs_opt, ys_opt, angles_opt, (cx_out_opt, cy_out_opt), 0.0)
                    if side_opt < best_final_side:
                        best_final_side = side_opt
                        best_final_cfg = [[xs_opt[i], ys_opt[i], angles_opt[i]] for i in range(N)]
                        best_final_center = (cx_out_opt, cy_out_opt)
                        best_final_angle = 0.0
        except Exception:
            pass
    
    # 6. Refine outer hexagon center and angle
    cx_best2 = np.array([p[0] for p in best_final_cfg])
    cy_best2 = np.array([p[1] for p in best_final_cfg])
    angle_best2 = np.array([p[2] for p in best_final_cfg])
    
    verts_best = [_hex_vertices(cx_best2[i], cy_best2[i], angle_best2[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 2):
        if time.time() - t0 > TIME_LIMIT - 1:
            break
        
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center[0], best_final_center[1]
        for _iter in range(10):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
                for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_lattice_initial()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
