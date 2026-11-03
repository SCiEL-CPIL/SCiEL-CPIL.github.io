import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 146.0  # Safety margin below 150s

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
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            if dist_sq < 4.0 - 1e-9:
                return False
            
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
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

def _quick_score(cx_arr, cy_arr, angle_arr, normals, apothem_unit):
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

def construct():
    t0 = time.time()
    random.seed(42)
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
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
    
    def make_compact_lattice():
        """Compact 4x3 triangular lattice with proper hexagonal spacing."""
        pts = []
        dx = 2.0  # horizontal spacing for touching hexagons
        dy = math.sqrt(3)  # vertical spacing
        rows = 3
        cols = 4
        for r in range(rows):
            for c in range(cols):
                x = c * dx + (r % 2) * dx * 0.5
                y = r * dy
                pts.append([x, y, 0.0])
        return pts[:12]
    
    initializers = [
        make_compact_lattice,
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 80:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    if best_cfg is None:
        cfg = make_compact_lattice()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # --- Phase 1: Rigid-body dynamics with repulsive/containment forces ---
    def _dynamics_settle(cfg, n_steps=2000, dt=0.01, damping=0.98):
        """
        Symplectic Euler integration with:
        - Repulsive forces between overlapping/near hexagons
        - Containment force pulling toward center
        - Damping to settle into equilibrium
        """
        cx = np.array([p[0] for p in cfg], dtype=float)
        cy = np.array([p[1] for p in cfg], dtype=float)
        ang = np.array([p[2] for p in cfg], dtype=float)
        vx = np.zeros(N)
        vy = np.zeros(N)
        va = np.zeros(N)
        
        min_dist = 2.0  # minimum non-overlap distance
        repulse_strength = 50.0
        contain_strength = 1.0
        
        for step in range(n_steps):
            if time.time() - t0 > TIME_LIMIT - 50:
                break
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            ta = np.zeros(N)
            
            # Pairwise repulsive forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1e-9:
                        dist = 1e-9
                        dx = random.uniform(-0.1, 0.1)
                        dy = random.uniform(-0.1, 0.1)
                    
                    if dist < min_dist:
                        # Strong repulsion when too close
                        overlap = min_dist - dist
                        force = repulse_strength * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
                    elif dist < min_dist * 1.2:
                        # Weaker repulsion to prevent clustering
                        overlap = min_dist * 1.2 - dist
                        force = repulse_strength * 0.1 * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
            
            # Containment force: pull toward center
            for i in range(N):
                r = math.sqrt(cx[i] * cx[i] + cy[i] * cy[i])
                if r > 0.1:
                    fx[i] -= contain_strength * cx[i]
                    fy[i] -= contain_strength * cy[i]
            
            # Symplectic Euler update
            for i in range(N):
                vx[i] = (vx[i] + fx[i] * dt) * damping
                vy[i] = (vy[i] + fy[i] * dt) * damping
                va[i] = (va[i] + ta[i] * dt) * damping
                
                # Velocity limit
                speed = math.sqrt(vx[i]*vx[i] + vy[i]*vy[i])
                if speed > 1.0:
                    vx[i] *= 1.0 / speed
                    vy[i] *= 1.0 / speed
                
                cx[i] += vx[i] * dt
                cy[i] += vy[i] * dt
                ang[i] += va[i] * dt
            
            # Keep angles in [0, 60)
            ang = ang % 60.0
        
        # Check validity and return
        if _is_valid_fast(cx, cy, ang):
            side = _quick_score(cx, cy, ang, normals, apothem_unit)
            return [[cx[i], cy[i], ang[i]] for i in range(N)], side
        return None, float('inf')
    
    # Run dynamics from best initial config
    dyn_cfg, dyn_side = _dynamics_settle(best_cfg, n_steps=3000, dt=0.008, damping=0.97)
    if dyn_cfg is not None and dyn_side < best_side:
        best_side = dyn_side
        best_cfg = dyn_cfg
    
    # Try dynamics from other initializers
    for init_func in [make_flower_initial, make_ring_initial, make_compact_lattice]:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        dyn_cfg, dyn_side = _dynamics_settle(cfg, n_steps=2000, dt=0.01, damping=0.98)
        if dyn_cfg is not None and dyn_side < best_side:
            best_side = dyn_side
            best_cfg = dyn_cfg
    
    # --- Phase 2: Gradient-based optimization ---
    def _sdf_objective(x):
        cx = x[0::3]
        cy = x[1::3]
        ang = x[2::3]
        
        # SDF penalty for overlapping hexagons
        sdf_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 2.0:
                    violation = 2.0 - dist
                    sdf_pen += violation * violation
        
        # Outer hex side length
        side = _quick_score(cx, cy, ang, normals, apothem_unit)
        
        return side + 100.0 * sdf_pen
    
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    x0 = np.concatenate([cx_arr, cy_arr, angle_arr])
    
    n_gradient_rounds = 6
    for round_idx in range(n_gradient_rounds):
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        
        if round_idx > 0:
            x0 = x0 + np.random.normal(0, 0.05, size=x0.shape)
        
        try:
            result = minimize(
                _sdf_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 600}
            )
            
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                side_opt = _quick_score(cx_opt, cy_opt, ang_opt, normals, apothem_unit)
                if side_opt < best_side:
                    best_side = side_opt
                    best_cfg = [[cx_opt[i], cy_opt[i], ang_opt[i]] for i in range(N)]
                    x0 = x_opt.copy()
        except Exception:
            pass
    
    # --- Phase 3: Simulated annealing fine-tuning ---
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    cur_score = best_side
    
    T = 0.2
    T_end = 0.001
    alpha = 0.997
    
    while time.time() - t0 < TIME_LIMIT - 3:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        idx2 = -1
        old_x2, old_y2, old_a2 = 0, 0, 0
        
        if r < 0.7:
            cx_arr[idx] += random.gauss(0, 0.05)
            cy_arr[idx] += random.gauss(0, 0.05)
            angle_arr[idx] += random.gauss(0, 1.0)
        elif r < 0.85:
            angle_arr[idx] += random.gauss(0, 3.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.04)
            cy_arr[idx] += random.gauss(0, 0.04)
            cx_arr[idx2] += random.gauss(0, 0.04)
            cy_arr[idx2] += random.gauss(0, 0.04)
            angle_arr[idx] += random.gauss(0, 0.8)
            angle_arr[idx2] += random.gauss(0, 0.8)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr, normals, apothem_unit)
            
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if r >= 0.85:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if r >= 0.85:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # --- Phase 4: Outer hexagon angle refinement ---
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_compact_lattice()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    else:
        best_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
    
    # Try different outer hexagon angles to minimize side length
    for outer_angle in range(0, 60, 5):
        side_test = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), float(outer_angle))
        if side_test < best_side:
            best_side = side_test
            best_angle = float(outer_angle)
    
    # Fine-tune outer angle
    best_angle_fine = best_angle
    for da in [1.0, 0.5, 0.25, 0.1]:
        improved = True
        while improved and time.time() - t0 < TIME_LIMIT - 1:
            improved = False
            for delta in [-da, da]:
                test_angle = (best_angle_fine + delta) % 60.0
                side_test = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), test_angle)
                if side_test < best_side - 1e-6:
                    best_side = side_test
                    best_angle_fine = test_angle
                    improved = True
    best_angle = best_angle_fine
    
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
