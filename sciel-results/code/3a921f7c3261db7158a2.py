import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 140.0
INRADIUS = math.sqrt(3) / 2.0

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

def _is_valid(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    # Quick distance check
    min_dist_sq = (2 * INRADIUS) ** 2 - 1e-9
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
            # Check axes from i
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            
            if overlap:
                # Check axes from j
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
    """
    Calculate the minimum side length of the outer hexagon (centered at center, rotated by angle_deg)
    to contain all inner hexagons.
    """
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # Normals of the outer hexagon sides
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    
    # For each inner hexagon, we need to check its vertices against the outer hexagon's supporting lines.
    # Actually, since both are convex, the maximum projection of the inner hexagon onto any of the 6 normals
    # determines the required apothem.
    # The side length S is related to apothem A by A = S * sqrt(3)/2.
    # So S = A / (sqrt(3)/2) = A * 2/sqrt(3).
    # Wait, the function returns "side".
    # Let's compute the required apothem first.
    
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                # Distance from center to this vertex along normal
                # If the vertex is "outside" the hexagon of apothem A, we need larger A.
                # Projection of (v - c) onto n.
                val = (nx * (vx - cx) + ny * (vy - cy))
                if val > max_need:
                    max_need = val
    
    # max_need is the required apothem.
    # Side length = max_need / (sqrt(3)/2)
    return max_need / apothem_unit

def _quick_score(cx_arr, cy_arr, angle_arr):
    """
    Fast estimate of the outer side length, assuming center (0,0) and angle 0.
    Used for optimization where we fix outer center/angle or optimize jointly.
    Here we assume outer hexagon is centered at origin and aligned (angle 0) for speed,
    but we will optimize the outer angle and center later.
    Actually, for the inner optimization, it's better to fix the outer hexagon to (0,0,0)
    and just minimize the max projection.
    """
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
                val = (nx * vx + ny * vy)
                if val > max_need:
                    max_need = val
    return max_need / apothem_unit

def _full_score(cfg):
    """
    cfg: list of [x, y, angle]
    Returns outer side length with optimized outer center and angle.
    """
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    if not _is_valid(cx_arr, cy_arr, angle_arr):
        return float('inf')
    
    # Optimize outer hexagon placement
    # We can fix the outer center to (0,0) by translating the inner config?
    # No, the inner config is absolute.
    # We can sweep outer angle (0 to 30 deg) and optimize center.
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Precompute vertices
    verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(N)]
    
    # Sweep outer angle
    for outer_angle_deg in range(0, 31, 2): # 0, 2, ..., 30
        a = math.radians(outer_angle_deg)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        # Optimize center (cx, cy)
        # We want to minimize max_i,k dot(n_i, v_k - c)
        # This is a convex optimization problem.
        # Simple local search or gradient step.
        # Initial guess: centroid of all vertices
        all_verts = [v for verts in verts_list for v in verts]
        cx_c = sum(v[0] for v in all_verts) / len(all_verts)
        cy_c = sum(v[1] for v in all_verts) / len(all_verts)
        
        # Local search for center
        step = 0.1
        for _iter in range(10):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_list[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c))
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            # Try 8 neighbors
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_list[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy))
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
            step *= 0.8
            
        side = max_proj / apothem_unit
        if side < best_side:
            best_side = side
            best_center = (cx_c, cy_c)
            best_angle = float(outer_angle_deg)
            
    return best_side

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # 1. Generate initial configurations
    # Honeycomb cluster
    def make_honeycomb():
        # Axial coordinates for a 12-hex cluster
        # (0,0)
        # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # (2,0), (2,-1), (2,-2), (1,-2), (0,-2)
        axial = [
            (0,0),
            (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
            (2,0), (2,-1), (2,-2), (1,-2), (0,-2)
        ]
        pts = []
        for q, r in axial:
            x = 1.5 * q
            y = math.sqrt(3) * (r + q/2.0)
            pts.append([x, y, 0.0])
        return pts

    def make_random():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    def make_ring():
        # Center + 6 around + 5 more
        pts = [[0,0,0]]
        for i in range(6):
            a = math.radians(60*i)
            x = 2*INRADIUS * math.cos(a)
            y = 2*INRADIUS * math.sin(a)
            pts.append([x, y, 0.0])
        # Add 5 more in a ring further out?
        # Just randomize the rest
        for _ in range(5):
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [make_honeycomb, make_random, make_ring]
    
    best_cfg = None
    best_side = float('inf')
    
    # Evaluate initializers
    for init_func in initializers:
        cfg = init_func()
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid(cx_arr, cy_arr, angle_arr):
            # Use full score for initial comparison? It's expensive but N=12 is small.
            # Actually, let's use quick_score first to filter, then full_score for best.
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                
    if best_cfg is None:
        best_cfg = make_honeycomb()
        # Scale up to ensure validity?
        # Honeycomb is valid.
        best_side = _quick_score(np.array([p[0] for p in best_cfg]), np.array([p[1] for p in best_cfg]), np.array([p[2] for p in best_cfg]))
        
    # 2. Differential Evolution
    # Variables: 12 * 3 = 36
    # x, y, angle for each hexagon
    
    pop_size = 30
    population = []
    
    # Add best
    population.append(best_cfg[:])
    
    # Add random perturbations
    for _ in range(pop_size - 1):
        cfg = best_cfg[:]
        for i in range(N):
            cfg[i][0] += random.gauss(0, 0.5)
            cfg[i][1] += random.gauss(0, 0.5)
            cfg[i][2] += random.gauss(0, 10.0)
        population.append(cfg)
        
    # Also add some random initializers to population
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        population.append(cfg)
        
    # Truncate population to pop_size
    population = population[:pop_size]
    
    # Evaluate population
    scores = []
    for cfg in population:
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid(cx_arr, cy_arr, angle_arr):
            # Use quick score for DE? It's faster.
            # But we want to optimize the real objective.
            # Let's use quick score for DE, and full score for the best candidate at the end.
            # Or, since N is small, maybe full score is okay?
            # Full score involves a local search for the outer hexagon.
            # Let's stick to quick score for DE iterations to save time, and refine at the end.
            score = _quick_score(cx_arr, cy_arr, angle_arr)
        else:
            score = float('inf')
        scores.append(score)
        
    # DE Loop
    F = 0.8
    CR = 0.9
    max_generations = 200
    
    for gen in range(max_generations):
        if time.time() - t0 > TIME_LIMIT - 30:
            break
            
        for i in range(pop_size):
            if time.time() - t0 > TIME_LIMIT - 30:
                break
                
            # Select 3 distinct individuals
            a, b, c = random.sample(range(pop_size), 3)
            
            trial = []
            for k in range(N):
                trial.append([
                    population[a][k][0] + F * (population[b][k][0] - population[c][k][0]),
                    population[a][k][1] + F * (population[b][k][1] - population[c][k][1]),
                    population[a][k][2] + F * (population[b][k][2] - population[c][k][2])
                ])
            
            # Crossover
            j_rand = random.randrange(N)
            candidate = population[i][:]
            for k in range(N):
                if random.random() < CR or k == j_rand:
                    candidate[k] = trial[k][:]
            
            # Evaluate candidate
            cx_arr = np.array([p[0] for p in candidate])
            cy_arr = np.array([p[1] for p in candidate])
            angle_arr = np.array([p[2] for p in candidate])
            
            if _is_valid(cx_arr, cy_arr, angle_arr):
                new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            else:
                new_score = float('inf')
                
            if new_score < scores[i]:
                population[i] = candidate
                scores[i] = new_score
                
    # Find best in population
    best_idx = np.argmin(scores)
    best_cfg = population[best_idx][:]
    
    # 3. Local Refinement
    # Use Nelder-Mead or similar on the best configuration
    # We will optimize the 36 variables.
    # Objective: _quick_score (with validity penalty)
    
    def objective(x):
        cfg = []
        for i in range(N):
            cfg.append([x[i*3], x[i*3+1], x[i*3+2]])
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if not _is_valid(cx_arr, cy_arr, angle_arr):
            return 100.0 + _quick_score(cx_arr, cy_arr, angle_arr)
        return _quick_score(cx_arr, cy_arr, angle_arr)
        
    x0 = []
    for p in best_cfg:
        x0.extend(p)
    x0 = np.array(x0)
    
    # Use scipy.optimize.minimize with Nelder-Mead
    from scipy.optimize import minimize
    
    if time.time() - t0 < TIME_LIMIT - 10:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < scores[best_idx]:
            x_opt = res.x
            cfg_opt = []
            for i in range(N):
                cfg_opt.append([x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]])
            
            # Verify validity
            cx_arr = np.array([p[0] for p in cfg_opt])
            cy_arr = np.array([p[1] for p in cfg_opt])
            angle_arr = np.array([p[2] for p in cfg_opt])
            
            if _is_valid(cx_arr, cy_arr, angle_arr):
                best_cfg = cfg_opt
                
    # 4. Final Optimization of Outer Hexagon
    # We have best_cfg. Now find the best outer hexagon (center, angle).
    cx_arr = np.array([p[0] for p in best_cfg])
    cy_arr = np.array([p[1] for p in best_cfg])
    angle_arr = np.array([p[2] for p in best_cfg])
    
    # We can also try to optimize the inner config slightly to fit a specific outer angle?
    # For now, just find the best outer hexagon for the fixed inner config.
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    verts_list = [_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]) for i in range(N)]
    
    # Sweep outer angle
    for outer_angle_deg in range(0, 31, 1):
        a = math.radians(outer_angle_deg)
        apothem_unit = math.sqrt(3) / 2.0
        normals_out = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals_out.append((math.cos(na), math.sin(na)))
        
        # Optimize center
        # Initial guess: centroid
        all_verts = [v for verts in verts_list for v in verts]
        cx_c = sum(v[0] for v in all_verts) / len(all_verts)
        cy_c = sum(v[1] for v in all_verts) / len(all_verts)
        
        step = 0.05
        for _iter in range(30):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_list[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c))
                        if val > max_proj:
                            max_proj = val
            
            best_cx, best_cy, best_val = cx_c, cy_c, max_proj
            
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_list[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy))
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-5 and abs(best_cy - cy_c) < 1e-5:
                break
            cx_c, cy_c = best_cx, best_cy
            step *= 0.8
            
        side = max_proj / apothem_unit
        if side < best_side:
            best_side = side
            best_center = (cx_c, cy_c)
            best_angle = float(outer_angle_deg)
            
    # Final check
    if not _is_valid(cx_arr, cy_arr, angle_arr):
        # Fallback to honeycomb
        best_cfg = make_honeycomb()
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)
        best_center = (0.0, 0.0)
        best_angle = 0.0
        
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
