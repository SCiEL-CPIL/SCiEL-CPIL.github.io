"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12
DEADLINE = 140.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
def get_hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def min_enclosing_hexagon_side(vertices, center_x, center_y, outer_angle_deg):
    min_side = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (x, y) in vertices:
        dx = x - center_x
        dy = y - center_y
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        val1 = abs(v)
        val2 = abs(math.sqrt(3)/2 * u + 1/2 * v)
        val3 = abs(math.sqrt(3)/2 * u - 1/2 * v)
        
        m = max(val1, val2, val3)
        s_req = (2.0 / math.sqrt(3.0)) * m
        
        if s_req > min_side:
            min_side = s_req
            
    return min_side

def point_in_hexagon(px, py, cx, cy, ang):
    cos_a = math.cos(math.radians(ang))
    sin_a = math.sin(math.radians(ang))
    dx = px - cx
    dy = py - cy
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a
    
    val1 = abs(v)
    val2 = abs(math.sqrt(3)/2 * u + 1/2 * v)
    val3 = abs(math.sqrt(3)/2 * u - 1/2 * v)
    
    m = max(val1, val2, val3)
    s_req = (2.0 / math.sqrt(3.0)) * m
    return s_req <= 1.0 + 1e-9

def segments_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
        if val > 1e-10: return 1
        if val < -1e-10: return -1
        return 0
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    return False

def hexagons_disjoint(h1, h2):
    v1 = get_hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = get_hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
    
    for p in v1:
        if point_in_hexagon(p[0], p[1], h2[0], h2[1], h2[2]):
            return False
    for p in v2:
        if point_in_hexagon(p[0], p[1], h1[0], h1[1], h1[2]):
            return False
        
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def objective(params):
    inner = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner.append((cx, cy, ang))
    
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_ang = params[3*N+2]
    
    all_verts = []
    for (cx, cy, ang) in inner:
        verts = get_hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts.tolist())
    
    side = min_enclosing_hexagon_side(all_verts, outer_cx, outer_cy, outer_ang)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = math.sqrt(3.0)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2 * 100.0
            else:
                if dist < 2.5:
                    if not hexagons_disjoint(inner[i], inner[j]):
                        penalty += 1000.0
    
    return side + penalty

def construct():
    np.random.seed(42)
    
    init_params = np.zeros(3*N + 3)
    for i in range(N):
        ang = 2 * math.pi * i / N
        r = 1.5
        init_params[3*i] = r * math.cos(ang)
        init_params[3*i+1] = r * math.sin(ang)
        init_params[3*i+2] = 0.0
    init_params[3*N] = 0.0
    init_params[3*N+1] = 0.0
    init_params[3*N+2] = 0.0
    
    best_val = float('inf')
    best_params = init_params.copy()
    
    if time.time() - START_TIME > DEADLINE:
        inner = [[init_params[3*i], init_params[3*i+1], init_params[3*i+2]] for i in range(N)]
        all_verts = []
        for (cx, cy, ang) in inner:
            verts = get_hexagon_vertices(cx, cy, 1.0, ang)
            all_verts.extend(verts.tolist())
        final_side = min_enclosing_hexagon_side(all_verts, init_params[3*N], init_params[3*N+1], init_params[3*N+2])
        return inner, (init_params[3*N], init_params[3*N+1]), final_side, init_params[3*N+2]
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    except Exception as e:
        pass
    
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N+1]
    outer_ang = best_params[3*N+2]
    
    all_verts = []
    for (cx, cy, ang) in inner:
        verts = get_hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts.tolist())
    
    final_side = min_enclosing_hexagon_side(all_verts, outer_cx, outer_cy, outer_ang)
    
    return inner, (outer_cx, outer_cy), final_side, outer_ang
# EVOLVE-BLOCK-END
