import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, r, angle):
    angles = np.radians(angle + np.arange(6) * 60)
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))

def project_polygon(vertices, axis):
    proj = np.dot(vertices, axis)
    return np.min(proj), np.max(proj)

def sat_overlap(v1, v2):
    max_pen = 0.0
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm == 0: continue
        normal /= norm
        
        min1, max1 = project_polygon(v1, normal)
        min2, max2 = project_polygon(v2, normal)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        else:
            pen = min(max1, max2) - max(min1, min2)
            if pen < max_pen:
                max_pen = pen
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm == 0: continue
        normal /= norm
        
        min1, max1 = project_polygon(v1, normal)
        min2, max2 = project_polygon(v2, normal)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        else:
            pen = min(max1, max2) - max(min1, min2)
            if pen < max_pen:
                max_pen = pen
    return max_pen

def point_in_hexagon(px, py, cx, cy, side, angle):
    dx = px - cx
    dy = py - cy
    rad = np.radians(-angle)
    lx = dx * np.cos(rad) - dy * np.sin(rad)
    ly = dx * np.sin(rad) + dy * np.cos(rad)
    
    a = side * np.sqrt(3) / 2
    max_violation = 0.0
    for k in range(6):
        theta = np.radians(30 + k * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        val = lx * nx + ly * ny
        if val > a:
            violation = val - a
            if violation > max_violation:
                max_violation = violation
    return max_violation

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_L = 40.0
    best_params = None
    
    # Define objective function factory
    def make_objective():
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            x = np.zeros(N)
            y = np.zeros(N)
            t = np.zeros(N)
            
            x[0] = 0.0
            y[0] = 0.0
            t[0] = 0.0
            
            x[1:] = params[0:10]
            y[1:] = params[10:20]
            t[1:] = params[20:30]
            theta_c = params[30]
            L = params[31]
            
            if L <= 0:
                return 1e10
                
            verts = []
            for i in range(N):
                v = hex_vertices(x[i], y[i], 1.0, t[i])
                verts.append(v)
                
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    pen = sat_overlap(verts[i], verts[j])
                    if pen > overlap_pen:
                        overlap_pen = pen
                        
            boundary_pen = 0.0
            for i in range(N):
                for px, py in verts[i]:
                    viol = point_in_hexagon(px, py, 0.0, 0.0, L, theta_c)
                    if viol > boundary_pen:
                        boundary_pen = viol
                        
            cost = L + 10.0 * overlap_pen + 10.0 * boundary_pen
            return cost
        return objective

    objective = make_objective()
    
    # Multi-start
    # Start 1: Lattice
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    t0 = np.zeros(N)
    dist = np.sqrt(3)
    for i in range(1, 7):
        angle = np.radians(i * 60)
        x0[i] = dist * np.cos(angle)
        y0[i] = dist * np.sin(angle)
    for i in range(7, 11):
        angle = np.radians((i - 6) * 60)
        x0[i] = 2 * dist * np.cos(angle)
        y0[i] = 2 * dist * np.sin(angle)
    init1 = np.concatenate([x0[1:], y0[1:], t0[1:], [0.0, 5.0]])
    
    # Start 2: Random
    np.random.seed(42)
    x1 = np.random.uniform(-2, 2, 10)
    y1 = np.random.uniform(-2, 2, 10)
    t1 = np.random.uniform(0, 360, 10)
    init2 = np.concatenate([x1, y1, t1, [0.0, 4.0]])
    
    # Start 3: Random tighter
    x2 = np.random.uniform(-1.5, 1.5, 10)
    y2 = np.random.uniform(-1.5, 1.5, 10)
    t2 = np.random.uniform(0, 360, 10)
    init3 = np.concatenate([x2, y2, t2, [0.0, 3.5]])
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(objective, init, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_L:
                p = result.x
                x = np.zeros(N)
                y = np.zeros(N)
                t = np.zeros(N)
                x[0] = 0.0
                y[0] = 0.0
                t[0] = 0.0
                x[1:] = p[0:10]
                y[1:] = p[10:20]
                t[1:] = p[20:30]
                theta_c = p[30]
                L = p[31]
                
                # Verify
                verts = [hex_vertices(x[i], y[i], 1.0, t[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap(verts[i], verts[j]) > 1e-5:
                            valid = False
                    for px, py in verts[i]:
                        if point_in_hexagon(px, py, 0.0, 0.0, L, theta_c) > 1e-5:
                            valid = False
                if valid:
                    best_L = L
                    best_params = (x, y, t, theta_c, L)
        except Exception:
            pass
            
    if best_params is not None:
        x, y, t, theta_c, L = best_params
        inner = [[x[i], y[i], t[i] * 180.0 / np.pi] for i in range(N)]
        return inner, (0.0, 0.0), L, theta_c * 180.0 / np.pi
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

