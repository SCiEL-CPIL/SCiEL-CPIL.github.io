import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
DEADLINE = 140.0

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(a)
        y = cy + side * math.sin(a)
        verts.append((x, y))
    return np.array(verts)

def point_in_hexagon_slack(p, cx, cy, angle_deg, side=1.0):
    """
    Returns the slack (distance to boundary) if p is inside.
    Returns -inf if outside? No, we want to know if it's inside.
    Let's return the minimum slack (positive if inside, negative if outside).
    """
    dx = p[0] - cx
    dy = p[1] - cy
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    x_local = dx * cos_a - dy * sin_a
    y_local = dx * sin_a + dy * cos_a
    
    normals = np.array([
        [1, 0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1, 0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    dists = normals @ np.array([x_local, y_local])
    max_dist = side * math.sqrt(3) / 2
    # Slack is max_dist - dist. If all slacks > 0, it's inside.
    # Min slack is the distance to the closest boundary.
    min_slack = max_dist - np.max(dists)
    return min_slack

def penalty(params, L):
    centers = np.array(params[0:2*N]).reshape(N, 2)
    angles = np.array(params[2*N:3*N])
    
    normals = np.array([
        [1, 0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1, 0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    max_dist_outer = L * math.sqrt(3) / 2
    
    pen = 0.0
    
    all_verts = []
    for i in range(N):
        verts = hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        all_verts.append(verts)
        
        # Containment
        for v in verts:
            dists = normals @ v
            violation = np.max(dists) - max_dist_outer
            if violation > 0:
                pen += violation * violation
    
    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            
            if d < math.sqrt(3):
                # Guaranteed overlap
                pen += (math.sqrt(3) - d) * (math.sqrt(3) - d)
            elif d < 2.0:
                # Possible overlap, check vertices
                # Check if any vertex of i is inside j
                for v in all_verts[i]:
                    slack = point_in_hexagon_slack(v, centers[j, 0], centers[j, 1], angles[j])
                    if slack > 0:
                        pen += slack * slack
                        
                # Check if any vertex of j is inside i
                for v in all_verts[j]:
                    slack = point_in_hexagon_slack(v, centers[i, 0], centers[i, 1], angles[i])
                    if slack > 0:
                        pen += slack * slack
                        
    return pen

def construct():
    start_time = time.time()
    best_L = 40.0
    best_config = None
    
    # Initial honeycomb
    points = []
    points.append((0, 0, 0))
    for i in range(6):
        a = i * 60
        r = math.sqrt(3)
        points.append((r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), 0))
    for i in range(5):
        a = i * 60
        r = 2 * math.sqrt(3)
        points.append((r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), 0))
        
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_r = max(math.sqrt(x*x + y*y) for x, y in zip(xs, ys))
    
    def try_L(L, init_params=None):
        if init_params is None:
            scale = (L * math.sqrt(3)/2 * 0.9) / max_r if max_r > 0 else 1.0
            init_params = []
            for p in points:
                init_params.extend([p[0]*scale, p[1]*scale, p[2]])
        else:
            init_params = list(init_params)
            
        res = minimize(penalty, init_params, args=(L,), method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        return res.fun, res.x

    # Try L=6.0
    pen, config = try_L(6.0)
    if pen < 1e-4:
        best_L = 6.0
        best_config = config
    else:
        pen, config = try_L(7.0)
        if pen < 1e-4:
            best_L = 7.0
            best_config = config
        else:
            return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Try smaller
    target_Ls = [5.8, 5.6, 5.4, 5.2, 5.0, 4.95, 4.9, 4.85, 4.8]
    
    current_config = best_config
    for L in target_Ls:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        centers = np.array(current_config[0:2*N]).reshape(N, 2)
        max_r = np.max(np.linalg.norm(centers, axis=1))
        target_r = L * math.sqrt(3)/2 * 0.9
        if max_r > 0:
            scale = target_r / max_r
            new_config = list(current_config)
            for i in range(N):
                new_config[2*i] = current_config[2*i] * scale
                new_config[2*i+1] = current_config[2*i+1] * scale
        else:
            new_config = list(current_config)
            
        pen, new_config_opt = try_L(L, new_config)
        
        if pen < 1e-4:
            best_L = L
            best_config = new_config_opt
        else:
            # Try random restart
            if time.time() - start_time < DEADLINE - 30:
                random_config = []
                for _ in range(N):
                    random_config.extend([np.random.uniform(-L, L), np.random.uniform(-L, L), np.random.uniform(0, 60)])
                pen_r, config_r = try_L(L, random_config)
                if pen_r < 1e-4:
                    best_L = L
                    best_config = config_r
            break

    if best_config is not None:
        res = minimize(penalty, best_config, args=(best_L,), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
        if res.fun < 1e-6:
            best_config = res.x

    inner = []
    for i in range(N):
        x = best_config[2*i]
        y = best_config[2*i+1]
        a = best_config[2*N+i]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), best_L, 0.0
