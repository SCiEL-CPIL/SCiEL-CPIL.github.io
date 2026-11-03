import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_to_cartesian(q, r):
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def generate_grid(w, h):
    points = []
    for r in range(h):
        for q in range(w):
            x, y = hex_to_cartesian(q, r)
            points.append((x, y))
    return points

def generate_compact_cluster(n, seed=0):
    np.random.seed(seed)
    lattice = set([(0,0)])
    current = [(0,0)]
    neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    while len(current) < n:
        q, r = current[np.random.randint(len(current))]
        dq, dr = neighbors[np.random.randint(6)]
        new_q, new_r = q + dq, r + dr
        if (new_q, new_r) not in lattice:
            lattice.add((new_q, new_r))
            current.append((new_q, new_r))
            
    points = []
    for q, r in current:
        x, y = hex_to_cartesian(q, r)
        points.append((x, y))
    return points

def calc_S(X, Y, alpha_deg, points):
    alpha = alpha_deg * np.pi / 180.0
    max_val = 0.0
    for deg in range(360):
        theta = deg * np.pi / 180.0
        delta = (theta - alpha) % (np.pi / 3.0)
        if delta > np.pi / 6.0:
            delta = np.pi / 3.0 - delta
        cos_d = np.cos(delta)
        if cos_d < 1e-9:
            continue
        delta_in = theta % (np.pi / 3.0)
        if delta_in > np.pi / 6.0:
            delta_in = np.pi / 3.0 - delta_in
        cos_d_in = np.cos(delta_in)
        if cos_d_in < 1e-9:
            continue
        h_unit = (np.sqrt(3) / 2.0) / cos_d_in
        
        h_c = 0.0
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        for px, py in points:
            val = (px - X) * c_theta + (py - Y) * s_theta
            if val > h_c:
                h_c = val
                
        s_candidate = (h_c + h_unit) * (2.0 / np.sqrt(3.0)) * cos_d
        if s_candidate > max_val:
            max_val = s_candidate
            
    return max_val

def optimize_cluster(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    X0 = np.mean(xs)
    Y0 = np.mean(ys)
    alpha0 = 0.0
    
    def objective(params):
        X, Y, alpha_deg = params
        return calc_S(X, Y, alpha_deg, points)
        
    res = minimize(objective, [X0, Y0, alpha0], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    X, Y, alpha_deg = res.x
    S = res.fun
    return X, Y, S, alpha_deg

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_S = float('inf')
    best_result = None
    
    candidates = []
    candidates.append(generate_grid(3, 4))
    candidates.append(generate_grid(4, 3))
    for i in range(5):
        candidates.append(generate_compact_cluster(12, seed=i))
        
    for points in candidates:
        if time.time() > deadline:
            break
        try:
            X, Y, S, alpha = optimize_cluster(points)
            if S < best_S:
                best_S = S
                best_result = (points, X, Y, S, alpha)
        except Exception:
            pass

    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    points, X, Y, S, alpha = best_result
    
    inner = []
    for px, py in points:
        inner.append([px, py, 0.0])
        
    return inner, (X, Y), S, alpha
