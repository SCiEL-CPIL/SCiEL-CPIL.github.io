import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hexagon_vertices(cx, cy, side, angle_rad):
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def signed_dist_point_to_convex_poly(point, poly):
    min_sd = np.inf
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        edge = v2 - v1
        normal = np.array([edge[1], -edge[0]])
        nn = np.linalg.norm(normal)
        if nn < 1e-12:
            continue
        normal = normal / nn
        sd = np.dot(point - v1, normal)
        if sd < min_sd:
            min_sd = sd
    return min_sd

def min_signed_dist_poly_to_poly(polyA, polyB):
    min_sd = np.inf
    for p in polyA:
        sd = signed_dist_point_to_convex_poly(p, polyB)
        if sd < min_sd:
            min_sd = sd
    return min_sd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def make_initial(spread):
        positions = [(0.0, 0.0, 0.0)]
        for k in range(6):
            angle = k * np.pi / 3
            positions.append((spread * np.cos(angle), spread * np.sin(angle), 0.0))
        for k in range(4):
            angle = k * np.pi / 2 + np.pi / 6
            positions.append((spread * 1.9 * np.cos(angle), spread * 1.9 * np.sin(angle), 0.0))
        x0 = []
        for (x, y, a) in positions:
            x0.extend([x, y, np.deg2rad(a)])
        x0.append(4.5)
        return np.array(x0)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        xs = params[:3*N].reshape(N, 3)
        s = max(params[3*N], 0.5)
        
        inner_polys = [hexagon_vertices(xs[i,0], xs[i,1], 1.0, xs[i,2]) for i in range(N)]
        outer_poly = hexagon_vertices(0.0, 0.0, s, 0.0)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sd1 = min_signed_dist_poly_to_poly(inner_polys[i], inner_polys[j])
                if sd1 < 0:
                    penalty += (-sd1) ** 2
                sd2 = min_signed_dist_poly_to_poly(inner_polys[j], inner_polys[i])
                if sd2 < 0:
                    penalty += (-sd2) ** 2
        
        for i in range(N):
            for v in inner_polys[i]:
                sd = signed_dist_point_to_convex_poly(v, outer_poly)
                if sd < 0:
                    penalty += (-sd) ** 2
        
        return s + 5.0 * penalty
    
    best_result = None
    best_s = np.inf
    
    x0 = make_initial(2.0)
    
    for method in ['L-BFGS-B', 'Nelder-Mead']:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(objective, x0, method=method, 
                              options={'maxiter': 2000} if method == 'L-BFGS-B' else {'maxiter': 2000, 'maxfev': 20000})
            s = max(result.x[3*N], 0.5)
            if s < best_s and result.fun < 1e6:
                best_s = s
                xs = result.x[:3*N].reshape(N, 3)
                inner = [[float(xs[i,0]), float(xs[i,1]), float(np.rad2deg(xs[i,2]))] for i in range(N)]
                best_result = (inner, (0.0, 0.0), float(s), 0.0)
        except Exception:
            pass
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
