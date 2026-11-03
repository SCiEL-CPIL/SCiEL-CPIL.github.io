import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def get_hex_verts_local(angle_deg=0):
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            verts.append((math.cos(theta), math.sin(theta)))
        return np.array(verts)

    def min_side_for_point(x, y):
        max_val = 0
        for k in range(6):
            theta = k * math.pi / 3
            val = abs(x * math.cos(theta) + y * math.sin(theta))
            if val > max_val:
                max_val = val
        return (2.0 / math.sqrt(3.0)) * max_val

    def get_hex_verts_global(cx, cy, angle_deg):
        local_verts = get_hex_verts_local(angle_deg)
        return local_verts + np.array([cx, cy])

    def min_dist_poly(verts1, verts2):
        min_d = float('inf')
        n1 = len(verts1)
        n2 = len(verts2)
        
        for v1 in verts1:
            for v2 in verts2:
                d = np.linalg.norm(v1 - v2)
                if d < min_d:
                    min_d = d
                    
        for i in range(n1):
            v1 = verts1[i]
            e1_start = verts1[(i-1)%n1]
            e1_end = verts1[(i+1)%n1]
            for v2 in verts2:
                v = e1_end - e1_start
                L2 = np.dot(v, v)
                if L2 == 0:
                    d = np.linalg.norm(v2 - e1_start)
                else:
                    t = np.dot(v2 - e1_start, v) / L2
                    t = max(0, min(1, t))
                    closest = e1_start + t * v
                    d = np.linalg.norm(v2 - closest)
                if d < min_d:
                    min_d = d
                    
        for i in range(n2):
            v2 = verts2[i]
            e2_start = verts2[(i-1)%n2]
            e2_end = verts2[(i+1)%n2]
            for v1 in verts1:
                v = e2_end - e2_start
                L2 = np.dot(v, v)
                if L2 == 0:
                    d = np.linalg.norm(v1 - e2_start)
                else:
                    t = np.dot(v1 - e2_start, v) / L2
                    t = max(0, min(1, t))
                    closest = e2_start + t * v
                    d = np.linalg.norm(v1 - closest)
                if d < min_d:
                    min_d = d
        return min_d

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        theta_in = params[2*N]
        
        s_req_max = 0
        verts_list = []
        
        for i in range(N):
            cx, cy = positions[i]
            verts = get_hex_verts_global(cx, cy, theta_in)
            verts_list.append(verts)
            
            s_req_i = 0
            for v in verts:
                s_req_v = min_side_for_point(v[0], v[1])
                if s_req_v > s_req_i:
                    s_req_i = s_req_v
            if s_req_i > s_req_max:
                s_req_max = s_req_i
                
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_poly(verts_list[i], verts_list[j])
                if d < 0.05:
                    penalty += (0.05 - d)**2
        
        return s_req_max + 100.0 * penalty

    u = np.array([math.sqrt(3), 0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    coords = [
        (0,0),
        (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    init_positions = []
    for q, r in coords:
        pos = q * u + r * v
        init_positions.append([pos[0], pos[1]])
        
    init_params = np.array(init_positions).flatten()
    init_params = np.append(init_params, 0.0)
    
    best_result = init_params
    best_score = objective(init_params)
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
    except Exception:
        pass

    # Check time for potential restart
    if time.time() < deadline - 20:
        # Try one random restart
        try:
            random_params = init_params + np.random.normal(0, 0.5, len(init_params))
            res2 = minimize(objective, random_params, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception:
            pass

    final_positions = best_result[:2*N].reshape(N, 2)
    final_theta_in = best_result[2*N]
    
    s_final = 0
    for i in range(N):
        cx, cy = final_positions[i]
        verts = get_hex_verts_global(cx, cy, final_theta_in)
        for v in verts:
            s_req = min_side_for_point(v[0], v[1])
            if s_req > s_final:
                s_final = s_req
                
    inner = [[final_positions[i][0], final_positions[i][1], final_theta_in] for i in range(N)]
    center = (0.0, 0.0)
    side = s_final
    angle = 0.0
    
    return inner, center, side, angle
