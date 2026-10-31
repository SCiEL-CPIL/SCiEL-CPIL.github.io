import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 110.0

    # Precompute indices for upper triangle
    iu = np.triu_indices(N, 1)
    
    # Initial state
    # Grid 5x5 minus corners
    coords = []
    for i in range(5):
        for j in range(5):
            if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                continue
            coords.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0])
            
    x = np.array([c[0] for c in coords])
    y = np.array([c[1] for c in coords])
    r = np.full(N, 0.1)
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_score = -np.inf
    
    T = 1.0
    T_min = 1e-4
    move_size = 0.05
    
    def get_cost(x, y, r):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx*dx + dy*dy)
        d_ij = d[iu]
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(0, r_sum - d_ij)
        p_ov = np.sum(overlap**2)
        
        W = np.max(x+r) - np.min(x-r)
        H = np.max(y+r) - np.min(y-r)
        P = 2*(W+H)
        p_pe = max(0, P-4)**2
        
        sc = np.sum(r)
        cost = p_ov + 10.0 * p_pe - sc
        return cost, p_ov, p_pe, sc

    cost, p_ov, p_pe, sc = get_cost(x, y, r)
    if p_ov < 1e-6 and p_pe < 1e-6:
        best_score = sc
        best_x = x.copy()
        best_y = y.copy()
        best_r = r.copy()
        
    while time.time() - start_time < time_limit:
        # Move
        dx_m = np.random.normal(0, move_size, N)
        dy_m = np.random.normal(0, move_size, N)
        dr_m = np.random.normal(0, move_size * 0.5, N)
        
        new_x = x + dx_m
        new_y = y + dy_m
        new_r = np.maximum(0.01, r + dr_m)
        
        new_cost, new_p_ov, new_p_pe, new_sc = get_cost(new_x, new_y, new_r)
        delta = new_cost - cost
        
        if delta < 0:
            accept = True
        else:
            if T > 1e-8:
                accept = np.random.rand() < np.exp(-delta / T)
            else:
                accept = False
                
        if accept:
            x, y, r = new_x, new_y, new_r
            cost, p_ov, p_pe, sc = new_cost, new_p_ov, new_p_pe, new_sc
            
            if p_ov < 1e-4 and p_pe < 1e-4:
                if sc > best_score:
                    best_score = sc
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
                    
        T *= 0.999
        move_size *= 0.999
        if T < T_min:
            break
            
    # Repair
    final_x, final_y, final_r = best_x, best_y, best_r
    
    def is_valid(x, y, r):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx*dx + dy*dy)
        d_ij = d[iu]
        r_sum = r[iu[0]] + r[iu[1]]
        if np.any(r_sum - d_ij > 1e-8):
            return False
        W = np.max(x+r) - np.min(x-r)
        H = np.max(y+r) - np.min(y-r)
        if 2*(W+H) > 4.0 + 1e-8:
            return False
        return True

    if not is_valid(final_x, final_y, final_r):
        scale = 1.0
        while not is_valid(final_x, final_y, final_r * scale):
            scale *= 0.95
            if scale < 1e-4:
                break
        final_r = final_r * scale

    return np.column_stack((final_x, final_y, final_r))
