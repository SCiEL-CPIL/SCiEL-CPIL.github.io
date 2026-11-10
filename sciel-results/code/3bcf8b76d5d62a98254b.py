import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 600.0
    
    def is_valid(v, tol=1e-7):
        cx = v[:n]
        cy = v[n:2*n]
        s = v[2*n:]
        
        if np.any(s <= 0):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
            
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < (s[i] + s[j])/2 - tol and dy < (s[i] + s[j])/2 - tol:
                    return False
        return True

    def objective(v):
        cx = v[:n]
        cy = v[n:2*n]
        s = v[2*n:]
        
        pen = 0.0
        # Boundary
        for i in range(n):
            if cx[i] - s[i]/2 < 0:
                pen += (cx[i] - s[i]/2)**2
            if cx[i] + s[i]/2 > 1:
                pen += (cx[i] + s[i]/2 - 1)**2
            if cy[i] - s[i]/2 < 0:
                pen += (cy[i] - s[i]/2)**2
            if cy[i] + s[i]/2 > 1:
                pen += (cy[i] + s[i]/2 - 1)**2
        
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j]) - (s[i] + s[j])/2
                dy = abs(cy[i] - cy[j]) - (s[i] + s[j])/2
                if dx < 0 and dy < 0:
                    pen += min(-dx, -dy)**2
        
        return -np.sum(s) + 100.0 * pen

    best_v = None
    best_val = 0.0
    
    def callback(xk):
        nonlocal best_v, best_val
        if is_valid(xk):
            val = np.sum(xk[2*n:])
            if val > best_val:
                best_val = val
                best_v = xk.copy()

    # Initial configurations
    configs = []
    
    # 3-3-4
    v1 = np.zeros(3*n)
    # Col 1: 3 squares, side 1/3
    for i in range(3):
        v1[i] = 1/6 # cx
        v1[n+i] = (i+0.5)/3 # cy
        v1[2*n+i] = 1/3 # s
    # Col 2: 3 squares, side 1/3
    for i in range(3):
        v1[3+i] = 1/3 + 1/6
        v1[n+3+i] = (i+0.5)/3
        v1[2*n+3+i] = 1/3
    # Col 3: 4 squares, side 1/4
    for i in range(4):
        v1[6+i] = 2/3 + 1/8
        v1[n+6+i] = (i+0.5)/4
        v1[2*n+6+i] = 1/4
    configs.append(v1)
    
    # 2-4-4
    v2 = np.zeros(3*n)
    for i in range(2):
        v2[i] = 1/4
        v2[n+i] = (i+0.5)/2
        v2[2*n+i] = 1/2
    for i in range(4):
        v2[2+i] = 1/2 + 1/8
        v2[n+2+i] = (i+0.5)/4
        v2[2*n+2+i] = 1/4
    for i in range(4):
        v2[6+i] = 3/4 + 1/8
        v2[n+6+i] = (i+0.5)/4
        v2[2*n+6+i] = 1/4
    configs.append(v2)
    
    # Random
    np.random.seed(42)
    for _ in range(5):
        s_rand = np.random.uniform(0.1, 0.3, n)
        cx_rand = np.random.uniform(s_rand/2, 1 - s_rand/2, n)
        cy_rand = np.random.uniform(s_rand/2, 1 - s_rand/2, n)
        v_rand = np.concatenate([cx_rand, cy_rand, s_rand])
        configs.append(v_rand)

    for v0 in configs:
        if time.time() - start_time > time_limit - 10:
            break
            
        res = minimize(objective, v0, method='Powell', options={'maxiter': 500, 'callback': callback, 'xatol': 1e-8, 'fatol': 1e-8})
        
        # Check if res.x is better and valid
        if is_valid(res.x):
            val = np.sum(res.x[2*n:])
            if val > best_val:
                best_val = val
                best_v = res.x.copy()

    if best_v is None:
        # Fallback to first config if no valid found (should not happen)
        best_v = configs[0]
        
    cx = best_v[:n]
    cy = best_v[n:2*n]
    s = best_v[2*n:]
    
    centers = np.stack([cx, cy], axis=1)
    return centers, s
