import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s buffer
    
    # Helper to extract x, y, r from flat vector
    def unpack(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r

    # Helper to pack x, y, r into flat vector
    def pack(x, y, r):
        return np.concatenate([x, y, r])

    # Objective function
    def objective(v, lam_overlap, lam_bound):
        x, y, r = unpack(v)
        # Ensure r is positive for calculation stability, though penalty handles it
        # We want to maximize sum(r), so minimize -sum(r)
        obj = -np.sum(r)
        
        # Boundary penalty
        # x in [r, 1-r]
        # penalty if x < r or x > 1-r
        b_pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                b_pen += (r[i] - x[i])**2
            if x[i] > 1 - r[i]:
                b_pen += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                b_pen += (r[i] - y[i])**2
            if y[i] > 1 - r[i]:
                b_pen += (y[i] - (1 - r[i]))**2
        
        # Overlap penalty
        o_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                req = r[i] + r[j]
                if dist < req:
                    o_pen += (req - dist)**2
        
        return obj + lam_bound * b_pen + lam_overlap * o_pen

    # Validity check
    def is_valid(v, eps=1e-6):
        x, y, r = unpack(v)
        if np.any(r <= 0): return False
        if np.any(x < r - eps) or np.any(x > 1 - r + eps): return False
        if np.any(y < r - eps) or np.any(y > 1 - r + eps): return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - eps:
                    return False
        return True

    def score(v):
        if not is_valid(v):
            return -np.inf
        x, y, r = unpack(v)
        return np.sum(r)

    best_score = -np.inf
    best_v = None

    # Restart loop
    num_restarts = 5
    
    for k in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Initialization: Grid + Jitter
        side = 6
        xs = []
        ys = []
        rs = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx < N:
                    # Center of cell
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    # Add jitter
                    cx += np.random.uniform(-0.01, 0.01)
                    cy += np.random.uniform(-0.01, 0.01)
                    # Initial radius slightly less than half cell width
                    r0 = 0.5 / side - 0.01
                    xs.append(cx)
                    ys.append(cy)
                    rs.append(r0)
                    idx += 1
        
        v0 = np.array(xs + ys + rs)
        
        # Optimization sequence
        v_current = v0
        lams = [10, 100, 1000, 5000]
        
        for lam in lams:
            if time.time() - start_time > deadline:
                break
                
            # Use Powell method
            # Bounds: x,y in [0,1], r in [0, 0.5]
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            
            res = minimize(
                objective, 
                v_current, 
                method='Powell', 
                bounds=bounds, 
                args=(lam, 100.0), # lam_overlap, lam_bound
                options={'maxiter': 200, 'maxfev': 2000}
            )
            
            if res.success or res.fun < objective(v_current, lam, 100.0):
                v_current = res.x
            
        # Check final result
        s = score(v_current)
        if s > best_score:
            best_score = s
            best_v = v_current.copy()
            
    if best_v is None:
        # Fallback: trivial grid
        side = 6
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([
                        (i + 0.5) / side, 
                        (j + 0.5) / side, 
                        0.5 / side - 1e-3
                    ])
        return np.array(cells)
    
    x, y, r = unpack(best_v)
    return np.column_stack((x, y, r))
