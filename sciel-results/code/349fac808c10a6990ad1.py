import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def make_constraints(W):
        H = 2.0 - W
        cons = []
        # x bounds
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i], 'jac': lambda v, i=i: np.concatenate([np.eye(N)[i], np.zeros(N), np.eye(N)[i] * (-1), np.zeros(1)])})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: W - v[2*N+i] - v[i], 'jac': lambda v, i=i: np.concatenate([np.eye(N)[i] * (-1), np.zeros(N), np.eye(N)[i], np.zeros(1)])})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i], 'jac': lambda v, i=i: np.concatenate([np.zeros(N), np.eye(N)[i], np.eye(N)[i] * (-1), np.zeros(1)])})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[2*N+i] - v[N+i], 'jac': lambda v, i=i: np.concatenate([np.zeros(N), np.eye(N)[i] * (-1), np.eye(N)[i], np.zeros(1)])})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i], 'jac': lambda v, i=i: np.concatenate([np.zeros(N), np.zeros(N), np.eye(N)[i], np.zeros(1)])})
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[N+i]-v[N+j])**2 - (v[2*N+i]+v[2*N+j])**2})
        return cons
    
    # This is too many constraints for SLSQP to handle efficiently.
    # Let me try a different approach.
    
    # Actually, let me use a simpler method: for a fixed rectangle, use a force-directed or gradient method to pack circles.
    
    # Let me try: fix W=1, H=1 (square), and optimize circle positions and radii.
    
    best_value = -np.inf
    best_vars = None
    
    # Try several rectangle aspect ratios
    for W in [0.8, 0.9, 1.0, 1.1, 1.2]:
        H = 2.0 - W
        if H < 0.3:
            continue
        
        # Initial guess: grid
        m = max(1, int(np.round(np.sqrt(N * W / H))))
        n = max(1, int(np.ceil(N / m)))
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for idx in range(N):
            i = idx % m
            j = idx // m
            x0[idx] = (i + 0.5) * W / m
            y0[idx] = (j + 0.5) * H / n
            r0[idx] = min(W/m, H/n) * 0.4
        
        x0 = np.clip(x0, 0.01, W - 0.01)
        y0 = np.clip(y0, 0.01, H - 0.01)
        r0 = np.clip(r0, 0.01, min(W, H) * 0.3)
        
        v0 = np.concatenate([x0, y0, r0])
        
        # Objective
        def obj(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            # Penalty
            pen = 0.0
            for i in range(N):
                if x[i] < r[i]: pen += 1e4 * (r[i] - x[i])**2
                if x[i] > W - r[i]: pen += 1e4 * (x[i] - W + r[i])**2
                if y[i] < r[i]: pen += 1e4 * (r[i] - y[i])**2
                if y[i] > H - r[i]: pen += 1e4 * (y[i] - H + r[i])**2
                if r[i] < 0: pen += 1e4 * r[i]**2
            for i in range(N):
                for j in range(i+1, N):
                    d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    d = np.sqrt(max(d2, 1e-12))
                    if d < r[i] + r[j]:
                        pen += 1e4 * (r[i] + r[j] - d)**2
            return -np.sum(r) + pen
        
        # This is O(N^2) per evaluation, and SLSQP will do many evaluations.
        # 21 circles -> 210 pairs. Might be okay.
        
        try:
            res = minimize(obj, v0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-8})
            if res.success:
                val = np.sum(res.x[2*N:3*N])
                if val > best_value:
                    best_value = val
                    best_vars = res.x.copy()
        except:
            pass
    
    if best_vars is not None:
        x = best_vars[:N]
        y = best_vars[N:2*N]
        r = best_vars[2*N:3*N]
        # Polish: try to increase radii
        for _ in range(100):
            improved = False
            for i in range(N):
                # Try to increase r[i] slightly
                dr = 0.001
                new_r = r[i] + dr
                # Check constraints
                ok = True
                if x[i] < new_r or x[i] > W - new_r:
                    ok = False
                if y[i] < new_r or y[i] > H - new_r:
                    ok = False
                if ok:
                    for j in range(N):
                        if j == i:
                            continue
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < new_r + r[j]:
                            ok = False
                            break
                if ok:
                    r[i] = new_r
                    improved = True
            if not improved:
                break
        
        result = np.column_stack([x, y, r])
        # Verify perimeter
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        perim = 2 * ((max_x - min_x) + (max_y - min_y))
        if perim <= 4.0 + 1e-6:
            return result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
