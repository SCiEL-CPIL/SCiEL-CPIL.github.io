import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21

def construct():
    start_time = time.time()
    best_score = 0
    best_sol = None
    
    # Helper to validate and score
    def validate_and_score(v):
        # v is [x1, y1, r1, ..., xN, yN, rN, W, H]
        n = N
        xs = v[0:n]
        ys = v[n:2*n]
        rs = v[2*n:3*n]
        W = v[3*n]
        H = v[3*n+1]
        
        # Check perimeter
        if 2 * (W + H) > 4 + 1e-6:
            return -1, None
            
        # Check bounds
        for i in range(n):
            if xs[i] < rs[i] - 1e-6 or xs[i] > W - rs[i] + 1e-6:
                return -1, None
            if ys[i] < rs[i] - 1e-6 or ys[i] > H - rs[i] + 1e-6:
                return -1, None
            if rs[i] <= 0:
                return -1, None
                
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                rad_sum = rs[i] + rs[j]
                if dist_sq < rad_sum*rad_sum - 1e-6:
                    return -1, None
                    
        score = np.sum(rs)
        return score, np.column_stack((xs, ys, rs))

    # Objective and constraints for SLSQP
    def objective(v):
        n = N
        rs = v[2*n:3*n]
        return -np.sum(rs)
        
    def constraints(v):
        n = N
        xs = v[0:n]
        ys = v[n:2*n]
        rs = v[2*n:3*n]
        W = v[3*n]
        H = v[3*n+1]
        
        cons = []
        # Perimeter
        cons.append({'type': 'ineq', 'fun': lambda v: 2 - (v[3*n] + v[3*n+1])})
        # Min size
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*n] - 0.1})
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*n+1] - 0.1})
        
        # Bounds
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]}) # x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*n] - v[i] - v[2*n+i]}) # W - x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]}) # y >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*n+1] - v[n+i] - v[2*n+i]}) # H - y >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n+i] - 0.01}) # r >= 0.01
            
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
                })
        return cons

    def hex_init():
        # Generate hexagonal grid
        w, h = 1.0, 1.0
        r_est = 0.15
        pts = []
        row = 0
        while len(pts) < N:
            y = (row + 0.5) * 2 * r_est * np.sqrt(3)
            if y > h - r_est:
                break
            offset = r_est if row % 2 else 0
            col = 0
            while len(pts) < N:
                x = (col + 0.5) * 2 * r_est + offset
                if x > w - r_est:
                    break
                pts.append((x, y))
                col += 1
            row += 1
        
        if len(pts) < N:
             pts = [(random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)) for _ in range(N)]
            
        pts = pts[:N]
        
        # Initial radii
        rs = []
        for i in range(N):
            min_d = min(pts[i][0], w-pts[i][0], pts[i][1], h-pts[i][1])
            for j in range(N):
                if i != j:
                    d = np.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
                    min_d = min(min_d, d/2)
            rs.append(min_d * 0.9)
            
        # Jitter
        v = []
        for i in range(N):
            v.extend([pts[i][0] + random.uniform(-0.01, 0.01), 
                      pts[i][1] + random.uniform(-0.01, 0.01), 
                      rs[i]])
        v.extend([1.0, 1.0])
        return np.array(v)

    # Main loop
    attempts = 0
    while time.time() - start_time < 110:
        v0 = hex_init()
        try:
            res = minimize(objective, v0, method='SLSQP', constraints=constraints, options={'maxiter': 100, 'ftol': 1e-4})
            if res.success:
                score, sol = validate_and_score(res.x)
                if score > best_score:
                    best_score = score
                    best_sol = sol
        except Exception:
            pass
        attempts += 1
        if attempts > 50: # Safety break if SLSQP is very fast
            break

    if best_sol is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_sol
