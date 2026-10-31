import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # Safety margin

def construct():
    best_sum = 0.0
    best_solution = None
    
    def time_ok():
        return time.time() < DEADLINE

    # Variables: [W, x1, y1, r1, ..., xN, yN, rN]
    # W is width, H = 2 - W
    # We want to maximize sum(r), so minimize -sum(r)
    
    def objective(v):
        W = v[0]
        H = 2.0 - W
        if W <= 0 or H <= 0:
            return 1e10
        rs = v[2::3]
        return -np.sum(rs)

    # Pre-define constraints
    cons_list = []
    for i in range(N):
        # x >= r
        cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+3*i] - v[3+3*i]})
        # W - x >= r  =>  W - x - r >= 0
        cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+3*i] - v[3+3*i]})
        # y >= r
        cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[2+3*i] - v[3+3*i]})
        # H - y >= r  =>  (2-W) - y - r >= 0
        cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[0]) - v[2+3*i] - v[3+3*i]})
        # r >= 0.001
        cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3+3*i] - 0.001})
        
    for i in range(N):
        for j in range(i+1, N):
            cons_list.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[1+3*i]-v[1+3*j])**2 + (v[2+3*i]-v[2+3*j])**2 - (v[3+3*i]+v[3+3*j])**2
            })

    def run_optimization(v0, maxiter=300):
        try:
            bounds = [(0.5, 1.5)] + [(0, 2.0)]*N + [(0, 2.0)]*N + [(0.001, 1.0)]*N
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                           constraints=cons_list, options={'maxiter': maxiter, 'ftol': 1e-6, 'disp': False})
            if res.success:
                # Verify feasibility
                W = res.x[0]
                H = 2.0 - W
                valid = True
                for i in range(N):
                    xi, yi, ri = res.x[1+3*i], res.x[2+3*i], res.x[3+3*i]
                    if xi < ri - 1e-5 or W - xi < ri - 1e-5 or yi < ri - 1e-5 or H - yi < ri - 1e-5:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            xi, yi, ri = res.x[1+3*i], res.x[2+3*i], res.x[3+3*i]
                            xj, yj, rj = res.x[1+3*j], res.x[2+3*j], res.x[3+3*j]
                            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                            if dist < ri + rj - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    return -res.fun, res.x
                else:
                    return 0, None
            else:
                return 0, None
        except Exception:
            return 0, None

    # Initializations
    inits = []
    
    # 1. Grid inits for various aspect ratios
    for W in [0.8, 1.0, 1.2]:
        H = 2.0 - W
        # 3x7 grid
        kx, ky = 7, 3
        x_coords = np.linspace(0, W, kx+2)[1:-1]
        y_coords = np.linspace(0, H, ky+2)[1:-1]
        v0 = np.zeros(1 + 3*N)
        v0[0] = W
        idx = 1
        count = 0
        for y in y_coords:
            for x in x_coords:
                if count >= N: break
                r_est = min(x, W-x, y, H-y, W/kx/2, H/ky/2) * 0.9
                v0[idx] = x
                v0[idx+1] = y
                v0[idx+2] = r_est
                idx += 3
                count += 1
        inits.append(v0)
        
        # 4x6 grid (24 circles, take 21)
        kx, ky = 6, 4
        x_coords = np.linspace(0, W, kx+2)[1:-1]
        y_coords = np.linspace(0, H, ky+2)[1:-1]
        v0 = np.zeros(1 + 3*N)
        v0[0] = W
        idx = 1
        count = 0
        for y in y_coords:
            for x in x_coords:
                if count >= N: break
                r_est = min(x, W-x, y, H-y, W/kx/2, H/ky/2) * 0.9
                v0[idx] = x
                v0[idx+1] = y
                v0[idx+2] = r_est
                idx += 3
                count += 1
        inits.append(v0)

    # 2. Random inits
    np.random.seed(42)
    for _ in range(5):
        W = np.random.uniform(0.8, 1.2)
        H = 2.0 - W
        v0 = np.zeros(1 + 3*N)
        v0[0] = W
        for i in range(N):
            v0[1+3*i] = np.random.uniform(0.1, W-0.1)
            v0[2+3*i] = np.random.uniform(0.1, H-0.1)
            v0[3+3*i] = np.random.uniform(0.05, 0.2)
        inits.append(v0)
        
    # Run optimization on all inits
    for v0 in inits:
        if not time_ok():
            break
        score, sol = run_optimization(v0, maxiter=200)
        if score > best_sum:
            best_sum = score
            best_solution = sol
            
    # Polish the best solution
    if best_solution is not None and time_ok():
        score, sol = run_optimization(best_solution, maxiter=500)
        if score > best_sum:
            best_sum = score
            best_solution = sol

    # Format output
    if best_solution is None:
        W = 1.0
        H = 1.0
        out = []
        k = 5
        for i in range(k):
            for j in range(k):
                if len(out) >= N: break
                x = (i + 0.5) / k
                y = (j + 0.5) / k
                r = min(x, W-x, y, H-y, 0.5/k) * 0.9
                out.append([x, y, r])
        return np.array(out)
    else:
        W = best_solution[0]
        H = 2.0 - W
        out = []
        for i in range(N):
            x = best_solution[1+3*i]
            y = best_solution[2+3*i]
            r = best_solution[3+3*i]
            x = np.clip(x, r, W-r)
            y = np.clip(y, r, H-r)
            r = min(r, x, W-x, y, H-y)
            out.append([x, y, r])
        return np.array(out)
