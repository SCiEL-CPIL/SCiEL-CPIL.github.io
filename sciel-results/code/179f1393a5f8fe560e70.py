import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0

    def unpack(v):
        w = v[0]
        x = v[1:1+N]
        y = v[1+N:1+2*N]
        r = v[1+2*N:1+3*N]
        return w, x, y, r

    def objective(v):
        w, x, y, r = unpack(v)
        return -np.sum(r)

    def constraints_array(v):
        w, x, y, r = unpack(v)
        h = 2.0 - w
        c = []
        # Bounds
        c.extend(x - r)
        c.extend(w - x - r)
        c.extend(y - r)
        c.extend(h - y - r)
        # Pairs
        for i in range(N):
            xi = x[i]
            yi = y[i]
            ri = r[i]
            for j in range(i+1, N):
                c.append((xi-x[j])**2 + (yi-y[j])**2 - (ri+r[j])**2)
        # w bounds
        c.append(w - 0.5)
        c.append(1.5 - w)
        # r > 0
        c.extend(r - 1e-6)
        return np.array(c)

    constraints = [{'type': 'ineq', 'fun': constraints_array}]
    bounds = [(0.5, 1.5)] + [(1e-6, 1.5)]*(3*N)

    def run_slsqp(init_v):
        if time.time() - start_time > deadline:
            return None, None
        try:
            res = minimize(
                objective,
                init_v,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.success or res.fun < 0: # Check if it did something
                 return res, objective(res.x)
            else:
                 return None, None
        except Exception:
            return None, None

    best_val = -np.inf
    best_v = None

    def make_grid(w, h, cols, rows):
        v = np.zeros(1 + 3*N)
        v[0] = w
        r_est = min(w/cols, h/rows) * 0.9
        for i in range(N):
            col = i % cols
            row = i // cols
            if row >= rows:
                break
            x_i = (col + 0.5) * (w / cols)
            y_i = (row + 0.5) * (h / rows)
            v[1+i] = x_i
            v[1+N+i] = y_i
            v[1+2*N+i] = r_est
        return v

    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 1.33, 1.1, 0.9]
    
    for ar in aspect_ratios:
        h_init = 2.0 / (1.0 + ar)
        w_init = 2.0 * ar / (1.0 + ar)
        w_init = np.clip(w_init, 0.5, 1.5)
        
        v_init = make_grid(w_init, 2.0-w_init, 5, 5)
        v_init[1:] += np.random.uniform(-0.01, 0.01, size=3*N)
        
        res, val = run_slsqp(v_init)
        if val is not None and val > best_val:
            best_val = val
            best_v = res.x.copy()
            
        if time.time() - start_time < deadline - 10:
             if best_v is not None:
                 v_pert = best_v.copy()
                 v_pert[1:] += np.random.normal(0, 0.05, size=3*N)
                 v_pert[0] = np.clip(v_pert[0], 0.5, 1.5)
                 for i in range(N):
                     v_pert[1+i] = np.clip(v_pert[1+i], 0.01, 1.49)
                     v_pert[1+N+i] = np.clip(v_pert[1+N+i], 0.01, 1.49)
                     v_pert[1+2*N+i] = np.clip(v_pert[1+2*N+i], 0.01, 0.5)
                 res, val = run_slsqp(v_pert)
                 if val is not None and val > best_val:
                     best_val = val
                     best_v = res.x.copy()

    if best_v is None:
        w = 1.0
        h = 1.0
        r = 0.1
        best_v = np.zeros(1 + 3*N)
        best_v[0] = w
        for i in range(N):
            col = i % 5
            row = i // 5
            best_v[1+i] = (col + 0.5) * 0.2
            best_v[1+N+i] = (row + 0.5) * 0.2
            best_v[1+2*N+i] = r

    w, x, y, r = unpack(best_v)
    h = 2.0 - w
    
    # Safety: Ensure feasibility
    for _ in range(5):
        feasible = True
        for i in range(N):
            if x[i] < r[i] - 1e-8 or x[i] > w - r[i] + 1e-8:
                feasible = False
            if y[i] < r[i] - 1e-8 or y[i] > h - r[i] + 1e-8:
                feasible = False
        for i in range(N):
            for j in range(i+1, N):
                d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                r_sum = r[i] + r[j]
                if d2 < r_sum**2 - 1e-8:
                    feasible = False
        if feasible:
            break
        r *= 0.95

    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result

# EVOLVE-BLOCK-END
