import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin
    
    # Variables: x1, y1, r1, ..., x21, y21, r21, W, H
    # Indices:
    # x: 0..20
    # y: 21..41
    # r: 42..62
    # W: 63
    # H: 64
    
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        return x, y, r, W, H
    
    def pack(x, y, r, W, H):
        return np.concatenate([x, y, r, [W, H]])
    
    def objective(v):
        # Maximize sum of radii
        r = v[2*N:3*N]
        return -np.sum(r)
    
    def constraint_funs(v):
        # Returns list of constraint values (must be >= 0)
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        vals = []
        
        # Perimeter: W + H <= 2
        vals.append(2.0 - (W + H))
        
        # W, H >= 0.1
        vals.append(W - 0.1)
        vals.append(H - 0.1)
        
        # r_i >= 1e-6
        for i in range(N):
            vals.append(r[i] - 1e-6)
            
        # Boundaries
        for i in range(N):
            vals.append(x[i] - r[i])       # Left
            vals.append(W - x[i] - r[i])   # Right
            vals.append(y[i] - r[i])       # Bottom
            vals.append(H - y[i] - r[i])   # Top
            
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                vals.append(dist_sq - rad_sum**2)
                
        return vals

    def constraints_list():
        # SLSQP requires a list of dicts
        # We can't easily vectorize the list creation for SLSQP in the standard interface without passing index
        # So we create a list of lambdas that call constraint_funs? No, that's inefficient.
        # Better: define individual constraint functions?
        # Or use a single function that returns an array?
        # SLSQP does NOT support array-returning inequality constraints directly in the list of dicts.
        # Each dict's 'fun' must return a scalar.
        
        # This is a known limitation.
        # We have to create ~280 constraint dicts.
        
        cons = []
        # Perimeter
        cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - (v[3*N] + v[3*N+1])})
        # W, H bounds
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*N] - 0.1})
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*N+1] - 0.1})
        
        # Radii bounds
        for i in range(N):
            idx = 2*N + i
            cons.append({'type': 'ineq', 'fun': lambda v, idx=idx: v[idx] - 1e-6})
            
        # Boundaries
        for i in range(N):
            xi = i
            yi = N + i
            ri = 2*N + i
            cons.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[xi] - v[ri]})
            cons.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri, W=3*N: v[W] - v[xi] - v[ri]})
            cons.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[yi] - v[ri]})
            cons.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri, H=3*N+1: v[H] - v[yi] - v[ri]})
            
        # Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                xi, xj = i, j
                yi, yj = N+i, N+j
                ri, rj = 2*N+i, 2*N+j
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, xi=xi, xj=xj, yi=yi, yj=yj, ri=ri, rj=rj: \
                          (v[xi]-v[xj])**2 + (v[yi]-v[yj])**2 - (v[ri]+v[rj])**2
                })
        return cons

    def initial_guess(config_type='grid'):
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.05)
        
        if config_type == 'grid':
            cols = 5
            rows = 5
            W_init = 1.0
            H_init = 1.0
            dx = W_init / cols
            dy = H_init / rows
            count = 0
            for i in range(cols):
                for j in range(rows):
                    if count >= N: break
                    x[count] = (i + 0.5) * dx
                    y[count] = (j + 0.5) * dy
                    count += 1
            return pack(x, y, r, W_init, H_init)
            
        elif config_type == 'hex':
            W_init = 1.0
            H_init = 1.0
            r_init = 0.15
            dx = 2 * r_init
            dy = np.sqrt(3) * r_init
            count = 0
            row = 0
            while count < N:
                for col in range(int(W_init / dx) + 2):
                    if count >= N: break
                    x_offset = (col + (row % 2) * 0.5) * dx
                    y_offset = (row + 0.5) * dy
                    if x_offset + r_init <= W_init and y_offset + r_init <= H_init:
                        x[count] = x_offset
                        y[count] = y_offset
                        count += 1
                row += 1
                if row > 10: break # Safety
            return pack(x, y, r, W_init, H_init)
            
        else:
            x = np.random.uniform(0.2, 0.8, N)
            y = np.random.uniform(0.2, 0.8, N)
            r = np.full(N, 0.05)
            return pack(x, y, r, 1.0, 1.0)

    best_value = -np.inf
    best_v = None
    cons = constraints_list()
    
    configs = ['grid', 'hex', 'random', 'random', 'random']
    
    for cfg in configs:
        if time.time() > deadline:
            break
            
        v0 = initial_guess(cfg)
        
        try:
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                constraints=cons, 
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            # Check if result is valid
            x, y, r, W, H = unpack(res.x)
            valid = True
            if W + H > 2.0001: valid = False
            if W < 0.1 or H < 0.1: valid = False
            for i in range(N):
                if r[i] < 1e-6: valid = False
                if x[i] - r[i] < -1e-6: valid = False
                if x[i] + r[i] - W > 1e-6: valid = False
                if y[i] - r[i] < -1e-6: valid = False
                if y[i] + r[i] - H > 1e-6: valid = False
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    rad_sum_sq = (r[i]+r[j])**2
                    if dist_sq < rad_sum_sq - 1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                val = -res.fun
                if val > best_value:
                    best_value = val
                    best_v = res.x
                    
        except Exception:
            pass

    if best_v is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, W, H = unpack(best_v)
    result = np.column_stack((x, y, r))
    return result
