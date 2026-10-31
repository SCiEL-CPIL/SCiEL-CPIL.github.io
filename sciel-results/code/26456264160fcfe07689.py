import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # H is height, W is width. W+H=2.
    # Aspect ratio AR = W/H.
    # H = 2 / (1 + AR), W = 2 * AR / (1 + AR)
    
    ar_values = np.linspace(0.6, 1.8, 15)
    
    for ar in ar_values:
        if time.time() > deadline:
            break
            
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        # Initial guess
        # Estimate radius
        r_est = np.sqrt(0.9 * W * H / (N * np.pi))
        
        # Hex grid init
        # Number of columns
        # dx = 2*r_est, dy = r_est * sqrt(3)
        # Fit as many as possible
        # Simpler: just grid
        cols = int(np.sqrt(N * W / H))
        rows = int(np.ceil(N / cols))
        
        xs = []
        ys = []
        # Center the grid
        # x coords
        x_step = W / cols
        y_step = H / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(xs) >= N:
                    break
                # Jitter or specific pattern
                # Hex offset
                offset = (j % 2) * (x_step / 2.0)
                x = (j + 0.5) * x_step + offset
                y = (i + 0.5) * y_step
                # Clip to bounds with margin
                x = np.clip(x, r_est*0.5, W - r_est*0.5)
                y = np.clip(y, r_est*0.5, H - r_est*0.5)
                xs.append(x)
                ys.append(y)
                
        # Pad if necessary
        while len(xs) < N:
            xs.append(W/2.0)
            ys.append(H/2.0)
            
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        rs = np.full(N, r_est * 0.8)
        
        # Variables: x, y, r, H (W is derived)
        # Var vector: [x1..xN, y1..yN, r1..rN, H]
        # Size: 3N + 1 = 64
        
        x0 = np.concatenate([xs, ys, rs, [H]])
        
        def objective(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            h = v[3*N]
            w = 2.0 - h
            if h <= 0 or w <= 0:
                return 1e10
            return -np.sum(r)
            
        def constraints(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            h = v[3*N]
            w = 2.0 - h
            
            cons = []
            
            # Boundary constraints
            # x >= r, x <= w-r, y >= r, y <= h-r
            cons.append({'type': 'ineq', 'fun': lambda v: v[:N] - v[2*N:3*N]}) # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: (2.0 - v[3*N]) - v[:N] - v[2*N:3*N]}) # w - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[N:2*N] - v[2*N:3*N]}) # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[3*N] - v[N:2*N] - v[2*N:3*N]}) # h - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[3*N]}) # h > 0
            cons.append({'type': 'ineq', 'fun': lambda v: 2.0 - v[3*N]}) # w > 0
            cons.append({'type': 'ineq', 'fun': lambda v: v[2*N:3*N]}) # r > 0
            
            # Non-overlap
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            for i in range(N):
                for j in range(i+1, N):
                    di = i
                    dj = j
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=di, j=dj: 
                            (v[i] - v[j])**2 + (v[N+i] - v[N+j])**2 - (v[2*N+i] + v[2*N+j])**2
                    })
            
            return cons

        # SLSQP is slow with many constraints.
        # 210 non-overlap + 6 boundary + 1 H = 217 constraints.
        # This might be too slow for 15 restarts.
        # Optimization: Combine constraints? No, SLSQP expects list of dicts or array.
        # Using array form for constraints is faster.
        
        def constraint_fun(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            h = v[3*N]
            w = 2.0 - h
            
            # Return array of constraint values (all >= 0)
            c = np.zeros(6 + N*(N-1)//2)
            idx = 0
            c[idx] = np.min(x - r); idx+=1
            c[idx] = np.min(w - x - r); idx+=1
            c[idx] = np.min(y - r); idx+=1
            c[idx] = np.min(h - y - r); idx+=1
            c[idx] = h; idx+=1
            c[idx] = w; idx+=1
            
            # Non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    c[idx] = (x[i] - x[j])**2 + (y[i] - y[j])**2 - (r[i] + r[j])**2
                    idx += 1
            return c

        cons = {'type': 'ineq', 'fun': constraint_fun}
        
        # Bounds
        # x: [0, 2], y: [0, 2], r: [0, 1], H: [0.1, 1.9]
        bounds = [(0, 2.0)] * N + [(0, 2.0)] * N + [(0, 1.0)] * N + [(0.1, 1.9)]
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                # Extract result
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                h = res.x[3*N]
                w = 2.0 - h
                # Validate
                valid = True
                if h <= 0 or w <= 0:
                    valid = False
                for i in range(N):
                    if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                        valid = False
                        break
                if valid:
                    best_result = np.column_stack((x, y, r))
                    
    if best_result is not None:
        return best_result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
