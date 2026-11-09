import numpy as np
import scipy.optimize
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 100  # Safety margin
    
    best_val = 0.0
    best_sol = None
    
    def flatten(x, y, r):
        return np.concatenate([x, y, r])
        
    def unpack(v):
        return v[:N], v[N:2*N], v[2*N:]
        
    def objective(v):
        x, y, r = unpack(v)
        return -np.sum(r)
        
    cons = []
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            def make_overlap(i, j):
                def fun(v):
                    x, y, r = unpack(v)
                    return (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
                return fun
            cons.append({'type': 'ineq', 'fun': make_overlap(i, j)})
            
    # Boundary constraints
    for i in range(N):
        def make_x_min(i):
            def fun(v):
                x, y, r = unpack(v)
                return x[i] - r[i]
            return fun
        def make_x_max(i):
            def fun(v):
                x, y, r = unpack(v)
                return (1 - x[i]) - r[i]
            return fun
        def make_y_min(i):
            def fun(v):
                x, y, r = unpack(v)
                return y[i] - r[i]
            return fun
        def make_y_max(i):
            def fun(v):
                x, y, r = unpack(v)
                return (1 - y[i]) - r[i]
            return fun
            
        cons.append({'type': 'ineq', 'fun': make_x_min(i)})
        cons.append({'type': 'ineq', 'fun': make_x_max(i)})
        cons.append({'type': 'ineq', 'fun': make_y_min(i)})
        cons.append({'type': 'ineq', 'fun': make_y_max(i)})
        
    bounds = [(0, 1) for _ in range(3*N)]
    
    inits = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    idx = np.arange(N)
    x0 = grid_x[idx].copy()
    y0 = grid_y[idx].copy()
    r0 = np.full(N, 0.1)
    inits.append(flatten(x0, y0, r0))
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.uniform(0.2, 0.8, N)
        y_rand = np.random.uniform(0.2, 0.8, N)
        r_rand = np.full(N, 0.1)
        inits.append(flatten(x_rand, y_rand, r_rand))
        
    for v0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = scipy.optimize.minimize(
                objective, 
                v0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                x_opt, y_opt, r_opt = unpack(res.x)
                valid = True
                for i in range(N):
                    if r_opt[i] <= 1e-9: valid = False
                    if x_opt[i] - r_opt[i] < -1e-7 or x_opt[i] + r_opt[i] > 1 + 1e-7: valid = False
                    if y_opt[i] - r_opt[i] < -1e-7 or y_opt[i] + r_opt[i] > 1 + 1e-7: valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        if d < r_opt[i] + r_opt[j] - 1e-7:
                            valid = False
                            
                if valid:
                    val = np.sum(r_opt)
                    if val > best_val:
                        best_val = val
                        best_sol = np.column_stack([x_opt, y_opt, r_opt])
                        
        except Exception:
            pass
            
    if best_sol is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_sol
