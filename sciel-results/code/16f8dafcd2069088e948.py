import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 110.0

def construct():
    start_time = time.time()
    
    def unpack(v):
        x = v[0::3][:N]
        y = v[1::3][:N]
        r = v[2::3][:N]
        w = v[-1]
        return x, y, r, w

    def objective(v):
        _, _, r, _ = unpack(v)
        return -np.sum(r)

    def constraint_fun(v):
        x = v[0::3][:N]
        y = v[1::3][:N]
        r = v[2::3][:N]
        w = v[-1]
        h = 2.0 - w
        
        c_x_min = x - r
        c_x_max = w - x - r
        c_y_min = y - r
        c_y_max = h - y - r
        
        c_overlap = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                c_overlap.append(dx*dx + dy*dy - dr*dr)
                
        all_c = np.concatenate([c_x_min, c_x_max, c_y_min, c_y_max, c_overlap])
        return all_c

    def constraint_jac(v):
        x = v[0::3][:N]
        y = v[1::3][:N]
        r = v[2::3][:N]
        w = v[-1]
        
        n_constraints = 4*N + N*(N-1)//2
        jac = np.zeros((n_constraints, 3*N + 1))
        
        for i in range(N):
            jac[i, 3*i] = 1.0
            jac[i, 3*i+2] = -1.0
            
            jac[N+i, 3*i] = -1.0
            jac[N+i, 3*i+2] = -1.0
            jac[N+i, 3*N] = 1.0
            
            jac[2*N+i, 3*i+1] = 1.0
            jac[2*N+i, 3*i+2] = -1.0
            
            jac[3*N+i, 3*i+1] = -1.0
            jac[3*N+i, 3*i+2] = -1.0
            jac[3*N+i, 3*N] = -1.0
            
        idx = 4*N
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                
                jac[idx, 3*i] = 2*dx
                jac[idx, 3*j] = -2*dx
                jac[idx, 3*i+1] = 2*dy
                jac[idx, 3*j+1] = -2*dy
                jac[idx, 3*i+2] = -2*dr
                jac[idx, 3*j+2] = -2*dr
                
                idx += 1
                
        return jac

    cons = [
        {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0, 2))
        bounds.append((0, 2))
        bounds.append((0, 1))
    bounds.append((0.5, 1.5))
    
    best_val = -np.inf
    best_v = None
    
    # Greedy initialization
    def greedy_init():
        w = 1.0
        h = 1.0
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        grid_size = 40
        xs = np.linspace(0.05, w-0.05, grid_size)
        ys = np.linspace(0.05, h-0.05, grid_size)
        
        for i in range(N):
            best_r = 0
            best_x = 0.5
            best_y = 0.5
            
            for gx in xs:
                for gy in ys:
                    r_bound = min(gx, w-gx, gy, h-gy)
                    if r_bound <= 0:
                        continue
                    
                    r_max = r_bound
                    for j in range(i):
                        dist = np.sqrt((gx - x[j])**2 + (gy - y[j])**2)
                        r_allowed = (dist - r[j]) / 2.0
                        if r_allowed < r_max:
                            r_max = r_allowed
                            if r_max <= 0:
                                break
                                
                    if r_max > best_r:
                        best_r = r_max
                        best_x = gx
                        best_y = gy
                        
            x[i] = best_x
            y[i] = best_y
            r[i] = best_r
            
        v0 = np.zeros(N*3 + 1)
        v0[0::3][:N] = x
        v0[1::3][:N] = y
        v0[2::3][:N] = r
        v0[-1] = w
        return v0

    v0_greedy = greedy_init()
    
    # Try greedy first
    try:
        res = minimize(objective, v0_greedy, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
        if res.success:
            val = -res.fun
            if val > best_val:
                best_val = val
                best_v = res.x
    except Exception:
        pass

    # Try random restarts if time permits
    num_restarts = 10
    for k in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        w0 = np.random.uniform(0.8, 1.2)
        h0 = 2.0 - w0
        x0 = np.random.uniform(0.1, w0 - 0.1, N)
        y0 = np.random.uniform(0.1, h0 - 0.1, N)
        r0 = np.random.uniform(0.05, 0.15, N)
        
        v0 = np.zeros(N*3 + 1)
        v0[0::3][:N] = x0
        v0[1::3][:N] = y0
        v0[2::3][:N] = r0
        v0[-1] = w0
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_v = res.x
        except Exception:
            pass
            
    if best_v is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, w = unpack(best_v)
    result = np.column_stack((x, y, r))
    return result
