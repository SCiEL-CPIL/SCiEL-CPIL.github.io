import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    best_val = -np.inf
    best_x = None
    
    # Objective function to minimize: -sum(r) + penalties
    def objective_and_grad(x):
        w = x[0]
        h = x[1]
        # x_i, y_i, r_i for i=0..20
        # x[2:2+N], x[2+N:2+2N], x[2+2N:2+3N]
        xs = x[2:2+N]
        ys = x[2+N:2+2*N]
        rs = x[2+2*N:2+3*N]
        
        # Sum of radii
        sum_r = np.sum(rs)
        
        # Penalties
        pen_perim = 0.0
        perim_viol = 2*(w+h) - 4
        if perim_viol > 0:
            pen_perim = perim_viol**2
        
        pen_overlap = 0.0
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                viol = rs[i] + rs[j] - dist
                if viol > 0:
                    pen_overlap += viol**2
        
        pen_bound = 0.0
        for i in range(N):
            # x bounds
            v1 = rs[i] - xs[i]
            if v1 > 0: pen_bound += v1**2
            v2 = xs[i] + rs[i] - w
            if v2 > 0: pen_bound += v2**2
            # y bounds
            v3 = rs[i] - ys[i]
            if v3 > 0: pen_bound += v3**2
            v4 = ys[i] + rs[i] - h
            if v4 > 0: pen_bound += v4**2
            
        val = -sum_r + 100*pen_perim + 100*pen_overlap + 100*pen_bound
        
        # Gradient
        grad = np.zeros_like(x)
        
        # w, h
        if perim_viol > 0:
            grad[0] = 100 * 2 * perim_viol * 2
            grad[1] = 100 * 2 * perim_viol * 2
        else:
            grad[0] = 0
            grad[1] = 0
            
        # Overlaps gradient
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                viol = rs[i] + rs[j] - dist
                if viol > 0:
                    # d(viol^2)/dx_i = 2*viol * d(viol)/dx_i
                    # d(viol)/dx_i = -d(dist)/dx_i = -dx/dist
                    # d(viol)/dx_j = dx/dist
                    # d(viol)/dr_i = 1
                    # d(viol)/dr_j = 1
                    if dist > 1e-8:
                        g_dist_i = -dx/dist
                        g_dist_j = dx/dist
                    else:
                        g_dist_i = 0
                        g_dist_j = 0
                        
                    grad[2+i] += 100 * 2 * viol * g_dist_i
                    grad[2+N+j] += 100 * 2 * viol * g_dist_j
                    grad[2+2*N+i] += 100 * 2 * viol * 1
                    grad[2+2*N+j] += 100 * 2 * viol * 1
                    # y components
                    g_dist_yi = -dy/dist
                    g_dist_yj = dy/dist
                    grad[2+N+i] += 100 * 2 * viol * g_dist_yi
                    grad[2+2*N+j] += 100 * 2 * viol * g_dist_yj
                    
        # Bounds gradient
        for i in range(N):
            v1 = rs[i] - xs[i]
            if v1 > 0:
                grad[2+i] += 100 * 2 * v1 * (-1)
                grad[2+2*N+i] += 100 * 2 * v1 * (1)
            v2 = xs[i] + rs[i] - w
            if v2 > 0:
                grad[2+i] += 100 * 2 * v2 * (1)
                grad[2+2*N+i] += 100 * 2 * v2 * (1)
                grad[0] += 100 * 2 * v2 * (-1)
                
            v3 = rs[i] - ys[i]
            if v3 > 0:
                grad[2+N+i] += 100 * 2 * v3 * (-1)
                grad[2+2*N+i] += 100 * 2 * v3 * (1)
            v4 = ys[i] + rs[i] - h
            if v4 > 0:
                grad[2+N+i] += 100 * 2 * v4 * (1)
                grad[2+2*N+i] += 100 * 2 * v4 * (1)
                grad[1] += 100 * 2 * v4 * (-1)
                
        # Sum of radii gradient
        for i in range(N):
            grad[2+2*N+i] -= 1
            
        return val, grad

    bounds = []
    bounds.append((0.1, 1.9)) # w
    bounds.append((0.1, 1.9)) # h
    for i in range(N):
        bounds.append((0, 2)) # x
    for i in range(N):
        bounds.append((0, 2)) # y
    for i in range(N):
        bounds.append((0.001, 1.0)) # r

    # Multiple restarts
    import time
    start_time = time.time()
    time_limit = 120 # Safety margin for 150s kill
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Random init
        w0 = 1.0
        h0 = 1.0
        xs0 = np.random.uniform(0.2, 0.8, N)
        ys0 = np.random.uniform(0.2, 0.8, N)
        rs0 = np.random.uniform(0.05, 0.15, N)
        
        x0 = np.concatenate(([w0, h0], xs0, ys0, rs0))
        
        # Optimize
        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                x_opt = res.x
                # Check feasibility
                w, h = x_opt[0], x_opt[1]
                xs, ys, rs = x_opt[2:2+N], x_opt[2+N:2+2*N], x_opt[2+2*N:2+3*N]
                
                # Verify constraints strictly
                valid = True
                if 2*(w+h) > 4 + 1e-6: valid = False
                for i in range(N):
                    if xs[i] < -1e-6 or xs[i] > w + 1e-6 or ys[i] < -1e-6 or ys[i] > h + 1e-6 or rs[i] < 1e-6:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if dist < rs[i] + rs[j] - 1e-6:
                            valid = False
                            
                if valid:
                    val = np.sum(rs)
                    if val > best_val:
                        best_val = val
                        best_x = x_opt
        except Exception as e:
            pass

    if best_x is not None:
        w, h = best_x[0], best_x[1]
        xs = best_x[2:2+N]
        ys = best_x[2+N:2+2*N]
        rs = best_x[2+2*N:2+3*N]
        # Clip to ensure no negative radii or tiny violations
        rs = np.maximum(rs, 1e-6)
        result = np.column_stack((xs, ys, rs))
        return result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
