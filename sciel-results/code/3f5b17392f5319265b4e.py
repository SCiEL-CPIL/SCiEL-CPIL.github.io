import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 110.0

def construct():
    start_time = time.time()
    best_sum_r = -np.inf
    best_array = None

    # Helper to unpack variables
    # vars: x1..xN, y1..yN, r1..rN, xmin, xmax, ymin, ymax
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        xmin = v[3*N]
        xmax = v[3*N+1]
        ymin = v[3*N+2]
        ymax = v[3*N+3]
        return x, y, r, xmin, xmax, ymin, ymax

    def objective(v):
        x, y, r, xmin, xmax, ymin, ymax = unpack(v)
        return -np.sum(r)

    def constraints(v):
        x, y, r, xmin, xmax, ymin, ymax = unpack(v)
        cons = []
        
        # Overlap constraints: dist^2 >= (ri+rj)^2
        # We want dist^2 - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist2 = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                cons.append(dist2 - sum_r*sum_r)
        
        # Bounds
        for i in range(N):
            cons.append(x[i] - r[i] - xmin)
            cons.append(xmax - (x[i] + r[i]))
            cons.append(y[i] - r[i] - ymin)
            cons.append(ymax - (y[i] + r[i]))
            cons.append(r[i]) # r >= 0
            
        # Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
        # W = xmax - xmin, H = ymax - ymin
        # 2 - (W+H) >= 0
        W = xmax - xmin
        H = ymax - ymin
        cons.append(2.0 - (W + H))
        
        return cons

    def make_constraint_dict():
        # SLSQP expects 'type': 'ineq' for g(x) >= 0
        # We have many constraints, so we pass them as a list of dicts or a single function returning array?
        # scipy.optimize.minimize supports 'constraints' as a list of dicts.
        # It is more efficient to define a single function that returns the array of constraint values
        # and use 'type': 'ineq' with that function? No, standard interface is list of dicts.
        # Alternatively, use a single constraint function returning an array and specify 'type': 'ineq'
        # Actually, passing a list of dicts for ~300 constraints is slow.
        # Better: define a function that returns the vector of constraint values, 
        # and use `constraints=[{'type': 'ineq', 'fun': constraint_func}]`?
        # No, the standard way for many constraints is:
        # constraints = [{'type': 'ineq', 'fun': lambda v: constraint_values(v)}]
        # But SLSQP might not support vector-valued constraint functions directly in older SciPy versions?
        # Modern SciPy (>= 1.0) supports it if 'fun' returns an array? 
        # Actually, the documentation says: "fun : callable, shape (..., n) ... if type is 'ineq', fun(x) >= 0"
        # It seems to expect a scalar for each dict.
        # To be safe and fast, we can use a penalty method or stick to SLSQP with a list of dicts if N is small.
        # N=21 -> 210 overlap + 84 bounds + 1 perimeter = 295 constraints.
        # A list of 295 dicts is manageable.
        
        cons_list = []
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                def f(v, i=i, j=j):
                    x, y, r, xmin, xmax, ymin, ymax = unpack(v)
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    return dx*dx + dy*dy - (r[i] + r[j])**2
                cons_list.append({'type': 'ineq', 'fun': f})
        
        # Bounds
        for i in range(N):
            def f_min_x(v, i=i):
                x, y, r, xmin, xmax, ymin, ymax = unpack(v)
                return x[i] - r[i] - xmin
            def f_max_x(v, i=i):
                x, y, r, xmin, xmax, ymin, ymax = unpack(v)
                return xmax - (x[i] + r[i])
            def f_min_y(v, i=i):
                x, y, r, xmin, xmax, ymin, ymax = unpack(v)
                return y[i] - r[i] - ymin
            def f_max_y(v, i=i):
                x, y, r, xmin, xmax, ymin, ymax = unpack(v)
                return ymax - (y[i] + r[i])
            def f_r(v, i=i):
                x, y, r, xmin, xmax, ymin, ymax = unpack(v)
                return r[i]
                
            cons_list.append({'type': 'ineq', 'fun': f_min_x})
            cons_list.append({'type': 'ineq', 'fun': f_max_x})
            cons_list.append({'type': 'ineq', 'fun': f_min_y})
            cons_list.append({'type': 'ineq', 'fun': f_max_y})
            cons_list.append({'type': 'ineq', 'fun': f_r})
            
        # Perimeter
        def f_perim(v):
            x, y, r, xmin, xmax, ymin, ymax = unpack(v)
            return 2.0 - ((xmax - xmin) + (ymax - ymin))
        cons_list.append({'type': 'ineq', 'fun': f_perim})
        
        return cons_list

    constraints_list = make_constraint_dict()
    
    # Initial guesses
    def get_grid_init(perturb=0.0):
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.09)
        
        for k in range(N):
            i = k % side
            j = k // side
            x[k] = (i + 0.5) / side
            y[k] = (j + 0.5) / side
            
        if perturb > 0:
            x += np.random.uniform(-perturb, perturb, N)
            y += np.random.uniform(-perturb, perturb, N)
            
        xmin = np.min(x - r)
        xmax = np.max(x + r)
        ymin = np.min(y - r)
        ymax = np.max(y + r)
        
        v = np.concatenate([x, y, r, [xmin, xmax, ymin, ymax]])
        return v

    # Run multi-start
    starts = [
        get_grid_init(0.0),
        get_grid_init(0.01),
        get_grid_init(0.02),
    ]
    
    # Add a random start
    # Random centers in [0.1, 0.9], r=0.05
    x_r = np.random.uniform(0.1, 0.9, N)
    y_r = np.random.uniform(0.1, 0.9, N)
    r_r = np.full(N, 0.05)
    xmin_r = np.min(x_r - r_r)
    xmax_r = np.max(x_r + r_r)
    ymin_r = np.min(y_r - r_r)
    ymax_r = np.max(y_r + r_r)
    starts.append(np.concatenate([x_r, y_r, r_r, [xmin_r, xmax_r, ymin_r, ymax_r]]))

    for v0 in starts:
        if time.time() > DEADLINE:
            break
            
        try:
            res = minimize(objective, v0, method='SLSQP', constraints=constraints_list, options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.success:
                x, y, r, xmin, xmax, ymin, ymax = unpack(res.x)
                
                # Verify constraints strictly
                valid = True
                # Check overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: continue
                
                # Check bounds
                for i in range(N):
                    if x[i] - r[i] < xmin - 1e-6 or x[i] + r[i] > xmax + 1e-6:
                        valid = False
                        break
                    if y[i] - r[i] < ymin - 1e-6 or y[i] + r[i] > ymax + 1e-6:
                        valid = False
                        break
                    if r[i] < -1e-6:
                        valid = False
                        break
                if not valid: continue
                
                # Check perimeter
                W = xmax - xmin
                H = ymax - ymin
                if 2*(W+H) > 4.0 + 1e-6:
                    valid = False
                    continue
                    
                if valid:
                    val = np.sum(r)
                    if val > best_sum_r:
                        best_sum_r = val
                        best_array = np.column_stack((x, y, r))
                        
        except Exception as e:
            continue
            
    if best_array is None:
        # Fallback: return the initial grid (which is valid)
        v0 = get_grid_init(0.0)
        x, y, r, xmin, xmax, ymin, ymax = unpack(v0)
        best_array = np.column_stack((x, y, r))
        
    return best_array
