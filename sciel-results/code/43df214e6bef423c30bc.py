import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    max_time = 100.0 # Safety margin
    
    best_score = 0.0
    best_result = None
    
    # Generate initial guesses
    def make_initial(W):
        H = 2.0 - W
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        for k in range(N):
            i = k % side
            j = k // side
            xs[k] = (i + 0.5) * (W / side)
            ys[k] = (j + 0.5) * (H / side)
            # Radius slightly smaller than grid cell to be feasible initially
            rs[k] = 0.5 * min(W / side, H / side) * 0.8
        return np.concatenate([xs, ys, rs, [W]])
        
    initial_configs = [make_initial(1.0), make_initial(1.2), make_initial(0.8)]
    
    # Add random perturbations
    np.random.seed(42) # For reproducibility in evaluation? No, random is fine.
    for _ in range(5):
        v0 = make_initial(1.0).copy()
        v0[:2*N] += np.random.normal(0, 0.05, 2*N)
        v0[2*N:3*N] = np.abs(v0[2*N:3*N]) + 0.05
        initial_configs.append(v0)

    for v0 in initial_configs:
        if time.time() - start_time > max_time - 5:
            break
            
        # Variables: x_1..x_N, y_1..y_N, r_1..r_N, W
        # H = 2 - W
        
        def objective(v):
            return -np.sum(v[2*N:3*N])
            
        def jac(v):
            grad = np.zeros_like(v)
            grad[2*N:3*N] = -1.0
            return grad
            
        cons = []
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                def pair_constraint(v, i=i, j=j):
                    dx = v[i] - v[j]
                    dy = v[N+i] - v[N+j]
                    ri = v[2*N+i]
                    rj = v[2*N+j]
                    return (dx*dx + dy*dy) - (ri+rj)**2
                cons.append({'type': 'ineq', 'fun': pair_constraint})
                
        # Boundary constraints
        for i in range(N):
            def x_min(v, i=i):
                return v[i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': x_min})
            
            def x_max(v, i=i):
                W = v[3*N]
                return W - v[i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': x_max})
            
            def y_min(v, i=i):
                return v[N+i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': y_min})
            
            def y_max(v, i=i):
                W = v[3*N]
                H = 2.0 - W
                return H - v[N+i] - v[2*N+i]
            cons.append({'type': 'ineq', 'fun': y_max})
            
            def r_pos(v, i=i):
                return v[2*N+i]
            cons.append({'type': 'ineq', 'fun': r_pos})
            
        def W_min(v):
            return v[3*N] - 0.1
        cons.append({'type': 'ineq', 'fun': W_min})
        
        def W_max(v):
            return 1.9 - v[3*N]
        cons.append({'type': 'ineq', 'fun': W_max})

        try:
            res = minimize(objective, v0, method='SLSQP', jac=jac, 
                           constraints=cons, options={'maxiter': 300, 'ftol': 1e-10})
            
            if time.time() - start_time > max_time:
                break
                
            if res.success:
                v_opt = res.x
                xs = v_opt[:N]
                ys = v_opt[N:2*N]
                rs = v_opt[2*N:3*N]
                W = v_opt[3*N]
                
                # Verify validity strictly
                valid = True
                if W <= 0.1 or 2.0-W <= 0.1:
                    valid = False
                else:
                    H = 2.0 - W
                    for i in range(N):
                        if xs[i] < rs[i] - 1e-8 or xs[i] > W - rs[i] + 1e-8:
                            valid = False; break
                        if ys[i] < rs[i] - 1e-8 or ys[i] > H - rs[i] + 1e-8:
                            valid = False; break
                        if rs[i] <= 0:
                            valid = False; break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = xs[i] - xs[j]
                                dy = ys[i] - ys[j]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < rs[i] + rs[j] - 1e-8:
                                    valid = False; break
                            if not valid: break
                            
                if valid:
                    score = np.sum(rs)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((xs, ys, rs))
                        
        except Exception as e:
            pass

    if best_result is not None:
        return best_result
    else:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
