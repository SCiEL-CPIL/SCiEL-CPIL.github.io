import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Grid
    # A 5x6 grid is better than 6x6 for 26 circles?
    # 5 rows, 6 cols = 30 spots. Use 26.
    # Or 6x5.
    # Let's use a 5x6 grid.
    rows, cols = 5, 6
    xs = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    
    init_x = []
    init_y = []
    init_r = []
    
    # Fill grid
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count < N:
                init_x.append(xs[j])
                init_y.append(ys[i])
                # Radius limited by distance to neighbors and boundaries
                # In a grid, max radius is half the spacing
                dx = xs[1] - xs[0] if cols > 1 else 1.0
                dy = ys[1] - ys[0] if rows > 1 else 1.0
                # Conservative radius to ensure feasibility
                r = min(dx, dy) * 0.5 * 0.9 
                init_r.append(r)
                count += 1
    
    x0 = np.array(init_x + init_y + init_r)
    
    # Define objective
    def objective(v):
        r = v[2*N:]
        return -np.sum(r)
    
    def objective_grad(v):
        grad = np.zeros_like(v)
        grad[2*N:] = -1.0
        return grad

    # Define constraints
    # We will use a list of constraint dictionaries
    # This can be memory intensive for 325 constraints if not handled carefully, but SLSQP can handle it.
    # Alternatively, we can use a single vectorized constraint function if possible, but SLSQP expects a list or a single function returning an array for 'eq'/'ineq'.
    # The 'constraints' argument can be a sequence of dictionaries.
    
    constraints = []
    
    # Pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            idx_xi = i
            idx_yi = N + i
            idx_ri = 2*N + i
            idx_xj = j
            idx_yj = N + j
            idx_rj = 2*N + j
            
            def fun(v, i=i, j=j):
                xi = v[idx_xi]
                yi = v[idx_yi]
                ri = v[idx_ri]
                xj = v[idx_xj]
                yj = v[idx_yj]
                rj = v[idx_rj]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            
            def fun_grad(v, i=i, j=j):
                grad = np.zeros_like(v)
                xi = v[idx_xi]
                yi = v[idx_yi]
                ri = v[idx_ri]
                xj = v[idx_xj]
                yj = v[idx_yj]
                rj = v[idx_rj]
                
                dx = xi - xj
                dy = yi - yj
                rsum = ri + rj
                
                grad[idx_xi] = 2 * dx
                grad[idx_yi] = 2 * dy
                grad[idx_ri] = -2 * rsum
                
                grad[idx_xj] = -2 * dx
                grad[idx_yj] = -2 * dy
                grad[idx_rj] = -2 * rsum
                
                return grad
            
            constraints.append({'type': 'ineq', 'fun': fun, 'jac': fun_grad})
            
    # Boundary constraints
    # x_i >= r_i
    for i in range(N):
        idx_xi = i
        idx_ri = 2*N + i
        
        def fun_x_min(v, i=i):
            return v[idx_xi] - v[idx_ri]
            
        def fun_x_min_grad(v, i=i):
            grad = np.zeros_like(v)
            grad[idx_xi] = 1.0
            grad[idx_ri] = -1.0
            return grad
            
        constraints.append({'type': 'ineq', 'fun': fun_x_min, 'jac': fun_x_min_grad})
        
        # 1 - x_i >= r_i  => 1 - x_i - r_i >= 0
        def fun_x_max(v, i=i):
            return 1.0 - v[idx_xi] - v[idx_ri]
            
        def fun_x_max_grad(v, i=i):
            grad = np.zeros_like(v)
            grad[idx_xi] = -1.0
            grad[idx_ri] = -1.0
            return grad
            
        constraints.append({'type': 'ineq', 'fun': fun_x_max, 'jac': fun_x_max_grad})
        
        # y_i >= r_i
        idx_yi = N + i
        def fun_y_min(v, i=i):
            return v[idx_yi] - v[idx_ri]
            
        def fun_y_min_grad(v, i=i):
            grad = np.zeros_like(v)
            grad[idx_yi] = 1.0
            grad[idx_ri] = -1.0
            return grad
            
        constraints.append({'type': 'ineq', 'fun': fun_y_min, 'jac': fun_y_min_grad})
        
        # 1 - y_i >= r_i
        def fun_y_max(v, i=i):
            return 1.0 - v[idx_yi] - v[idx_ri]
            
        def fun_y_max_grad(v, i=i):
            grad = np.zeros_like(v)
            grad[idx_yi] = -1.0
            grad[idx_ri] = -1.0
            return grad
            
        constraints.append({'type': 'ineq', 'fun': fun_y_max, 'jac': fun_y_max_grad})
        
        # r_i >= 0
        def fun_r_pos(v, i=i):
            return v[idx_ri]
            
        def fun_r_pos_grad(v, i=i):
            grad = np.zeros_like(v)
            grad[idx_ri] = 1.0
            return grad
            
        constraints.append({'type': 'ineq', 'fun': fun_r_pos, 'jac': fun_r_pos_grad})

    # Run optimization
    best_val = -np.inf
    best_x = x0
    
    # Try SLSQP
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                       constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        
        if res.success or (res.fun < best_val and is_feasible(res.x, constraints)):
            if is_feasible(res.x, constraints):
                best_val = -res.fun
                best_x = res.x
            else:
                # If not feasible, try to project or just ignore? 
                # SLSQP should maintain feasibility.
                pass
    except Exception as e:
        pass

    # If time remains, try another initial guess or polish
    if time.time() < deadline - 20.0:
        # Try a perturbed start
        np.random.seed(42)
        x0_pert = x0.copy()
        # Add small noise to positions
        x0_pert[:N] += np.random.uniform(-0.01, 0.01, N)
        x0_pert[N:2*N] += np.random.uniform(-0.01, 0.01, N)
        
        # Ensure initial feasibility for perturbed start might be hard.
        # Instead, let's just run again with different maxiter or tolerance?
        # Or try a different grid orientation?
        pass

    # Final polish: Check feasibility and adjust radii if slightly violated
    # If the result is slightly infeasible, we can reduce radii slightly to make it feasible.
    final_x = best_x
    # Check constraints
    # For robustness, if any constraint is violated, scale down radii
    # Simple check:
    valid = True
    for i in range(N):
        x = final_x[i]
        y = final_x[N+i]
        r = final_x[2*N+i]
        if r < 0:
            final_x[2*N+i] = 0
            valid = False
        if x - r < -1e-5 or 1 - x - r < -1e-5 or y - r < -1e-5 or 1 - y - r < -1e-5:
            # Adjust radius to fit
            max_r = min(x, 1-x, y, 1-y)
            if max_r < 0:
                 # Move center? This is complex. Just clamp radius.
                 final_x[2*N+i] = max(0, max_r)
            valid = False
            
    for i in range(N):
        for j in range(i+1, N):
            xi = final_x[i]
            yi = final_x[N+i]
            ri = final_x[2*N+i]
            xj = final_x[j]
            yj = final_x[N+j]
            rj = final_x[2*N+j]
            dist_sq = (xi-xj)**2 + (yi-yj)**2
            sum_r = ri + rj
            if dist_sq < sum_r**2:
                # Overlap found. Reduce radii.
                # Simple fix: reduce both radii by half the overlap
                overlap = sum_r - np.sqrt(dist_sq)
                if overlap > 0:
                    final_x[2*N+i] -= overlap/2
                    final_x[2*N+j] -= overlap/2
                    valid = False

    # Convert to output format
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = final_x[i]
        result[i, 1] = final_x[N+i]
        result[i, 2] = final_x[2*N+i]
        
    return result

def is_feasible(v, constraints, tol=1e-5):
    for c in constraints:
        val = c['fun'](v)
        if val < -tol:
            return False
    return True
