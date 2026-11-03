import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _compute_constraints_and_grads(x, w, h):
    """
    x: [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    Returns:
        constraints: list of dicts for SLSQP
        grads: list of gradient arrays for constraints
    """
    n = N
    pos_r = x.reshape(n, 3)
    
    constraints = []
    grads = []
    
    # Wall constraints
    # x_i - r_i >= 0
    # w - x_i - r_i >= 0
    # y_i - r_i >= 0
    # h - y_i - r_i >= 0
    
    for i in range(n):
        # Left wall
        c_val = pos_r[i, 0] - pos_r[i, 2]
        g = np.zeros(3 * n)
        g[3*i] = 1.0
        g[3*i+2] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, c=c_val: c})
        grads.append(g)
        
        # Right wall
        c_val = w - pos_r[i, 0] - pos_r[i, 2]
        g = np.zeros(3 * n)
        g[3*i] = -1.0
        g[3*i+2] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, c=c_val: c})
        grads.append(g)
        
        # Bottom wall
        c_val = pos_r[i, 1] - pos_r[i, 2]
        g = np.zeros(3 * n)
        g[3*i+1] = 1.0
        g[3*i+2] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, c=c_val: c})
        grads.append(g)
        
        # Top wall
        c_val = h - pos_r[i, 1] - pos_r[i, 2]
        g = np.zeros(3 * n)
        g[3*i+1] = -1.0
        g[3*i+2] = -1.0
        constraints.append({'type': 'ineq', 'fun': lambda x, c=c_val: c})
        grads.append(g)

    # Circle-circle constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = pos_r[i]
            xj, yj, rj = pos_r[j]
            
            dx = xi - xj
            dy = yi - yj
            dr = ri + rj
            
            c_val = dx*dx + dy*dy - dr*dr
            
            g = np.zeros(3 * n)
            # Gradient w.r.t xi
            g[3*i] = 2 * dx
            # Gradient w.r.t yi
            g[3*i+1] = 2 * dy
            # Gradient w.r.t ri
            g[3*i+2] = -2 * dr
            # Gradient w.r.t xj
            g[3*j] = -2 * dx
            # Gradient w.r.t yj
            g[3*j+1] = -2 * dy
            # Gradient w.r.t rj
            g[3*j+2] = -2 * dr
            
            constraints.append({'type': 'ineq', 'fun': lambda x, c=c_val: c})
            grads.append(g)
            
    return constraints, grads

def _pack_qcp(w, h, n_restarts=5, max_iter=100):
    """
    Pack circles in w x h rectangle using SLSQP with QCP formulation.
    """
    best_score = -1
    best_x = None
    
    # Objective: maximize sum of radii -> minimize -sum(radii)
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)
        
    def objective_grad(x):
        g = np.zeros(3 * N)
        # d(-sum(r_i))/dr_i = -1
        g[2::3] = -1.0
        return g

    for r in range(n_restarts):
        # Generate initial guess
        if r == 0:
            # Grid init
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            x0 = np.zeros(3 * N)
            for i in range(N):
                col = i % cols
                row = i // cols
                x0[3*i] = (col + 0.5) * w / cols
                x0[3*i+1] = (row + 0.5) * h / rows
                # Initial radius estimate: half min spacing
                r_est = min(w / cols, h / rows) * 0.4
                x0[3*i+2] = r_est
        else:
            # Random init
            x0 = np.random.rand(3 * N)
            x0[0::3] *= w
            x0[1::3] *= h
            # Small radii to ensure feasibility initially
            x0[2::3] *= 0.1 * min(w, h) / N
            
        # Ensure initial feasibility is not too violated, though SLSQP handles it
        # If initial radii are too large, constraints are violated. 
        # SLSQP can handle infeasible starts but it's risky.
        # Let's scale down radii if they seem too big relative to spacing.
        
        try:
            # Get initial constraints and grads
            cons, grads = _compute_constraints_and_grads(x0, w, h)
            
            # We need to recompute constraints and grads at each step for SLSQP?
            # SLSQP accepts 'jac' for constraints if provided as a callable or list.
            # However, the standard SLSQP interface with a list of constraint dicts 
            # expects 'fun' and optionally 'jac'. 
            # Providing 'jac' as a list of arrays corresponding to the current point 
            # is not how SLSQP works directly in the list format. 
            # SLSQP uses finite differences for constraints if 'jac' is not a callable 
            # that returns the Jacobian matrix for all constraints at once.
            
            # To use analytical gradients efficiently, we should define a single 
            # Jacobian function that returns a 2D array (n_constraints x n_variables).
            
            def jacobian(x):
                n_cons = 4 * N + N * (N - 1) // 2
                jac = np.zeros((n_cons, 3 * N))
                pos_r = x.reshape(N, 3)
                
                idx = 0
                for i in range(N):
                    # Left
                    jac[idx, 3*i] = 1.0
                    jac[idx, 3*i+2] = -1.0
                    idx += 1
                    # Right
                    jac[idx, 3*i] = -1.0
                    jac[idx, 3*i+2] = -1.0
                    idx += 1
                    # Bottom
                    jac[idx, 3*i+1] = 1.0
                    jac[idx, 3*i+2] = -1.0
                    idx += 1
                    # Top
                    jac[idx, 3*i+1] = -1.0
                    jac[idx, 3*i+2] = -1.0
                    idx += 1
                    
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = pos_r[i]
                        xj, yj, rj = pos_r[j]
                        dx = xi - xj
                        dy = yi - yj
                        dr = ri + rj
                        
                        jac[idx, 3*i] = 2 * dx
                        jac[idx, 3*i+1] = 2 * dy
                        jac[idx, 3*i+2] = -2 * dr
                        jac[idx, 3*j] = -2 * dx
                        jac[idx, 3*j+1] = -2 * dy
                        jac[idx, 3*j+2] = -2 * dr
                        idx += 1
                        
                return jac

            def constraint_fun(x):
                c = np.zeros(4 * N + N * (N - 1) // 2)
                pos_r = x.reshape(N, 3)
                idx = 0
                for i in range(N):
                    c[idx] = pos_r[i, 0] - pos_r[i, 2]
                    idx += 1
                    c[idx] = w - pos_r[i, 0] - pos_r[i, 2]
                    idx += 1
                    c[idx] = pos_r[i, 1] - pos_r[i, 2]
                    idx += 1
                    c[idx] = h - pos_r[i, 1] - pos_r[i, 2]
                    idx += 1
                    
                for i in range(N):
                    for j in range(i+1, N):
                        xi, yi, ri = pos_r[i]
                        xj, yj, rj = pos_r[j]
                        dx = xi - xj
                        dy = yi - yj
                        dr = ri + rj
                        c[idx] = dx*dx + dy*dy - dr*dr
                        idx += 1
                return c

            # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
            bounds = []
            max_r = min(w, h) / 2.0
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((0, max_r))

            result = minimize(objective, x0, method='SLSQP', jac=objective_grad,
                              bounds=bounds,
                              constraints=[{'type': 'ineq', 'fun': constraint_fun, 'jac': jacobian}],
                              options={'maxiter': max_iter, 'ftol': 1e-8})
            
            if result.success:
                # Verify feasibility
                x_opt = result.x
                pos_r = x_opt.reshape(N, 3)
                feasible = True
                for i in range(N):
                    if pos_r[i, 0] < -1e-6 or pos_r[i, 0] > w + 1e-6 or \
                       pos_r[i, 1] < -1e-6 or pos_r[i, 1] > h + 1e-6 or \
                       pos_r[i, 2] < -1e-6:
                        feasible = False
                        break
                    if pos_r[i, 0] - pos_r[i, 2] < -1e-6 or \
                       w - pos_r[i, 0] - pos_r[i, 2] < -1e-6 or \
                       pos_r[i, 1] - pos_r[i, 2] < -1e-6 or \
                       h - pos_r[i, 1] - pos_r[i, 2] < -1e-6:
                        feasible = False
                        break
                    
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = pos_r[i, 0] - pos_r[j, 0]
                            dy = pos_r[i, 1] - pos_r[j, 1]
                            dr = pos_r[i, 2] + pos_r[j, 2]
                            if dx*dx + dy*dy < dr*dr - 1e-6:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    score = np.sum(pos_r[:, 2])
                    if score > best_score:
                        best_score = score
                        best_x = x_opt
                        
        except Exception as e:
            pass
            
    if best_x is not None:
        pos_r = best_x.reshape(N, 3)
        # Clip radii to be strictly positive and within bounds to avoid evaluator issues
        radii = np.maximum(0, pos_r[:, 2])
        # Ensure centers are within [r, w-r]
        for i in range(N):
            pos_r[i, 0] = np.clip(pos_r[i, 0], radii[i], w - radii[i])
            pos_r[i, 1] = np.clip(pos_r[i, 1], radii[i], h - radii[i])
        return np.column_stack([pos_r[:, 0], pos_r[:, 1], radii]), best_score
    else:
        return None, -1

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit
    
    best_score = 0
    best_result = None
    
    # Perimeter 4 => w + h = 2.
    # w in [0.5, 1.0] due to symmetry.
    # Sample aspect ratios
    w_values = np.linspace(0.5, 1.0, 10)
    
    for w in w_values:
        if time.time() > deadline:
            break
        h = 2.0 - w
        result, score = _pack_qcp(w, h, n_restarts=3, max_iter=50)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
            
    # Fallback to seed if nothing works
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
