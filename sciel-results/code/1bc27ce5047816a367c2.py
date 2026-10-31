import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint

N = 26
EPS = 1e-5

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_result = None
    
    # Pre-calculate constraint indices for non-overlap
    # There are N*(N-1)/2 pairs
    n_pairs = N * (N - 1) // 2
    pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Total number of constraints:
    # 4 per circle for boundaries (x-r, 1-x-r, y-r, 1-y-r)
    # 1 per circle for radius positivity (r - EPS)
    # 1 per pair for non-overlap
    n_boundary = 4 * N
    n_radius = N
    n_non_overlap = n_pairs
    n_constraints = n_boundary + n_radius + n_non_overlap
    
    def objective(v):
        # v layout: x (N), y (N), r (N)
        r = v[2*N:]
        return -np.sum(r)
        
    def objective_grad(v):
        grad = np.zeros(3*N)
        grad[2*N:] = -1.0
        return grad

    def constraints_func(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        c = np.zeros(n_constraints)
        
        # Boundary constraints:
        # x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        idx = 0
        for i in range(N):
            c[idx] = x[i] - r[i]
            idx += 1
            c[idx] = 1.0 - x[i] - r[i]
            idx += 1
            c[idx] = y[i] - r[i]
            idx += 1
            c[idx] = 1.0 - y[i] - r[i]
            idx += 1
            
        # Radius positivity: r - EPS >= 0
        for i in range(N):
            c[idx] = r[i] - EPS
            idx += 1
            
        # Non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i, j in pair_indices:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            c[idx] = dx*dx + dy*dy - dr*dr
            idx += 1
            
        return c

    def constraints_jac(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:]
        
        # Shape: (n_constraints, 3*N)
        jac = np.zeros((n_constraints, 3*N))
        
        # Boundary constraints gradients
        idx = 0
        for i in range(N):
            # x - r >= 0
            jac[idx, i] = 1.0
            jac[idx, 2*N + i] = -1.0
            idx += 1
            
            # 1 - x - r >= 0
            jac[idx, i] = -1.0
            jac[idx, 2*N + i] = -1.0
            idx += 1
            
            # y - r >= 0
            jac[idx, N + i] = 1.0
            jac[idx, 2*N + i] = -1.0
            idx += 1
            
            # 1 - y - r >= 0
            jac[idx, N + i] = -1.0
            jac[idx, 2*N + i] = -1.0
            idx += 1
            
        # Radius positivity gradients
        for i in range(N):
            jac[idx, 2*N + i] = 1.0
            idx += 1
            
        # Non-overlap gradients
        for i, j in pair_indices:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            
            # d/dxi = 2*dx
            jac[idx, i] = 2.0 * dx
            # d/dxj = -2*dx
            jac[idx, j] = -2.0 * dx
            # d/dyi = 2*dy
            jac[idx, N + i] = 2.0 * dy
            # d/dyj = -2*dy
            jac[idx, N + j] = -2.0 * dy
            # d/dri = -2*dr
            jac[idx, 2*N + i] = -2.0 * dr
            # d/drj = -2*dr
            jac[idx, 2*N + j] = -2.0 * dr
            
            idx += 1
            
        return jac

    # Define the constraint object
    # lb=0, ub=np.inf means c(v) >= 0
    constraint = NonlinearConstraint(constraints_func, 0.0, np.inf, jac=constraints_jac)
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [EPS, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
    for _ in range(N):
        bounds.append((0.0, 1.0)) # y
    for _ in range(N):
        bounds.append((EPS, 0.5)) # r

    n_restarts = 30
    np.random.seed(42) # Fixed seed for reproducibility in restarts, but we vary init
    
    for k in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialization Strategy:
        # Try a mix of random and structured starts.
        # For even k, use a grid-like start.
        # For odd k, use a random start with corner bias.
        
        if k % 2 == 0:
            # Grid-like start
            side = int(np.ceil(np.sqrt(N)))
            # We only have N circles, so we might not fill the whole grid
            # Let's just place them in a grid pattern
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            # Simple grid placement
            for i in range(N):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                # Initial radius small
                r[i] = 0.1 / side
                
            # Add some noise to break symmetry
            x += np.random.uniform(-0.01, 0.01, N)
            y += np.random.uniform(-0.01, 0.01, N)
            
        else:
            # Random with corner bias
            # Bias towards corners: sample from beta distribution or similar
            # Or just random uniform and then push some towards corners
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.02, 0.08, N)
            
            # Push first 4 circles to corners
            corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
            for i in range(4):
                cx, cy = corners[i]
                x[i] = cx + np.random.uniform(-0.05, 0.05)
                y[i] = cy + np.random.uniform(-0.05, 0.05)
                r[i] = np.random.uniform(0.05, 0.15)

        v0 = np.concatenate([x, y, r])
        
        # Ensure v0 is within bounds for the optimizer start
        # trust-constr can handle out-of-bounds starts but it's safer to clip
        v0 = np.clip(v0, [0.0, 0.0, EPS], [1.0, 1.0, 0.5])
        
        try:
            # Use trust-constr
            # maxiter is limited by time, but we also have a global deadline
            # We'll set a reasonable maxiter and rely on the deadline check in the loop
            # However, minimize doesn't have a time limit. We must be careful.
            # We can run it and if it takes too long, we can't easily interrupt it in Python without threads.
            # Instead, we rely on maxiter and maxfun to keep individual calls short.
            # If a call takes too long, we might miss the deadline.
            # To mitigate, we can set strict maxiter/maxfun.
            
            res = minimize(
                objective, v0, 
                method='trust-constr', 
                jac=objective_grad,
                bounds=bounds,
                constraints=[constraint],
                options={
                    'maxiter': 50, # Keep it short per restart
                    'maxfun': 200,
                    'gtol': 1e-4,
                    'xtol': 1e-6
                }
            )
            
            if res.success or (res.fun < objective(v0)):
                v_opt = res.x
                x_opt = v_opt[:N]
                y_opt = v_opt[N:2*N]
                r_opt = v_opt[2*N:]
                
                # Verify validity strictly
                valid = True
                for i in range(N):
                    if x_opt[i] < r_opt[i] - 1e-5 or x_opt[i] > 1 - r_opt[i] + 1e-5 or \
                       y_opt[i] < r_opt[i] - 1e-5 or y_opt[i] > 1 - r_opt[i] + 1e-5:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if d < r_opt[i] + r_opt[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    current_sum = np.sum(r_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
                        
        except Exception as e:
            continue
            
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.4 / side))
        return np.array(cells[:N])
        
    return best_result
