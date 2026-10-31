import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s safety margin for overhead
    
    best_score = -np.inf
    best_result = None
    
    # We will try multiple restarts with different initializations
    # Variables: x[0..N-1], y[0..N-1], r[0..N-1], W, H
    # Total variables: 3*N + 2
    
    def constraints_def(x):
        # x is flattened array [x0..xN-1, y0..yN-1, r0..rN-1, W, H]
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]
        
        cons = []
        
        # Perimeter constraint: 2(W+H) <= 4  =>  W+H <= 2
        # We want to maximize sum(r), so we should push W+H to 2.
        # Inequality constraint for SLSQP: func(x) >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: 2 - (x[3*N] + x[3*N+1])})
        
        # Boundary constraints for each circle i
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i]})
            # W - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N] - x[i] - x[2*N+i]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i]})
            # H - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*N+1] - x[N+i] - x[2*N+i]})
            # r_i >= 0 (small positive to avoid degenerate)
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N+i] - 1e-6})
            
        # Non-overlap constraints for each pair i, j
        for i in range(N):
            for j in range(i+1, N):
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: 
                        (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2
                })
                
        return cons

    def objective(x):
        # Maximize sum of radii -> Minimize -sum(r)
        rs = x[2*N:3*N]
        return -np.sum(rs)

    def is_valid(x):
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]
        
        if W <= 0 or H <= 0:
            return False
        if 2*(W+H) > 4 + 1e-5:
            return False
        
        for i in range(N):
            if xs[i] < rs[i] - 1e-5 or xs[i] > W - rs[i] + 1e-5:
                return False
            if ys[i] < rs[i] - 1e-5 or ys[i] > H - rs[i] + 1e-5:
                return False
            if rs[i] < 1e-6:
                return False
                
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum_sq = (rs[i]+rs[j])**2
                if dist_sq < rad_sum_sq - 1e-5:
                    return False
        return True

    def get_score(x):
        if not is_valid(x):
            return -np.inf
        rs = x[2*N:3*N]
        return np.sum(rs)

    # Initializations
    def hex_init(W, H):
        # Generate hexagonal packing in W x H
        # Approximate radius for hex packing in square area A: r ~ sqrt(A/N) * constant
        # Let's just place them in a grid and perturb
        # We want to fill W x H
        # Use a simple grid for initialization, then let optimizer work
        side = int(np.ceil(np.sqrt(N)))
        # Scale to fit W, H
        # Center of cells
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Simple grid placement
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        dx = W / cols
        dy = H / rows
        
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N: break
                xs[k] = (i + 0.5) * dx
                ys[k] = (j + 0.5) * dy
                # Initial radius estimate
                # Max possible radius in this cell is min(dx, dy)/2
                # But we need to account for neighbors.
                # Start with a small radius, optimizer will grow them
                rs[k] = 0.5 * min(dx, dy) * 0.8
                k += 1
                
        return np.concatenate([xs, ys, rs, [W, H]])

    def random_init(W, H):
        xs = np.random.uniform(0.1, W-0.1, N)
        ys = np.random.uniform(0.1, H-0.1, N)
        rs = np.random.uniform(0.05, 0.2, N)
        return np.concatenate([xs, ys, rs, [W, H]])

    attempts = 0
    max_attempts = 50 # Cap to ensure we don't loop forever if one is very slow
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Vary aspect ratio
        # W + H = 2
        # W in [0.5, 1.5] roughly (avoid too extreme)
        W = np.random.uniform(0.8, 1.2)
        H = 2.0 - W
        
        # Choose initialization strategy
        if attempts % 3 == 0:
            x0 = random_init(W, H)
        else:
            x0 = hex_init(W, H)
            # Add small noise to escape symmetry
            x0[:3*N] += np.random.normal(0, 0.05, 3*N)
            
        # Ensure x0 is roughly valid for start (clip radii if they are too big for box)
        # If r > W/2 or r > H/2, it's invalid immediately.
        # Let's just run it. SLSQP will try to fix.
        
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           bounds=[(1e-4, None)] * (3*N) + [(0.5, 1.5), (0.5, 1.5)],
                           constraints=constraints_def,
                           options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success:
                x_opt = res.x
                score = get_score(x_opt)
                if score > best_score:
                    best_score = score
                    best_result = x_opt
            else:
                # Sometimes SLSQP fails but finds a good point
                x_opt = res.x
                score = get_score(x_opt)
                if score > best_score:
                    best_score = score
                    best_result = x_opt
                    
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed-like solution if optimization failed completely
        # Grid in 1x1 square (perimeter 4)
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Extract final result
    xs = best_result[:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    
    # Post-processing: Ensure strict validity by shrinking radii slightly if needed
    # The optimizer might leave them touching exactly.
    # We can try to find a global scale factor or just return as is if valid.
    # The check is_valid uses 1e-5 tolerance.
    
    # If the result is valid, return it.
    # The evaluator expects array of shape (n, 3): [x, y, r]
    result = np.column_stack((xs, ys, rs))
    
    return result
