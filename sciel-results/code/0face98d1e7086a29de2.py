import numpy as np
from scipy.optimize import minimize

N = 21

def construct():
    # Initial grid in 1x1 square
    side = int(np.ceil(np.sqrt(N)))
    x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
    y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
    r = np.full(N, 0.5 / side - 1e-4)
    W = 1.0
    H = 1.0
    
    # Combine into vector: [x0, y0, r0, x1, y1, r1, ..., W, H]
    # Actually, let's fix W=1, H=1 for now to simplify constraints, 
    # or allow them to vary? 
    # Perimeter 4 => W+H=2. 
    # Let's optimize W and H too.
    
    # Variables: x (N), y (N), r (N), W, H
    # Total 3N + 2 variables
    x0 = np.concatenate([x, y, r, [W, H]])
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # Penalize negative radii
        r_pos = np.maximum(r, 0)
        
        # Overlaps
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r_pos[i] + r_pos[j]
                if dist_sq < rad_sum**2:
                    # Penalty for overlap
                    penalty += (rad_sum**2 - dist_sq)
        
        # Bounds
        for i in range(N):
            if x[i] < r_pos[i]:
                penalty += (r_pos[i] - x[i])**2
            if x[i] > W - r_pos[i]:
                penalty += (x[i] - (W - r_pos[i]))**2
            if y[i] < r_pos[i]:
                penalty += (r_pos[i] - y[i])**2
            if y[i] > H - r_pos[i]:
                penalty += (y[i] - (H - r_pos[i]))**2
        
        # Perimeter constraint penalty
        perim = 2 * (W + H)
        penalty += (perim - 4)**2
        
        return -np.sum(r_pos) + 100 * penalty

    # Run optimization
    # Use Nelder-Mead or L-BFGS-B? 
    # L-BFGS-B is good for box constraints, but we have complex inequalities.
    # SLSQP is better for general constraints but slower.
    # Let's try a simpler heuristic first: Greedy expansion.
    
    # --- Greedy Expansion Phase ---
    # This is faster and often sufficient.
    # We will perform a few passes of greedy radius increase.
    
    def is_valid(x, y, r, W, H):
        if W <= 0 or H <= 0:
            return False
        if abs(2*(W+H) - 4) > 1e-6:
            return False
        for i in range(N):
            if r[i] <= 0:
                return False
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6:
                return False
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                return False
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-6:
                    return False
        return True

    # Start with grid
    x_curr = x.copy()
    y_curr = y.copy()
    r_curr = r.copy()
    W_curr = 1.0
    H_curr = 1.0
    
    step = 0.005
    for iteration in range(100):
        improved = False
        for i in range(N):
            # Try to increase r[i]
            new_r = r_curr[i] + step
            # Check validity
            if new_r <= 0:
                continue
            # Check bounds for i
            if x_curr[i] - new_r < 0 or x_curr[i] + new_r > W_curr:
                continue
            if y_curr[i] - new_r < 0 or y_curr[i] + new_r > H_curr:
                continue
            # Check overlap with others
            valid = True
            for j in range(N):
                if i == j:
                    continue
                dist_sq = (x_curr[i]-x_curr[j])**2 + (y_curr[i]-y_curr[j])**2
                rad_sum = new_r + r_curr[j]
                if dist_sq < rad_sum**2:
                    valid = False
                    break
            if valid:
                r_curr[i] = new_r
                improved = True
        if not improved:
            # Try smaller step?
            step *= 0.9
            if step < 1e-5:
                break

    # --- Local Optimization Phase ---
    # Use SLSQP to refine positions and radii
    def obj(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        
        # Objective: Maximize sum of r
        return -np.sum(r)
        
    def jac(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    constraints = []
    
    # Perimeter constraint
    constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
    
    # Non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
            })
            
    # Boundary constraints
    for i in range(N):
        # x_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
        # x_i + r_i <= W
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
        # y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
        # y_i + r_i <= H
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
        
    # Radii > 0
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
        
    # W, H > 0
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
    constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})

    x_start = np.concatenate([x_curr, y_curr, r_curr, [W_curr, H_curr]])
    
    res = minimize(obj, x_start, jac=jac, method='SLSQP', constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
    
    if res.success:
        x_final = res.x[:N]
        y_final = res.x[N:2*N]
        r_final = res.x[2*N:3*N]
        W_final = res.x[3*N]
        H_final = res.x[3*N+1]
    else:
        x_final = x_curr
        y_final = y_curr
        r_final = r_curr
        W_final = W_curr
        H_final = H_curr

    # Ensure validity before returning
    if not is_valid(x_final, y_final, r_final, W_final, H_final):
        # Fall back to greedy result
        x_final = x_curr
        y_final = y_curr
        r_final = r_curr
        W_final = W_curr
        H_final = H_curr
        
    # Scale to ensure perimeter is exactly 4 if there are floating point errors
    # The optimizer should handle this, but let's check.
    perim = 2 * (W_final + H_final)
    if perim > 4:
        scale = 4.0 / perim
        x_final *= scale
        y_final *= scale
        r_final *= scale
        W_final *= scale
        H_final *= scale

    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
