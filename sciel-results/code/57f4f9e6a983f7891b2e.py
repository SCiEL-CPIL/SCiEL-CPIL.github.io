import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # Helper to generate a hexagonal grid
    def hex_grid(n):
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        # Hexagonal packing
        # Rows offset
        for row in range(side + 1):
            for col in range(side + 1):
                x = col + 0.5 * (row % 2)
                y = row * 0.866 # sqrt(3)/2
                pts.append((x, y))
        # Sort by y then x to make it deterministic? Or just take first N
        # We need to scale this to fit in 1x1 or similar
        # Let's just use a standard hex pattern for first N points
        x_coords = []
        y_coords = []
        for i in range(n):
            row = i // side
            col = i % side
            x = col + 0.5 * (row % 2)
            y = row * 0.866
            x_coords.append(x)
            y_coords.append(y)
        
        # Normalize to 1x1 box roughly
        x_max = max(x_coords)
        y_max = max(y_coords)
        x_norm = np.array(x_coords) / (x_max + 1) # +1 to add margin
        y_norm = np.array(y_coords) / (y_max + 1)
        return x_norm, y_norm

    # Helper to generate a square grid
    def square_grid(n):
        side = int(np.ceil(np.sqrt(n)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:n])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:n])
        return x, y

    def evaluate_config(x, y, r, W, H):
        # Returns sum of radii if valid, else -inf
        if W <= 0 or H <= 0:
            return -np.inf
        if abs(2*(W+H) - 4) > 1e-6:
            return -np.inf
        for i in range(N):
            if r[i] <= 0:
                return -np.inf
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6:
                return -np.inf
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                return -np.inf
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-6:
                    return -np.inf
        return np.sum(r)

    def greedy_expand(x, y, r, W, H):
        # Simple greedy radius expansion
        step = 0.01
        for iteration in range(50):
            improved = False
            for i in range(N):
                new_r = r[i] + step
                if new_r <= 0:
                    continue
                if x[i] - new_r < 0 or x[i] + new_r > W:
                    continue
                if y[i] - new_r < 0 or y[i] + new_r > H:
                    continue
                valid = True
                for j in range(N):
                    if i == j: continue
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    rad_sum = new_r + r[j]
                    if dist_sq < rad_sum**2:
                        valid = False
                        break
                if valid:
                    r[i] = new_r
                    improved = True
            if not improved:
                step *= 0.8
                if step < 1e-4:
                    break
        return r

    def obj(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        W = v[3*N]
        H = v[3*N+1]
        return -np.sum(r)
        
    def jac(v):
        g = np.zeros_like(v)
        g[2*N:3*N] = -1.0
        return g
        
    def get_constraints():
        constraints = []
        # Perimeter constraint
        constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
        
        # Circle-circle non-overlap
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
                
        # Circle-boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
            
        # Rectangle positivity
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})
        
        return constraints

    constraints = get_constraints()

    def run_optimization(x_start, y_start, r_start, W_start, H_start):
        v_start = np.concatenate([x_start, y_start, r_start, [W_start, H_start]])
        try:
            res = minimize(obj, v_start, jac=jac, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 300, 'ftol': 1e-9})
            if res.success:
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = res.x[3*N+1]
                score = evaluate_config(x_final, y_final, r_final, W_final, H_final)
                if score > -np.inf:
                    return x_final, y_final, r_final, W_final, H_final, score
            else:
                # Even if not success, check if the result is valid and better
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = res.x[3*N+1]
                score = evaluate_config(x_final, y_final, r_final, W_final, H_final)
                if score > -np.inf:
                    return x_final, y_final, r_final, W_final, H_final, score
        except Exception:
            pass
        return None

    # Generate diverse starting points
    starts = []
    
    # 1. Square grid, W=1, H=1
    x_sq, y_sq = square_grid(N)
    r_sq = np.full(N, 0.05)
    W_sq, H_sq = 1.0, 1.0
    r_sq = greedy_expand(x_sq, y_sq, r_sq, W_sq, H_sq)
    starts.append((x_sq, y_sq, r_sq, W_sq, H_sq))

    # 2. Hex grid, W=1, H=1
    x_hex, y_hex = hex_grid(N)
    r_hex = np.full(N, 0.05)
    W_hex, H_hex = 1.0, 1.0
    r_hex = greedy_expand(x_hex, y_hex, r_hex, W_hex, H_hex)
    starts.append((x_hex, y_hex, r_hex, W_hex, H_hex))

    # 3. Random jittered square, W=1, H=1
    x_rand = x_sq + np.random.normal(0, 0.05, N)
    y_rand = y_sq + np.random.normal(0, 0.05, N)
    r_rand = np.full(N, 0.05)
    W_rand, H_rand = 1.0, 1.0
    r_rand = greedy_expand(x_rand, y_rand, r_rand, W_rand, H_rand)
    starts.append((x_rand, y_rand, r_rand, W_rand, H_rand))
    
    # 4. Rectangular aspect ratios
    # Perimeter 4 => 2(W+H)=4 => W+H=2.
    # Let W = 1 + d, H = 1 - d.
    for d in [0.2, -0.2, 0.4, -0.4]:
        W_rect = 1.0 + d
        H_rect = 1.0 - d
        if W_rect <= 0 or H_rect <= 0: continue
        # Scale square grid to fit W_rect x H_rect
        x_rect = x_sq * W_rect
        y_rect = y_sq * H_rect
        r_rect = np.full(N, 0.05)
        r_rect = greedy_expand(x_rect, y_rect, r_rect, W_rect, H_rect)
        starts.append((x_rect, y_rect, r_rect, W_rect, H_rect))

    best_score = -np.inf
    best_result = None
    
    for s in starts:
        x_s, y_s, r_s, W_s, H_s = s
        res = run_optimization(x_s, y_s, r_s, W_s, H_s)
        if res:
            score = res[5]
            if score > best_score:
                best_score = score
                best_result = res

    if best_result is None:
        # Fallback to a very simple valid config
        side = int(np.ceil(np.sqrt(N)))
        x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
        y = np.array([(j + 0.5) / side for _ in range(side) for j in range(side)][:N])
        r = np.full(N, 0.5 / side - 0.01)
        W, H = 1.0, 1.0
        return np.column_stack([x, y, r])
        
    x_final, y_final, r_final, W_final, H_final, _ = best_result
    result = np.zeros((N, 3))
    result[:, 0] = x_final
    result[:, 1] = y_final
    result[:, 2] = r_final
    
    return result
