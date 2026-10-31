import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    best_result = None
    best_score = -np.inf
    
    # Objective: minimize -sum(r)
    def objective(vars):
        # vars: [x0, y0, r0, x1, y1, r1, ...]
        rs = vars[2::3]
        return -np.sum(rs)
    
    # Constraints
    def make_constraints():
        cons = []
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        for i in range(N):
            xi = 3*i
            yi = 3*i + 1
            ri = 3*i + 2
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[xi] - v[ri]})
            # 1 - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: 1 - v[xi] - v[ri]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[yi] - v[ri]})
            # 1 - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: 1 - v[yi] - v[ri]})
            # r_i >= 0 (slightly positive to avoid degeneracy)
            cons.append({'type': 'ineq', 'fun': lambda v, ri=ri: v[ri]})
            
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                xi = 3*i
                yi = 3*i + 1
                ri = 3*i + 2
                xj = 3*j
                yj = 3*j + 1
                rj = 3*j + 2
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                            (v[xi] - v[xj])**2 + (v[yi] - v[yj])**2 - (v[ri] + v[rj])**2
                })
        return cons

    constraints = make_constraints()
    
    # Helper to check validity and calculate score
    def check_and_score(arr):
        # arr: (N, 3)
        xs = arr[:, 0]
        ys = arr[:, 1]
        rs = arr[:, 2]
        
        # Check bounds
        if np.any(xs < rs - 1e-9) or np.any(1 - xs < rs - 1e-9) or np.any(ys < rs - 1e-9) or np.any(1 - ys < rs - 1e-9):
            return -np.inf
        if np.any(rs <= 1e-9):
            return -np.inf
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                sum_r_sq = (rs[i] + rs[j])**2
                if dist_sq < sum_r_sq - 1e-9:
                    return -np.inf
        
        return np.sum(rs)

    # Generate initial points
    
    # 1. Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) for i in range(side) for j in range(side)]
    initial_grid = np.array(grid_cells[:N])
    x0_grid = initial_grid.flatten()
    
    # 2. Random valid initialization
    def random_valid_init(seed):
        np.random.seed(seed)
        points = []
        # Try to place N circles with small radii
        for _ in range(N):
            placed = False
            for attempt in range(20):
                # Start with larger radius, shrink if needed
                r = 0.05 if attempt == 0 else 0.02
                x = np.random.uniform(r, 1-r)
                y = np.random.uniform(r, 1-r)
                
                # Check against existing points
                valid = True
                for px, py, pr in points:
                    if (x-px)**2 + (y-py)**2 < (r+pr)**2:
                        valid = False
                        break
                if valid:
                    points.append((x, y, r))
                    placed = True
                    break
            if not placed:
                # If we can't place it, we can't use this init
                return None
        return np.array(points).flatten()

    inits = [x0_grid]
    # Try a few random inits
    for s in range(5, 15):
        if time.time() > deadline:
            break
        r_init = random_valid_init(s)
        if r_init is not None:
            inits.append(r_init)

    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Run SLSQP
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                arr = res.x.reshape(N, 3)
                score = check_and_score(arr)
                if score > best_score:
                    best_score = score
                    best_result = arr
        except Exception as e:
            pass

    # If no valid result found, return grid
    if best_result is None:
        return initial_grid
    
    return best_result
