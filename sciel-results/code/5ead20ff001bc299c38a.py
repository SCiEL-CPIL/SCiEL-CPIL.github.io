import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial guess: Grid layout
    # 5x5 grid is 25, so we need 26. 
    # Let's use a 5x5 grid for the first 25, and place the 26th in a gap or just start with 26 in a distorted grid.
    # Actually, a 5x5 grid of 25 circles leaves space. 
    # Let's just generate 26 circles in a grid pattern: 5 rows, 6 cols (30 spots, take 26).
    # Or 5x5 + 1.
    # Let's use a 5x5 grid for 25, and the 26th placed in the center of one cell? No, that overlaps.
    # Let's just use a 5x6 grid and pick 26.
    
    def get_grid_init(seed):
        rng = np.random.RandomState(seed)
        # 5 rows, 6 cols
        rows, cols = 5, 6
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                r = min(0.5/cols, 0.5/rows) * 0.9 # Slightly smaller to allow movement
                pts.append([x, y, r])
        # Shuffle and take 26
        rng.shuffle(pts)
        pts = pts[:N]
        # Convert to flat vector [x1..xN, y1..yN, r1..rN]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = np.array([p[2] for p in pts])
        return np.concatenate([x, y, r])

    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return -np.sum(r)

    def constraints(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        cons = []
        
        # Boundary constraints
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, r=r: x - r})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, x=x, r=r: 1 - x - r})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, y=y, r=r: y - r})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, y=y, r=r: 1 - y - r})
        # r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, r=r: r})
        
        # Non-overlap constraints
        # dist(i,j) - r_i - r_j >= 0
        # Using vectorized calculation for efficiency in the lambda might be tricky with SLSQP's scalar calls, 
        # but we can compute the matrix of distances.
        # SLSQP calls the function with a single vector.
        
        def non_overlap(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            # Calculate all pairwise distances
            # dx[i,j] = x[i] - x[j]
            dx = np.subtract.outer(x, x)
            dy = np.subtract.outer(y, y)
            dist = np.sqrt(dx**2 + dy**2)
            
            # r_i + r_j matrix
            r_sum = np.add.outer(r, r)
            
            # We need dist[i,j] - r_sum[i,j] >= 0 for i != j
            # We can flatten the upper triangle
            iu = np.triu_indices(N, k=1)
            return dist[iu] - r_sum[iu]
        
        cons.append({'type': 'ineq', 'fun': non_overlap})
        
        return cons

    # Bounds
    bounds = [(0, 1) for _ in range(N)] + [(0, 1) for _ in range(N)] + [(0, 0.5) for _ in range(N)]
    
    # Run multiple restarts
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        v0 = get_grid_init(i)
        
        # Add some noise to v0
        # v0 = v0 + np.random.normal(0, 0.05, v0.shape)
        # Ensure v0 is within bounds roughly
        v0 = np.clip(v0, 0.01, 0.99)
        # Radii should be positive
        v0[2*N:] = np.clip(v0[2*N:], 0.01, 0.4)
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
            
            if res.success:
                # Verify validity
                x = res.x[:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                
                # Check boundaries
                if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6) or np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6) or np.any(r < -1e-6):
                    continue
                
                # Check overlaps
                valid = True
                for a in range(N):
                    for b in range(a+1, N):
                        d = np.sqrt((x[a]-x[b])**2 + (y[a]-y[b])**2)
                        if d < r[a] + r[b] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x, y, r))
                        
        except Exception as e:
            continue
            
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
