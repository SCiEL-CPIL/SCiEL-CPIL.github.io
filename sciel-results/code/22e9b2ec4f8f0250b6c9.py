import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_r = 0.0
    best_x = None
    best_y = None
    best_W = 1.0
    best_H = 1.0

    # Helper to unpack variables
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N]
        W = v[2*N+1]
        H = 2.0 - W
        return x, y, r, W, H

    # Objective function
    def objective(v):
        x, y, r, W, H = unpack(v)
        return -r

    # Constraints
    def con_bounds(v):
        x, y, r, W, H = unpack(v)
        cons = []
        # x bounds
        cons.append(x - r)
        cons.append((W - r) - x)
        # y bounds
        cons.append(y - r)
        cons.append((H - r) - y)
        # W, H, r positive
        cons.append(r)
        cons.append(W)
        cons.append(H)
        # Pairwise distances
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (2*r)**2
                cons.append(dist_sq - min_dist_sq)
        return np.concatenate(cons)

    constraints = [{'type': 'ineq', 'fun': con_bounds}]

    # Initial guesses
    def make_hex_initial():
        # Approximate hex packing
        # 5, 4, 5, 4, 3
        rows = [5, 4, 5, 4, 3]
        x_list = []
        y_list = []
        # Assume square-ish container initially
        # Just place them in a grid-like hex pattern
        # Let's use a simple grid for init, it's robust
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(x_list) < N:
                    x_list.append((i + 0.5) / side)
                    y_list.append((j + 0.5) / side)
        x0 = np.array(x_list[:N])
        y0 = np.array(y_list[:N])
        r0 = 0.1
        W0 = 1.0
        v0 = np.concatenate([x0, y0, [r0, W0]])
        return v0

    def make_random_initial(seed):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0.1, 1.9, N)
        y0 = rng.uniform(0.1, 1.9, N)
        r0 = 0.05
        W0 = 1.0
        v0 = np.concatenate([x0, y0, [r0, W0]])
        return v0

    inits = [make_hex_initial()]
    for s in range(10):
        inits.append(make_random_initial(s))

    for i, v0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=[(0, 2)]*N + [(0, 2)]*N + [(0.01, 1.0)] + [(0.5, 1.5)], constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success:
                x, y, r, W, H = unpack(res.x)
                # Verify constraints manually to be safe
                valid = True
                if r <= 0 or W <= 0 or H <= 0:
                    valid = False
                for k in range(N):
                    if x[k] < r or x[k] > W - r or y[k] < r or y[k] > H - r:
                        valid = False
                        break
                if valid:
                    for k in range(N):
                        for l in range(k+1, N):
                            if (x[k]-x[l])**2 + (y[k]-y[l])**2 < (2*r)**2 - 1e-9:
                                valid = False
                                break
                        if not valid: break
                
                if valid and r > best_r:
                    best_r = r
                    best_x = x
                    best_y = y
                    best_W = W
                    best_H = H
        except Exception:
            continue

    if best_x is not None:
        # Construct the output array
        # Format: (x, y, r)
        result = np.column_stack((best_x, best_y, np.full(N, best_r)))
        return result
    else:
        # Fallback to the simple grid if optimization fails completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
