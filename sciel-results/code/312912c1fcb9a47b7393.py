import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = -np.inf
    best_sol = None
    
    num_restarts = 8
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        x0 = generate_initial_guess(r)
        
        # Optimize positions and radii
        res = optimize_slsqp(x0)
        
        if res is not None:
            # Extract positions and radii
            # x0 format: [x1, y1, r1, ..., xN, yN, rN]
            pos = res.x[:2*N].reshape(N, 2)
            rad = res.x[2*N:]
            
            # Ensure radii are positive
            rad = np.maximum(rad, 1e-6)
            
            # LP Radius Growth
            rad_lp = optimize_radii_lp(pos)
            
            # Calculate value
            val = np.sum(rad_lp)
            
            if val > best_val:
                best_val = val
                best_sol = np.column_stack((pos, rad_lp))
                
    if best_sol is None:
        return trivial_solution()
        
    return validate_and_return(best_sol)

def generate_initial_guess(idx):
    # Different initialization strategies
    if idx % 3 == 0:
        # 5x6 grid
        rows, cols = 5, 6
        x_coords = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
        y_coords = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
        xs, ys = np.meshgrid(x_coords, y_coords)
        xs = xs.flatten()
        ys = ys.flatten()
        # Take first 26
        idxs = np.arange(N)
    elif idx % 3 == 1:
        # 6x5 grid
        rows, cols = 6, 5
        x_coords = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
        y_coords = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
        xs, ys = np.meshgrid(x_coords, y_coords)
        xs = xs.flatten()
        ys = ys.flatten()
        idxs = np.arange(N)
    else:
        # Random
        xs = np.random.rand(N) * 0.8 + 0.1
        ys = np.random.rand(N) * 0.8 + 0.1
        idxs = np.arange(N)
        
    xs = xs[idxs]
    ys = ys[idxs]
    
    # Add jitter
    xs += np.random.rand(N) * 0.02 - 0.01
    ys += np.random.rand(N) * 0.02 - 0.01
    
    # Initial radii: small
    # Estimate max radius based on grid size
    if idx % 3 == 0:
        r_init = 0.5 / 6 * 0.9
    elif idx % 3 == 1:
        r_init = 0.5 / 5 * 0.9
    else:
        r_init = 0.02
        
    rs = np.full(N, r_init)
    
    # Flatten to vector
    x0 = np.concatenate([xs, ys, rs])
    return x0

def objective(x):
    r = x[2*N:]
    return -np.sum(r)

def objective_grad(x):
    g = np.zeros(3*N)
    g[2*N:] = -1.0
    return g

def constraints(x):
    # Returns list of constraints
    cons = []
    
    # Boundary constraints
    for i in range(N):
        xi = x[i]
        yi = x[N + i]
        ri = x[2*N + i]
        
        # xi >= ri
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N + i], 'jac': lambda x, i=i: np.array([1 if k==i else (-1 if k==2*N+i else 0) for k in range(3*N)])})
        # yi >= ri
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[2*N + i], 'jac': lambda x, i=i: np.array([1 if k==N+i else (-1 if k==2*N+i else 0) for k in range(3*N)])})
        # 1 - xi >= ri
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*N + i], 'jac': lambda x, i=i: np.array([-1 if k==i else (-1 if k==2*N+i else 0) for k in range(3*N)])})
        # 1 - yi >= ri
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[N + i] - x[2*N + i], 'jac': lambda x, i=i: np.array([-1 if k==N+i else (-1 if k==2*N+i else 0) for k in range(3*N)])})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            xi = x[i]
            yi = x[N + i]
            ri = x[2*N + i]
            xj = x[j]
            yj = x[N + j]
            rj = x[2*N + j]
            
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r = ri + rj
            
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[i]-x[j])**2 + (x[N+i]-x[N+j])**2 - (x[2*N+i]+x[2*N+j])**2,
                'jac': lambda x, i=i, j=j: compute_overlap_jac(x, i, j)
            })
            
    return cons

def compute_overlap_jac(x, i, j):
    jac = np.zeros(3*N)
    dx = x[i] - x[j]
    dy = x[N+i] - x[N+j]
    sum_r = x[2*N+i] + x[2*N+j]
    
    jac[i] = 2 * dx
    jac[j] = -2 * dx
    jac[N+i] = 2 * dy
    jac[N+j] = -2 * dy
    jac[2*N+i] = -2 * sum_r
    jac[2*N+j] = -2 * sum_r
    return jac

def optimize_slsqp(x0):
    try:
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, constraints=constraints(x0), 
                       options={'maxiter': 100, 'ftol': 1e-6})
        return res
    except Exception as e:
        return None

def optimize_radii_lp(pos):
    # pos: (N, 2)
    # Maximize sum(r)
    # Minimize -sum(r)
    c = -np.ones(N)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: ri + rj <= dij
    for i in range(N):
        for j in range(i+1, N):
            dij = np.linalg.norm(pos[i] - pos[j])
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dij)
            
    # Boundary constraints: ri <= min(xi, yi, 1-xi, 1-yi)
    for i in range(N):
        bi = np.min([pos[i][0], pos[i][1], 1-pos[i][0], 1-pos[i][1]])
        row = np.zeros(N)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(bi)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback to simple min distance
            return np.array([np.min([pos[i][0], pos[i][1], 1-pos[i][0], 1-pos[i][1]]) for i in range(N)])
    except Exception:
        return np.array([np.min([pos[i][0], pos[i][1], 1-pos[i][0], 1-pos[i][1]]) for i in range(N)])

def trivial_solution():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def validate_and_return(sol):
    # sol: (N, 3)
    x = sol[:, 0]
    y = sol[:, 1]
    r = sol[:, 2]
    
    # Clip positions
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    
    # Ensure radii are non-negative and circles are inside
    # r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    r_max = np.minimum(np.minimum(x, y), np.minimum(1-x, 1-y))
    r = np.minimum(r, r_max)
    r = np.maximum(r, 0)
    
    # Check overlaps and shrink if necessary
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(sol[i, :2] - sol[j, :2])
            sum_r = r[i] + r[j]
            if sum_r > dist:
                # Scale down both
                scale = dist / sum_r * 0.999
                r[i] *= scale
                r[j] *= scale
    
    return np.column_stack((x, y, r))
