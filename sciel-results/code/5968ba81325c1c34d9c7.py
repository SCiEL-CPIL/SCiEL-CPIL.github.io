"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def bounds_and_constraints(x, y, r):
        # Returns list of constraint dicts for scipy
        # Constraint: x_i >= r_i  => x_i - r_i >= 0
        # Constraint: x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        # ...
        # Constraint: dist(i,j) >= r_i + r_j
        # We use squared distance to avoid sqrt in gradient? 
        # No, sqrt is fine, but squared is smoother at 0? 
        # Actually, distance - (ri+rj) is standard.
        
        constraints = []
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2], 'jac': lambda v, i=i: np.array([1.0, 0.0, -1.0])}) # x - r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2], 'jac': lambda v, i=i: np.array([-1.0, 0.0, -1.0])}) # 1-x-r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2], 'jac': lambda v, i=i: np.array([0.0, 1.0, -1.0])}) # y - r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2], 'jac': lambda v, i=i: np.array([0.0, -1.0, -1.0])}) # 1-y-r
            
        for i in range(N):
            for j in range(i+1, N):
                def fun_ij(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    return np.sqrt((xi-xj)**2 + (yi-yj)**2) - (ri + rj)
                
                def jac_ij(v, i=i, j=j):
                    grad = np.zeros(3*N)
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        dist = 1e-9
                    inv_dist = 1.0 / dist
                    grad[3*i] = dx * inv_dist
                    grad[3*i+1] = dy * inv_dist
                    grad[3*i+2] = -1.0
                    grad[3*j] = -dx * inv_dist
                    grad[3*j+1] = -dy * inv_dist
                    grad[3*j+2] = -1.0
                    return grad
                
                constraints.append({'type': 'ineq', 'fun': fun_ij, 'jac': jac_ij})
        return constraints

    def objective(v):
        r = v[2::3]
        return -np.sum(r)

    def is_valid(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        if np.any(r <= 0): return False
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5): return False
        if np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5): return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-5:
                    return False
        return True

    best_score = -np.inf
    best_v = None
    rng_global = np.random.default_rng(42)

    # Grid initialization helper
    def get_grid_init(seed):
        rng = np.random.default_rng(seed)
        # 5x6 grid or 6x5? 26 circles.
        # 5 cols, 6 rows -> 30 spots. Use first 26.
        # Or 6x5.
        side_x = 5
        side_y = 6
        total = side_x * side_y
        # Create grid points
        xs = (np.arange(side_x) + 0.5) / side_x
        ys = (np.arange(side_y) + 0.5) / side_y
        X, Y = np.meshgrid(xs, ys)
        X = X.flatten()
        Y = Y.flatten()
        
        # Select 26 random points from the 30 grid points to add variety
        idx = rng.choice(total, N, replace=False)
        x_init = X[idx]
        y_init = Y[idx]
        
        # Initial radii: small enough to not overlap
        # Max dist between neighbors is min(1/5, 1/6) = 1/6.
        # Half of that is 1/12 ~ 0.083
        # Let's use 0.05 to be safe
        r_init = np.full(N, 0.05)
        
        v = np.zeros(3*N)
        v[0::3] = x_init
        v[1::3] = y_init
        v[2::3] = r_init
        return v

    # Run optimization
    num_restarts = 0
    while time.time() < deadline:
        seed = num_restarts
        v0 = get_grid_init(seed)
        
        # Add small noise to positions to break symmetry
        v0[0::3] += rng_global.uniform(-0.01, 0.01, N)
        v0[1::3] += rng_global.uniform(-0.01, 0.01, N)
        
        # Ensure initial validity
        if not is_valid(v0):
            # If invalid, shrink radii
            v0[2::3] *= 0.5
            if not is_valid(v0):
                num_restarts += 1
                continue

        constraints = bounds_and_constraints(None, None, None)
        
        try:
            res = minimize(objective, v0, method='SLSQP', jac=None, 
                           constraints=constraints, 
                           bounds=[(1e-5, 1.0)] * (3*N),
                           options={'maxiter': 200, 'ftol': 1e-8})
        except Exception:
            pass
        else:
            if res.success or res.status == 0:
                v_res = res.x
                if is_valid(v_res):
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_v = v_res.copy()
        
        num_restarts += 1
        # Check time
        if time.time() > deadline - 10:
            break

    if best_v is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Reshape best_v to (N, 3)
    return best_v.reshape(N, 3)
# EVOLVE-BLOCK-END
