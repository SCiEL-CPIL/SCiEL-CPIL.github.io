import numpy as np
from scipy.optimize import minimize
import time

N = 26

def softplus(x, beta=20):
    """Numerically stable softplus approximation of max(0, x)."""
    # For large x, log(1 + exp(beta*x)) / beta ~ x
    # For small x, it's smooth
    # Using np.where to avoid overflow in exp
    # beta * x can be large. 
    # If beta * x > 20, exp is large, log(exp) is x.
    # If beta * x < -20, exp is small, log(1) is 0.
    # Standard softplus: log(1 + exp(z))
    # Stable: max(z, 0) + log1p(exp(-abs(z)))
    z = beta * x
    return (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # 1. Initial Guess: Hexagonal Packing
    # We want to place 26 circles.
    # A 5x6 hex grid gives 30 spots. We take 26.
    # Let's generate coordinates for a hex lattice.
    # Let initial radius r0 be such that they fit in the square.
    # If we have 5 rows and 6 cols, width ~ 6 * 2r, height ~ 5 * sqrt(3) * r.
    # 12r <= 1 => r <= 1/12 ~ 0.083
    # 5*1.732*r <= 1 => r <= 1/8.66 ~ 0.115
    # So r0 = 0.08 is a safe start.
    
    r0 = 0.08
    dx = 2 * r0
    dy = r0 * np.sqrt(3)
    
    positions = []
    # 5 rows
    for j in range(5):
        for i in range(6):
            x = (i + 0.5 * (j % 2)) * dx
            y = j * dy
            positions.append([x, y])
            
    # Take first N
    pos = np.array(positions[:N])
    
    # Center the points
    pos[:, 0] -= pos[:, 0].mean()
    pos[:, 1] -= pos[:, 1].mean()
    
    # Scale to fit within [0.1, 0.9] roughly to avoid immediate boundary violations
    x_max = pos[:, 0].max()
    x_min = pos[:, 0].min()
    y_max = pos[:, 1].max()
    y_min = pos[:, 1].min()
    
    # We want the extent to be < 0.8
    scale_x = 0.8 / (x_max - x_min) if x_max > x_min else 1.0
    scale_y = 0.8 / (y_max - y_min) if y_max > y_min else 1.0
    scale = min(scale_x, scale_y)
    
    pos *= scale
    pos[:, 0] += 0.5
    pos[:, 1] += 0.5
    
    # Initial radii
    # After scaling, the distance between centers has changed.
    # We should set initial radii based on the new min distance.
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                min_dist = d
    r_init = min_dist * 0.45  # Leave some gap
    
    radii = np.full(N, r_init)
    
    v0 = np.concatenate([pos[:, 0], pos[:, 1], radii])
    
    # 2. Objective and Penalty
    # We maximize sum(r) subject to constraints.
    # Minimize -sum(r) + lambda * (Penalty)
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # 1. Sum of radii
        sum_r = np.sum(r)
        
        # 2. Boundary Penalties
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # Violation if r - x > 0, etc.
        p_bound = 0.0
        p_bound += np.sum(softplus(r - x))
        p_bound += np.sum(softplus(r - (1.0 - x)))
        p_bound += np.sum(softplus(r - y))
        p_bound += np.sum(softplus(r - (1.0 - y)))
        
        # 3. Overlap Penalties
        # d_ij >= r_i + r_j
        # Violation if r_i + r_j - d_ij > 0
        p_overlap = 0.0
        # Vectorized calculation for speed? 
        # N is small (26), loop is fine and simpler.
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                viol = r[i] + r[j] - d
                p_overlap += softplus(viol) ** 2
        
        # Combine
        # Weight for penalty. High weight ensures feasibility.
        # If weight is too high, optimizer gets stuck.
        # 50 seems reasonable for this scale.
        lambda_pen = 50.0
        return -sum_r + lambda_pen * (p_bound + p_overlap)
    
    # 3. Optimization
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
    for i in range(N):
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.0, 0.5))  # r (max possible is 0.5)
        
    best_v = v0
    best_score = -np.inf
    
    # Run optimizer
    # We can try multiple restarts if time allows
    restarts = 3
    for k in range(restarts):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial guess slightly for restarts > 0
        if k == 0:
            v_start = v0
        else:
            noise = np.random.normal(0, 0.01, size=v0.shape)
            # Keep noise small to stay in feasible region roughly
            v_start = v0 + noise
            # Clip to bounds
            v_start[:N] = np.clip(v_start[:N], 0, 1)
            v_start[N:2*N] = np.clip(v_start[N:2*N], 0, 1)
            v_start[2*N:3*N] = np.clip(v_start[2*N:3*N], 0, 0.5)
            
        try:
            res = minimize(objective, v_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_v = res.x
        except Exception:
            pass
            
    # 4. Validation and Repair
    x = best_v[:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    
    def is_valid(x, y, r):
        # Check boundaries
        for i in range(N):
            if r[i] > x[i] + 1e-9 or r[i] > 1 - x[i] + 1e-9 or r[i] > y[i] + 1e-9 or r[i] > 1 - y[i] + 1e-9:
                return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if r[i] + r[j] > d + 1e-9:
                    return False
        return True
    
    # If invalid, repair by shrinking radii
    if not is_valid(x, y, r):
        # Simple repair: scale down radii
        # Find the minimum scale factor that makes it valid
        # This is a bisection search on scale factor s in [0, 1]
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            r_test = r * mid
            if is_valid(x, y, r_test):
                lo = mid
            else:
                hi = mid
        r = r * lo
        # Ensure r is not negative
        r = np.maximum(r, 0)
        
    # Final check: if still invalid (e.g. positions are bad), fallback to seed
    if not is_valid(x, y, r):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.column_stack((x, y, r))
