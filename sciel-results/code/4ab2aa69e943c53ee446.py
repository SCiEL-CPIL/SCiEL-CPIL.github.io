# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 21

def _constraint_fun(vars, w, h):
    """
    Vectorized constraint function for SLSQP.
    Returns array of constraint values (should be >= 0).
    """
    x = vars[:N]
    y = vars[N:2*N]
    r = vars[2*N:]
    
    # Pairwise constraints: dist^2 - (ri + rj)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx * dx + dy * dy
    min_dist = r[:, None] + r[None, :]
    
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_constr = (dist_sq - min_dist * min_dist)[mask]
    
    # Wall constraints:
    # x - r >= 0
    # w - x - r >= 0
    # y - r >= 0
    # h - y - r >= 0
    wall_constr = np.concatenate([
        x - r,
        w - x - r,
        y - r,
        h - y - r
    ])
    
    return np.concatenate([pair_constr, wall_constr])

def _solve_packing_slsqp(w, h):
    """
    Solve max sum(r) for given w, h using SLSQP.
    """
    # Initial guess: Grid
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x0[i] = (col + 0.5) * w / cols
        y0[i] = (row + 0.5) * h / rows
        r0[i] = min(w / cols, h / rows) * 0.3
        
    x_init = np.concatenate([x0, y0, r0])
    
    def objective(vars):
        r = vars[2*N:]
        return -np.sum(r)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))
    for i in range(N):
        bounds.append((0, h))
    for i in range(N):
        bounds.append((1e-6, min(w, h) * 0.49)) # Max radius constraint
        
    constraints = {
        'type': 'ineq',
        'fun': lambda v: _constraint_fun(v, w, h)
    }
    
    try:
        res = minimize(objective, x_init, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 150, 'ftol': 1e-6})
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    
    return None, -np.inf

def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    """
    best_val = -np.inf
    best_result = None
    
    # w + h = 2
    # Scan aspect ratios
    # Optimal is likely near square or slightly elongated.
    w_values = np.linspace(0.85, 1.15, 7)
    
    for w in w_values:
        h = 2.0 - w
        res_x, val = _solve_packing_slsqp(w, h)
        if res_x is not None:
            if val > best_val:
                best_val = val
                best_result = res_x
                
    if best_result is not None:
        x = best_result[:N]
        y = best_result[N:2*N]
        r = best_result[2*N:]
        return np.column_stack([x, y, r])
        
    # Fallback: Simple grid
    w, h = 1.0, 1.0
    cols = 5
    rows = 5
    x = []
    y = []
    r = []
    for i in range(cols):
        for j in range(rows):
            if len(x) >= N: break
            x.append((i + 0.5) * w / cols)
            y.append((j + 0.5) * h / rows)
            r.append(min(w/cols, h/rows) * 0.4)
            
    return np.array(list(zip(x, y, r)))[:N]

# EVOLVE-BLOCK-END
