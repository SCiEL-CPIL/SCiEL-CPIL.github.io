"""New program for circle packing using Multi-start SLSQP optimization."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    Uses Multi-start SLSQP optimization with vectorized constraints.
    """
    n = N
    best_val = -np.inf
    best_sol = None
    
    # Pre-compute indices for non-overlap constraints
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def objective(v):
        # v is shape (3*n,)
        return -np.sum(v[2::3])
        
    def boundary_and_radius(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        c1 = x - r
        c2 = 1 - x - r
        c3 = y - r
        c4 = 1 - y - r
        c5 = r - 1e-5
        return np.concatenate([c1, c2, c3, c4, c5])
        
    def non_overlap(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dr = r[i_idx] + r[j_idx]
        return dx*dx + dy*dy - dr*dr
        
    constraints = [
        {'type': 'ineq', 'fun': boundary_and_radius},
        {'type': 'ineq', 'fun': non_overlap}
    ]
    
    # Bounds for variables
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-5, 0.5)) # r
        
    # Initial guesses
    np.random.seed(42)
    
    initial_guesses = []
    
    # 1. Hexagonal grid
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            x = (i + 0.5) / side + (0.5 / side if j % 2 == 1 else 0)
            y = (j + 0.5) / side
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            pts.append([x, y, 0.5/side - 0.01])
            
    if len(pts) == n:
        initial_guesses.append(np.array(pts).flatten())
        
    # 2. Random restarts
    for _ in range(10):
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.full(n, 0.02)
        v = np.zeros(3*n)
        v[0::3] = x
        v[1::3] = y
        v[2::3] = r
        initial_guesses.append(v)
        
    # Run optimization
    for v0 in initial_guesses:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300, 'ftol': 1e-8})
            if res.success or (not res.success and res.fun < best_val):
                # Check validity manually to be safe
                x = res.x[0::3]
                y = res.x[1::3]
                r = res.x[2::3]
                
                # Verify constraints
                valid = True
                if np.any(x < r - 1e-6) or np.any(x > 1 - r + 1e-6): valid = False
                if np.any(y < r - 1e-6) or np.any(y > 1 - r + 1e-6): valid = False
                if np.any(r < 1e-6): valid = False
                
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                            min_dist = r[i] + r[j]
                            if dist_sq < min_dist**2 - 1e-6:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_sol = res.x.copy()
                        
        except Exception as e:
            pass
            
    if best_sol is None:
        # Fallback to simple grid if optimization failed
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= n:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
        return np.array(cells)
        
    # Reshape best_sol to (n, 3)
    x = best_sol[0::3]
    y = best_sol[1::3]
    r = best_sol[2::3]
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
