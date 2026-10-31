"""Rewritten program for circle packing in unit square; EVOLVE-BLOCK replaced."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _pairwise_constraints(x, radii):
    """
    x: (2N,) flattened positions.
    radii: (N,) array.
    Returns (A, b) for linear constraints A @ x >= b (converted to <= for scipy).
    Actually we need: ||p_i - p_j|| >= r_i + r_j for all i<j.
    This is non-linear. We cannot use LinearConstraint directly for the non-overlap.
    We will use a penalty method or SLSQP with non-linear constraints.
    """
    pass

def _objective(x, radii):
    """
    Minimize negative sum of radii? No, we want to maximize sum of radii.
    But radii are also variables.
    Let variables be: x (2N positions), r (N radii).
    Objective: minimize -sum(r).
    Constraints:
    1. Boundary: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i.
    2. Non-overlap: ||p_i - p_j|| >= r_i + r_j.
    3. r_i > 0.
    """
    return -np.sum(radii)

def _run_optimization(initial_x, initial_r, max_iter=2000, tol=1e-8):
    """
    Use SLSQP to optimize positions and radii.
    Variables: z = [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    n = len(initial_r)
    x0 = np.concatenate([initial_x, initial_r])
    
    # Constraints
    # We define constraints as g(z) >= 0
    constraints = []
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=i: z[2*i] - z[2*n + i],
            'jac': lambda z, i=i: np.concatenate([[-1 if j == 2*i else 0 for j in range(2*n)], [0 if j != 2*n + i else -1 for j in range(n)]])
        })
        # 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=i: 1 - z[2*i] - z[2*n + i],
            'jac': lambda z, i=i: np.concatenate([[-1 if j == 2*i else 0 for j in range(2*n)], [0 if j != 2*n + i else -1 for j in range(n)]])
        })
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=i: z[2*i + 1] - z[2*n + i],
            'jac': lambda z, i=i: np.concatenate([[-1 if j == 2*i + 1 else 0 for j in range(2*n)], [0 if j != 2*n + i else -1 for j in range(n)]])
        })
        # 1 - y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, i=i: 1 - z[2*i + 1] - z[2*n + i],
            'jac': lambda z, i=i: np.concatenate([[-1 if j == 2*i + 1 else 0 for j in range(2*n)], [0 if j != 2*n + i else -1 for j in range(n)]])
        })
    
    # Non-overlap constraints:
    # ||p_i - p_j|| - (r_i + r_j) >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: np.sqrt((z[2*i] - z[2*j])**2 + (z[2*i+1] - z[2*j+1])**2) - (z[2*n + i] + z[2*n + j]),
                'jac': lambda z, i=i, j=j: _nonoverlap_jac(z, i, j, n)
            })
    
    # Bounds
    # Positions in [0, 1] (loose, refined by constraints)
    # Radii in [1e-4, 0.5]
    bounds = []
    for i in range(2*n):
        bounds.append((0, 1))
    for i in range(n):
        bounds.append((1e-4, 0.5))
    
    def obj_func(z):
        r = z[2*n:]
        return -np.sum(r)
    
    def obj_jac(z):
        jac = np.zeros_like(z)
        jac[2*n:] = -1.0
        return jac
    
    try:
        res = minimize(obj_func, x0, jac=obj_jac, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': max_iter, 'ftol': tol, 'disp': False})
        if res.success:
            x_opt = res.x[:2*n].reshape(n, 2)
            r_opt = res.x[2*n:]
            return x_opt, r_opt, res.fun
        else:
            # If failed, return best found so far? SLSQP doesn't always provide best.
            # We can just return the input if it failed, or try to salvage.
            # For now, return input.
            return initial_x, initial_r, _objective(initial_x, initial_r)
    except Exception as e:
        return initial_x, initial_r, _objective(initial_x, initial_r)

def _nonoverlap_jac(z, i, j, n):
    """
    Jacobian for constraint: ||p_i - p_j|| - (r_i + r_j) >= 0
    z = [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    """
    dx = z[2*i] - z[2*j]
    dy = z[2*i+1] - z[2*j+1]
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-12:
        dist = 1e-12
        dx = 1e-12
        dy = 0
    
    jac = np.zeros(3*n)
    jac[2*i] = dx / dist
    jac[2*i+1] = dy / dist
    jac[2*j] = -dx / dist
    jac[2*j+1] = -dy / dist
    jac[2*n + i] = -1.0
    jac[2*n + j] = -1.0
    return jac

def _is_valid(x, r):
    """Check if packing is valid."""
    n = len(r)
    for i in range(n):
        if x[i, 0] < r[i] - 1e-8 or x[i, 0] > 1 - r[i] + 1e-8:
            return False
        if x[i, 1] < r[i] - 1e-8 or x[i, 1] > 1 - r[i] + 1e-8:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2)
            if dist < r[i] + r[j] - 1e-8:
                return False
    return True

def _initial_grid(n):
    """Create initial grid positions and small radii."""
    side = int(np.ceil(np.sqrt(n)))
    x = np.zeros((n, 2))
    r = np.zeros(n)
    for i in range(n):
        ix = i % side
        iy = i // side
        x[i, 0] = (ix + 0.5) / side
        x[i, 1] = (iy + 0.5) / side
        r[i] = 0.5 / side * 0.5  # Start with small radii
    return x, r

def _local_search(x, r, num_restarts=5, max_iter=500, tol=1e-6):
    """
    Perform local search from multiple starting points.
    """
    best_x = x.copy()
    best_r = r.copy()
    best_val = np.sum(r)
    
    for restart in range(num_restarts):
        # Perturb initial positions
        x_init = x.copy()
        r_init = r.copy()
        if restart > 0:
            noise = 0.05 * np.random.randn(n, 2)
            x_init = np.clip(x_init + noise, 0.05, 0.95)
            # Keep radii same for now
        
        x_opt, r_opt, val = _run_optimization(x_init, r_init, max_iter=max_iter, tol=tol)
        
        if _is_valid(x_opt, r_opt):
            current_val = np.sum(r_opt)
            if current_val > best_val:
                best_val = current_val
                best_x = x_opt
                best_r = r_opt
    
    return best_x, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial grid
    x, r = _initial_grid(N)
    
    # Run local search
    best_x, best_r = _local_search(x, r, num_restarts=3, max_iter=1000, tol=1e-7)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Not enough time for more, return current best
        pass
    else:
        # Try a few more restarts or higher precision
        # For now, just return. In a real scenario, we might do more.
        pass
    
    # Construct result
    result = np.zeros((N, 3))
    result[:, 0] = best_x[:, 0]
    result[:, 1] = best_x[:, 1]
    result[:, 2] = best_r
    
    # Final validity check
    if not _is_valid(best_x, best_r):
        # Fallback to a safe, smaller packing
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
    
    return result
# EVOLVE-BLOCK-END
