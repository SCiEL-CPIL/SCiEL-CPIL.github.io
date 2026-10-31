"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _pairwise_indices(n):
    """Return list of (i, j) for i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _make_initial_grid(n, seed=42):
    """Create an initial grid-based configuration with some perturbation."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    # Place circles in a grid pattern
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                r = 0.5 / side * 0.95
                positions.append([x, y, r])
    
    # Pad if needed
    while len(positions) < n:
        positions.append([0.5, 0.5, 0.01])
    
    return np.array(positions[:n])

def _objective_and_grad(x, n, pairs):
    """
    Objective: minimize negative sum of radii = -sum(r_i)
    Subject to:
      - r_i >= 0
      - r_i <= x_i (distance to left edge)
      - r_i <= 1 - x_i (distance to right edge)
      - r_i <= y_i (distance to bottom edge)
      - r_i <= 1 - y_i (distance to top edge)
      - r_i + r_j <= ||p_i - p_j|| for all pairs (i,j)
    """
    # x is flattened: [x0, y0, r0, x1, y1, r1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
    pos = x.reshape(n, 3)
    radii = pos[:, 2]
    
    # Objective: maximize sum of radii -> minimize negative
    obj = -np.sum(radii)
    
    # Gradient
    grad = np.zeros_like(x)
    grad[2::3] = -1.0  # d(-sum(r))/dr_i = -1
    
    return obj, grad

def _constraints(x, n, pairs):
    """
    Define inequality constraints (g(x) >= 0).
    Boundary constraints:
      r_i <= x_i -> x_i - r_i >= 0
      r_i <= 1 - x_i -> 1 - x_i - r_i >= 0
      r_i <= y_i -> y_i - r_i >= 0
      r_i <= 1 - y_i -> 1 - y_i - r_i >= 0
      r_i >= 0 -> r_i >= 0
    Pairwise non-overlap:
      ||p_i - p_j|| - r_i - r_j >= 0
    """
    pos = x.reshape(n, 3)
    
    cons = []
    
    # Boundary constraints
    for i in range(n):
        xi, yi, ri = pos[i]
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2],  # x_i - r_i
            'jac': lambda x, i=i: np.concatenate([np.eye(3*n)[3*i] - np.eye(3*n)[3*i+2]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2],  # 1 - x_i - r_i
            'jac': lambda x, i=i: np.concatenate([-np.eye(3*n)[3*i] - np.eye(3*n)[3*i+2]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2],  # y_i - r_i
            'jac': lambda x, i=i: np.concatenate([np.eye(3*n)[3*i+1] - np.eye(3*n)[3*i+2]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2],  # 1 - y_i - r_i
            'jac': lambda x, i=i: np.concatenate([-np.eye(3*n)[3*i+1] - np.eye(3*n)[3*i+2]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+2],  # r_i >= 0
            'jac': lambda x, i=i: np.eye(3*n)[3*i+2]
        })
    
    # Pairwise non-overlap constraints
    for i, j in pairs:
        def fun(x, i=i, j=j):
            pi = x[3*i:3*i+2]
            pj = x[3*j:3*j+2]
            ri = x[3*i+2]
            rj = x[3*j+2]
            dist = np.linalg.norm(pi - pj)
            return dist - ri - rj
        
        def jac(x, i=i, j=j):
            g = np.zeros(3*n)
            pi = x[3*i:3*i+2]
            pj = x[3*j:3*j+2]
            ri = x[3*i+2]
            rj = x[3*j+2]
            diff = pi - pj
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad_dist_i = diff / dist
                grad_dist_j = -diff / dist
            else:
                grad_dist_i = np.array([1.0, 0.0])
                grad_dist_j = np.array([-1.0, 0.0])
            g[3*i:3*i+2] = grad_dist_i
            g[3*i+2] = -1.0
            g[3*j:3*j+2] = grad_dist_j
            g[3*j+2] = -1.0
            return g
        
        cons.append({'type': 'ineq', 'fun': fun, 'jac': jac})
    
    return cons

def _is_valid(circles, tol=1e-7):
    """Check if circles are valid (disjoint interiors, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < -tol or x > 1 + tol or y < -tol or y > 1 + tol:
            return False
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _post_process(circles, n):
    """Clamp circles to be strictly inside the unit square with small margin."""
    result = circles.copy()
    for i in range(n):
        x, y, r = result[i]
        # Clamp position to ensure circle is inside [0,1]^2
        r = max(r, 1e-8)
        x = min(max(x, r), 1 - r)
        y = min(max(y, r), 1 - r)
        result[i] = [x, y, r]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = N
    pairs = _pairwise_indices(n)
    
    # Multiple restarts with different initial configurations
    best_score = -np.inf
    best_result = None
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial guess
        if restart == 0:
            x0 = _make_initial_grid(n, seed=42)
        else:
            # Perturb the grid
            rng = np.random.RandomState(42 + restart)
            x0 = _make_initial_grid(n, seed=42 + restart)
            # Add more perturbation
            x0_flat = x0.flatten().copy()
            x0_flat[:2*n] += rng.uniform(-0.05, 0.05, size=2*n)
            x0 = x0_flat.reshape(n, 3)
            # Ensure radii are positive
            x0[:, 2] = np.maximum(x0[:, 2], 0.01)
        
        # Flatten for optimization
        x0_flat = x0.flatten()
        
        # Bounds: positions in [0,1], radii in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0, 0.5))  # r
        
        # Run SLSQP optimization
        try:
            result = minimize(
                _objective_and_grad,
                x0_flat,
                args=(n, pairs),
                method='SLSQP',
                bounds=bounds,
                constraints=_constraints(x0_flat, n, pairs),
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                x_opt = result.x.reshape(n, 3)
                # Post-process to ensure validity
                x_opt = _post_process(x_opt, n)
                score = np.sum(x_opt[:, 2])
                
                if _is_valid(x_opt):
                    if score > best_score:
                        best_score = score
                        best_result = x_opt.copy()
                        
        except Exception:
            continue
    
    # If no valid result found, fall back to grid
    if best_result is None:
        best_result = _make_initial_grid(n, seed=42)
        best_result = _post_process(best_result, n)
    
    return best_result
# EVOLVE-BLOCK-END
