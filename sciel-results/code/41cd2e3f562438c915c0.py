# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Vectorized overlap check
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[i_idx] + radii[j_idx]
    if np.any(dist < min_dist - tol):
        return False
    return True

def _objective_and_grad(z):
    """
    Objective: -sum(radii)
    Constraints handled separately in scipy.minimize.
    z: [x1, y1, r1, x2, y2, r2, ...]
    """
    n = N
    pos = z[0:3*n:3].reshape(n, 2) # This slicing is wrong, z is flat
    # Correct unpacking:
    # z = [x1, y1, r1, x2, y2, r2, ...]
    # pos = z.reshape(n, 3)[:, :2]
    # radii = z.reshape(n, 3)[:, 2]
    
    z_reshaped = z.reshape(n, 3)
    pos = z_reshaped[:, :2]
    radii = z_reshaped[:, 2]
    
    # Objective: Minimize -sum(radii)
    val = -np.sum(radii)
    
    # Gradient:
    # d(-sum(r))/dz = [0, 0, -1, 0, 0, -1, ...]
    grad = np.zeros_like(z)
    grad[2::3] = -1.0
    
    return val, grad

def _constraint_circle_circle(z):
    """
    Constraint: dist(i, j) >= r_i + r_j
    Returns array of (dist - r_i - r_j) for all pairs.
    We want these to be >= 0.
    """
    n = N
    z_reshaped = z.reshape(n, 3)
    pos = z_reshaped[:, :2]
    radii = z_reshaped[:, 2]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Constraint value: dist - r_i - r_j
    # Must be >= 0
    val = dist - (radii[i_idx] + radii[j_idx])
    return val

def _constraint_circle_boundary(z):
    """
    Constraints:
    x_i >= r_i  => x_i - r_i >= 0
    1 - x_i >= r_i => 1 - x_i - r_i >= 0
    y_i >= r_i  => y_i - r_i >= 0
    1 - y_i >= r_i => 1 - y_i - r_i >= 0
    
    Returns array of all these values.
    """
    n = N
    z_reshaped = z.reshape(n, 3)
    pos = z_reshaped[:, :2]
    radii = z_reshaped[:, 2]
    
    c1 = pos[:, 0] - radii
    c2 = 1 - pos[:, 0] - radii
    c3 = pos[:, 1] - radii
    c4 = 1 - pos[:, 1] - radii
    
    return np.concatenate([c1, c2, c3, c4])

def _grad_constraint_circle_circle(z):
    """
    Gradient of circle-circle constraints.
    Returns a list of gradients, one for each constraint.
    Since scipy.minimize with 'L-BFGS-B' expects constraints as a dict with 'fun' and 'jac',
    and 'jac' can be a function returning a matrix (Jacobian) or a list of gradients.
    For sparse problems, returning a list of gradients (sparse) is efficient, but 
    L-BFGS-B in scipy typically handles dense Jacobians or finite differences if not provided.
    However, providing the Jacobian significantly improves speed.
    
    The Jacobian is of shape (num_constraints, num_vars).
    num_constraints = n*(n-1)/2
    num_vars = 3*n
    
    This is a large dense matrix (325 x 78 for N=26). It's manageable.
    """
    n = N
    z_reshaped = z.reshape(n, 3)
    pos = z_reshaped[:, :2]
    radii = z_reshaped[:, 2]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = pos[i_idx, 0] - pos[j_idx, 0]
    dy = pos[i_idx, 1] - pos[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    # num_constraints
    m = len(i_idx)
    # num_vars
    d = 3 * n
    
    jac = np.zeros((m, d))
    
    # For each pair (i, j), the constraint is dist(i,j) - r_i - r_j
    # d(dist)/dx_i = (x_i - x_j) / dist
    # d(dist)/dy_i = (y_i - y_j) / dist
    # d(dist)/dx_j = (x_j - x_i) / dist = -d(dist)/dx_i
    # d(dist)/dy_j = (y_j - y_i) / dist = -d(dist)/dy_i
    # d(dist)/dr_i = -1
    # d(dist)/dr_j = -1
    
    # Indices in z:
    # x_i is at 3*i, y_i is at 3*i+1, r_i is at 3*i+2
    # x_j is at 3*j, y_j is at 3*j+1, r_j is at 3*j+2
    
    grad_dist_x_i = dx / dist_safe
    grad_dist_y_i = dy / dist_safe
    grad_dist_x_j = -grad_dist_x_i
    grad_dist_y_j = -grad_dist_y_i
    
    # Fill Jacobian rows
    # Row k corresponds to pair (i_idx[k], j_idx[k])
    jac[np.arange(m), 3 * i_idx] = grad_dist_x_i
    jac[np.arange(m), 3 * i_idx + 1] = grad_dist_y_i
    jac[np.arange(m), 3 * i_idx + 2] = -1.0
    
    jac[np.arange(m), 3 * j_idx] = grad_dist_x_j
    jac[np.arange(m), 3 * j_idx + 1] = grad_dist_y_j
    jac[np.arange(m), 3 * j_idx + 2] = -1.0
    
    return jac

def _grad_constraint_circle_boundary(z):
    """
    Gradient of circle-boundary constraints.
    c1 = x - r => grad: [1, 0, -1] for x, y, r
    c2 = 1 - x - r => grad: [-1, 0, -1]
    c3 = y - r => grad: [0, 1, -1]
    c4 = 1 - y - r => grad: [0, -1, -1]
    
    Returns Jacobian matrix of shape (4*n, 3*n).
    """
    n = N
    jac = np.zeros((4 * n, 3 * n))
    
    # c1: x_i - r_i
    # Rows 0 to n-1
    jac[np.arange(n), 3 * np.arange(n)] = 1.0
    jac[np.arange(n), 3 * np.arange(n) + 2] = -1.0
    
    # c2: 1 - x_i - r_i
    # Rows n to 2n-1
    jac[np.arange(n) + n, 3 * np.arange(n)] = -1.0
    jac[np.arange(n) + n, 3 * np.arange(n) + 2] = -1.0
    
    # c3: y_i - r_i
    # Rows 2n to 3n-1
    jac[np.arange(n) + 2 * n, 3 * np.arange(n) + 1] = 1.0
    jac[np.arange(n) + 2 * n, 3 * np.arange(n) + 2] = -1.0
    
    # c4: 1 - y_i - r_i
    # Rows 3n to 4n-1
    jac[np.arange(n) + 3 * n, 3 * np.arange(n) + 1] = -1.0
    jac[np.arange(n) + 3 * n, 3 * np.arange(n) + 2] = -1.0
    
    return jac

def _build_initial_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    # Small initial radius to ensure feasibility
    r = 0.4 / side
    radii = np.full(n, r)
    
    z = np.zeros(3 * n)
    for i in range(n):
        z[3*i] = pos[i, 0]
        z[3*i+1] = pos[i, 1]
        z[3*i+2] = radii[i]
    return z

def _build_initial_hex(n):
    pos_list = []
    row = 0
    # Hex packing logic similar to before but ensuring fit
    spacing = 0.18
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * spacing
            y = (row + 0.5) * spacing * 0.866
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    
    z = np.zeros(3 * n)
    for i in range(n):
        z[3*i] = pos[i, 0]
        z[3*i+1] = pos[i, 1]
        z[3*i+2] = radii[i]
    return z

def _build_initial_random(n, seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    r = 0.05
    radii = np.full(n, r)
    
    z = np.zeros(3 * n)
    for i in range(n):
        z[3*i] = pos[i, 0]
        z[3*i+1] = pos[i, 1]
        z[3*i+2] = radii[i]
    return z

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_score = -np.inf
    best_z = None
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': _constraint_circle_circle, 'jac': _grad_constraint_circle_circle},
        {'type': 'ineq', 'fun': _constraint_circle_boundary, 'jac': _grad_constraint_circle_boundary}
    ]
    
    # Initial configurations
    inits = []
    inits.append(_build_initial_grid(N))
    inits.append(_build_initial_hex(N))
    for seed in [42, 123, 456, 789, 1000]:
        inits.append(_build_initial_random(N, seed))
    
    for z0 in inits:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            res = minimize(
                _objective_and_grad,
                z0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.status in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                z_res = res.x
                # Verify validity strictly
                n = N
                z_reshaped = z_res.reshape(n, 3)
                pos = z_reshaped[:, :2]
                radii = z_reshaped[:, 2]
                
                # Clamp tiny violations due to floating point
                pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
                pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
                radii = np.clip(radii, 0, 0.5)
                
                if _check_validity(pos, radii):
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_z = z_res.copy()
        except Exception as e:
            pass # Ignore optimization failures and continue to next init
    
    # Fallback if no valid solution found
    if best_z is None:
        best_z = _build_initial_grid(N)
        # Force validity for fallback
        z_reshaped = best_z.reshape(N, 3)
        pos = z_reshaped[:, :2]
        radii = z_reshaped[:, 2]
        # Shrink radii to ensure validity
        for i in range(N):
            # Check boundaries
            r_max = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
            radii[i] = min(radii[i], r_max * 0.9)
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = radii[i] + radii[j]
                if dist < r_sum:
                    # Shrink both
                    shrink = (r_sum - dist) / 2 + 1e-5
                    radii[i] = max(0, radii[i] - shrink)
                    radii[j] = max(0, radii[j] - shrink)
        best_z[2::3] = radii
    
    # Extract final result
    z_final = best_z.reshape(N, 3)
    result_arr = z_final
    
    return result_arr
# EVOLVE-BLOCK-END
