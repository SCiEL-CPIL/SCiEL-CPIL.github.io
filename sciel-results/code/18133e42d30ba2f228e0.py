"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _hexagonal_lattice(n, margin=0.05):
    """
    Generate n points on a hexagonal lattice inside [margin, 1-margin]^2.
    Returns array of shape (n, 2).
    """
    points = []
    # Hexagonal lattice with spacing s
    # For a hex lattice, horizontal spacing = s, vertical spacing = s * sqrt(3)/2
    # We want to pack as many as possible in the square.
    
    # Let's try to fit a grid of hex points.
    # Number of rows and columns.
    # Approximate area per point ~ s^2 * sqrt(3)/2
    # Total area ~ (1 - 2*margin)^2
    # s ~ sqrt(2 * area / (n * sqrt(3)))
    
    side = 1.0 - 2.0 * margin
    s = np.sqrt(2.0 * side * side / (n * np.sqrt(3)))
    
    row_spacing = s * np.sqrt(3) / 2
    col_spacing = s
    
    # Determine number of rows and columns
    n_rows = int(np.floor(side / row_spacing)) + 1
    n_cols = int(np.floor(side / col_spacing)) + 1
    
    # Generate points
    for i in range(n_rows):
        y = margin + i * row_spacing
        offset = (i % 2) * (col_spacing / 2.0)
        for j in range(n_cols):
            x = margin + j * col_spacing + offset
            if margin <= x <= 1.0 - margin and margin <= y <= 1.0 - margin:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than n points, randomly sample n
    if len(points) > n:
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        while len(points) < n:
            points = np.vstack([points, np.array([np.random.uniform(margin, 1-margin), 
                                                   np.random.uniform(margin, 1-margin)])])
    
    return points[:n]


def _objective_and_grad(z, n):
    """
    Objective: maximize sum of radii subject to constraints.
    We use a penalty method: minimize -sum(r) + penalty * (violation)
    
    z = [x_0, ..., x_{n-1}, y_0, ..., y_{n-1}, r_0, ..., r_{n-1}]
    """
    x = z[:n]
    y = z[n:2*n]
    r = z[2*n:3*n]
    
    # Ensure positive radii
    r_pos = np.maximum(r, 1e-6)
    
    # Sum of radii
    sum_r = np.sum(r_pos)
    
    # Penalty for boundary violations
    # x_i - r_i >= 0, x_i + r_i <= 1, same for y
    boundary_viol = 0.0
    for i in range(n):
        # Left boundary
        v = max(0, r_pos[i] - x[i])
        boundary_viol += v * v
        # Right boundary
        v = max(0, x[i] + r_pos[i] - 1.0)
        boundary_viol += v * v
        # Bottom boundary
        v = max(0, r_pos[i] - y[i])
        boundary_viol += v * v
        # Top boundary
        v = max(0, y[i] + r_pos[i] - 1.0)
        boundary_viol += v * v
    
    # Penalty for overlap violations
    # dist(i,j) >= r_i + r_j
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r_pos[i] + r_pos[j]
            v = max(0, min_dist - dist)
            overlap_viol += v * v
    
    penalty_weight = 100.0
    obj = -sum_r + penalty_weight * (boundary_viol + overlap_viol)
    
    # Gradient (approximate via finite differences for simplicity, or analytic)
    # For speed, we'll use analytic gradient
    grad = np.zeros_like(z)
    
    # Gradient of -sum(r)
    grad[2*n:3*n] = -1.0
    
    # Gradient of boundary penalties
    for i in range(n):
        # Left: max(0, r - x)^2
        v = r_pos[i] - x[i]
        if v > 0:
            # d/dx: -2v, d/dr: 2v
            grad[i] += penalty_weight * (-2 * v)
            grad[2*n + i] += penalty_weight * (2 * v)
        
        # Right: max(0, x + r - 1)^2
        v = x[i] + r_pos[i] - 1.0
        if v > 0:
            grad[i] += penalty_weight * (2 * v)
            grad[2*n + i] += penalty_weight * (2 * v)
        
        # Bottom: max(0, r - y)^2
        v = r_pos[i] - y[i]
        if v > 0:
            grad[n + i] += penalty_weight * (-2 * v)
            grad[2*n + i] += penalty_weight * (2 * v)
        
        # Top: max(0, y + r - 1)^2
        v = y[i] + r_pos[i] - 1.0
        if v > 0:
            grad[n + i] += penalty_weight * (2 * v)
            grad[2*n + i] += penalty_weight * (2 * v)
    
    # Gradient of overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r_pos[i] + r_pos[j]
            v = min_dist - dist
            if v > 0:
                if dist > 1e-10:
                    # d/dx_i: 2v * (dx/dist), d/dx_j: -2v * (dx/dist)
                    # d/dr_i: 2v, d/dr_j: 2v
                    factor = 2 * v / dist
                    grad[i] += penalty_weight * factor * dx
                    grad[j] -= penalty_weight * factor * dx
                    grad[n + i] += penalty_weight * factor * dy
                    grad[n + j] -= penalty_weight * factor * dy
                    grad[2*n + i] += penalty_weight * 2 * v
                    grad[2*n + j] += penalty_weight * 2 * v
    
    return obj, grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple random restarts with different initializations
    n_restarts = 10
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with hexagonal lattice + random perturbation
        x0 = _hexagonal_lattice(N, margin=0.05)
        # Add small random perturbation
        x0 += rng.uniform(-0.02, 0.02, size=x0.shape)
        
        # Initial radii: small
        r0 = np.full(N, 0.03)
        
        z0 = np.concatenate([x0[:, 0], x0[:, 1], r0])
        
        # Bounds
        # x in [0, 1], y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 1.0 - 1e-6))  # x
        for i in range(N):
            bounds.append((1e-6, 1.0 - 1e-6))  # y
        for i in range(N):
            bounds.append((1e-6, 0.5))  # r
        
        # Optimize
        try:
            res = minimize(
                _objective_and_grad,
                z0,
                args=(N,),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.status == 0:
                z_opt = res.x
                x_opt = z_opt[:N]
                y_opt = z_opt[N:2*N]
                r_opt = z_opt[2*N:3*N]
                
                # Check validity: all constraints satisfied
                valid = True
                for i in range(N):
                    if x_opt[i] - r_opt[i] < -1e-5 or x_opt[i] + r_opt[i] > 1.0 + 1e-5:
                        valid = False
                        break
                    if y_opt[i] - r_opt[i] < -1e-5 or y_opt[i] + r_opt[i] > 1.0 + 1e-5:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if dist < r_opt[i] + r_opt[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_opt, y_opt, r_opt])
        
        except Exception:
            continue
    
    # If no valid result found, return a trivial valid packing
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
