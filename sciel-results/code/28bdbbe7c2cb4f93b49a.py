"""Improved circle packing in unit square using physics-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _initial_grid(n):
    """Create a grid layout as initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y])
    positions = positions[:n]
    # Initial radius estimate
    r0 = 0.5 / side * 0.9
    return np.array(positions), r0

def _initial_hexagonal(n):
    """Create a hexagonal-like layout as initial guess."""
    positions = []
    # Approximate hexagonal packing
    # Rows offset
    row_h = 0.5 / np.sqrt(3)
    col_w = 0.5
    row = 0
    while len(positions) < n:
        for col in range(10): # Enough columns to cover square
            x = (col + 0.5) * col_w
            if x > 1.0: break
            y = (row + 0.5) * row_h
            if y > 1.0: break
            # Offset every other row
            if row % 2 == 1:
                x += col_w / 2
                if x > 1.0: continue
            positions.append([x, y])
        row += 1
    
    # Shuffle and take first n
    np.random.shuffle(positions)
    positions = np.array(positions[:n])
    
    # Estimate radius based on average distance to neighbors
    # Simple heuristic: assume they are roughly close
    r0 = 0.1
    return positions, r0

def _slsqp_objective(params, n):
    """
    Objective: Minimize negative sum of radii.
    params: [x0, y0, r0, x1, y1, r1, ...]
    """
    radii = params[2*n:]
    return -np.sum(radii)

def _slsqp_constraints(params, n):
    """
    Generate constraints for SLSQP.
    1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    2. Non-overlap constraints: dist(i,j) >= r_i + r_j
    """
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    constraints = []
    
    # Boundary constraints: g(x) >= 0
    # x - r >= 0
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i+1] - p[2*n+i]})
    
    # Non-overlap constraints: dist - (r_i + r_j) >= 0
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(p, i=i, j=j):
                dx = p[2*i] - p[2*j]
                dy = p[2*i+1] - p[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - p[2*n+i] - p[2*n+j]
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
            
    return constraints

def _is_valid(params, n, tol=1e-6):
    """Check if configuration is valid (no overlaps, within bounds)."""
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Check bounds
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, keep 5s margin for safety
    
    best_result = None
    best_score = -np.inf
    
    # Bounds for SLSQP
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square is 0.5)
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    
    # Try multiple random restarts with different initializations
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        
        # Alternate between grid and hexagonal initializations
        if n_restarts % 2 == 0:
            positions, r0 = _initial_grid(N)
        else:
            positions, r0 = _initial_hexagonal(N)
        
        # Add random perturbation to escape local minima
        np.random.seed(42 + n_restarts)
        positions += np.random.uniform(-0.05, 0.05, size=positions.shape)
        positions = np.clip(positions, 0.0, 1.0)
        
        # Initial parameters
        params0 = np.zeros(3*N)
        params0[:2*N] = positions.flatten()
        params0[2*N:] = r0
        
        # Optimize
        try:
            result = minimize(
                _slsqp_objective,
                params0,
                args=(N,),
                method='SLSQP',
                bounds=bounds,
                constraints=_slsqp_constraints(params0, N),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success:
                # Verify validity strictly
                if _is_valid(result.x, N):
                    score = np.sum(result.x[2*N:])
                    if score > best_score:
                        best_score = score
                        best_result = result.x.reshape(N, 3)
                        
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline:
            break
    
    # If no valid result found, return the grid seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
