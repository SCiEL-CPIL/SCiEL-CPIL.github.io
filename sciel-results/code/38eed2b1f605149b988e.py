"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _grid_initial(n, rng):
    """Create a grid-based initial configuration with slight jitter."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-0.01, 0.01)
            y = (j + 0.5) / side + rng.uniform(-0.01, 0.01)
            r = 0.5 / side * 0.95
            cells.append([x, y, r])
    # Pad if needed
    while len(cells) < n:
        cells.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.02])
    return np.array(cells[:n])

def _random_initial(n, rng):
    """Create a random initial configuration."""
    x = rng.uniform(0.05, 0.95, size=n)
    y = rng.uniform(0.05, 0.95, size=n)
    r = rng.uniform(0.02, 0.1, size=n)
    return np.column_stack([x, y, r])

def _hex_initial(n, rng):
    """Create a hexagonal dense initial configuration."""
    # Place circles in a hexagonal pattern
    rows = []
    row_y = 0.15
    spacing = 0.28
    while len(rows) < n and row_y < 0.9:
        row = []
        offset = 0 if len(rows) % 2 == 0 else spacing / 2
        x = 0.15 + offset
        while x < 0.9 and len(rows) + len(row) < n:
            row.append([x, row_y, 0.12])
            x += spacing
        rows.extend(row)
        row_y += spacing * 0.866
    
    # Pad if needed
    while len(rows) < n:
        rows.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.05])
    
    return np.array(rows[:n])

def _objective(params, n):
    """Objective: minimize negative sum of radii."""
    r = params[2*n:3*n]
    return -np.sum(r)

def _objective_grad(params, n):
    """Gradient of the objective."""
    grad = np.zeros_like(params)
    grad[2*n:3*n] = -1.0
    return grad

def _make_constraints(n):
    """Create constraint functions for SLSQP."""
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # Non-overlap: dist(i,j) >= r[i] + r[j]
    
    def boundary_x_lower(params):
        x = params[:n]
        r = params[2*n:3*n]
        return x - r
    
    def boundary_x_upper(params):
        x = params[:n]
        r = params[2*n:3*n]
        return (1 - r) - x
    
    def boundary_y_lower(params):
        y = params[n:2*n]
        r = params[2*n:3*n]
        return y - r
    
    def boundary_y_upper(params):
        y = params[n:2*n]
        r = params[2*n:3*n]
        return (1 - r) - y
    
    # Non-overlap constraints for each pair
    def make_nonoverlap(i, j):
        def constraint(params):
            x = params[:n]
            y = params[n:2*n]
            r = params[2*n:3*n]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-12
            return dist - r[i] - r[j]
        return constraint
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_x_lower},
        {'type': 'ineq', 'fun': boundary_x_upper},
        {'type': 'ineq', 'fun': boundary_y_lower},
        {'type': 'ineq', 'fun': boundary_y_upper},
    ]
    
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_nonoverlap(i, j)})
    
    return constraints

def _enforce_validity(circles, n):
    """Clamp circles to be valid (inside square, non-negative radius)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Clamp radii to be positive
    r = np.maximum(r, 1e-6)
    
    # Clamp positions so circles stay inside [0,1]^2
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    return np.column_stack([x, y, r])

def _score_valid(circles, n):
    """Check if all circles are valid and return sum of radii if so."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check non-negative radius
    if np.any(r < 0):
        return 0.0
    
    # Check circles inside unit square
    if np.any(x < r - 1e-9) or np.any(y < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y > 1 - r + 1e-9):
        return 0.0
    
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    
    return np.sum(r)

def _slsqp_optimize(init, n, constraints, deadline):
    """Run SLSQP optimization with explicit constraints."""
    try:
        result = minimize(
            _objective,
            init.flatten(),
            jac=_objective_grad,
            args=(n,),
            method='SLSQP',
            bounds=[(1e-6, None)] * n + [(1e-6, None)] * n + [(1e-6, 0.5)] * n,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        return result
    except Exception:
        return None

def construct():
    n = N
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Fallback: grid solution
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    fallback = np.array(cells[:n])
    best_circles = fallback.copy()
    best_score = _score_valid(best_circles, n)
    
    rng = np.random.default_rng(42)
    
    # Pre-compute constraints
    constraints = _make_constraints(n)
    
    # Try multiple initializations and optimization runs
    n_attempts = 0
    while time.time() < deadline:
        n_attempts += 1
        
        # Choose initialization strategy
        if n_attempts % 3 == 1:
            init = _grid_initial(n, rng)
        elif n_attempts % 3 == 2:
            init = _hex_initial(n, rng)
        else:
            init = _random_initial(n, rng)
        
        # Run SLSQP optimization
        result = _slsqp_optimize(init, n, constraints, deadline)
        
        if result is not None and result.success:
            params = result.x
            circles = params.reshape(n, 3)
            circles = _enforce_validity(circles, n)
            
            score = _score_valid(circles, n)
            
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
        
        # Try polishing the best solution found so far
        if best_score > 0 and time.time() < deadline - 5 and n_attempts % 2 == 0:
            result = _slsqp_optimize(best_circles, n, constraints, deadline)
            if result is not None and result.success:
                params = result.x
                circles = params.reshape(n, 3)
                circles = _enforce_validity(circles, n)
                score = _score_valid(circles, n)
                if score > best_score:
                    best_score = score
                    best_circles = circles.copy()
        
        # Break if we've done enough attempts
        if n_attempts >= 10:
            break
    
    # Final validation
    final_score = _score_valid(best_circles, n)
    if final_score <= 0:
        return fallback
    
    return best_circles
# EVOLVE-BLOCK-END
