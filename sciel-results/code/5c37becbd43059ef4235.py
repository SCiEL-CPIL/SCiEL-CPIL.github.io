# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 26

def _objective(v, penalty=5000.0):
    """
    Objective: -sum(r) + penalty * sum(violations^2)
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary violations
    v_x_low = np.maximum(0.0, r - x)
    v_x_high = np.maximum(0.0, x + r - 1.0)
    v_y_low = np.maximum(0.0, r - y)
    v_y_high = np.maximum(0.0, y + r - 1.0)
    
    pen += np.sum(v_x_low**2) + np.sum(v_x_high**2) + np.sum(v_y_low**2) + np.sum(v_y_high**2)
    
    # Circle-circle violations
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-15)
            viol = r[i] + r[j] - d
            if viol > 0:
                pen += viol * viol
                
    return obj + penalty * pen

def _objective_grad(v, penalty=5000.0):
    """
    Analytical gradient of the penalized objective.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    grad = np.zeros(3*N)
    grad[2*N:] = -1.0
    
    # Boundary gradient contributions
    v_x_low = r - x
    v_x_high = x + r - 1.0
    v_y_low = r - y
    v_y_high = y + r - 1.0
    
    mask_x_low = v_x_low > 0
    mask_x_high = v_x_high > 0
    mask_y_low = v_y_low > 0
    mask_y_high = v_y_high > 0
    
    grad[:N][mask_x_low] += penalty * 2 * v_x_low[mask_x_low] * (-1)
    grad[:N][mask_x_high] += penalty * 2 * v_x_high[mask_x_high] * (1)
    
    grad[N:2*N][mask_y_low] += penalty * 2 * v_y_low[mask_y_low] * (-1)
    grad[N:2*N][mask_y_high] += penalty * 2 * v_y_high[mask_y_high] * (1)
    
    grad[2*N:][mask_x_low] += penalty * 2 * v_x_low[mask_x_low]
    grad[2*N:][mask_x_high] += penalty * 2 * v_x_high[mask_x_high]
    grad[2*N:][mask_y_low] += penalty * 2 * v_y_low[mask_y_low]
    grad[2*N:][mask_y_high] += penalty * 2 * v_y_high[mask_y_high]
    
    # Circle-circle gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-15)
            viol = r[i] + r[j] - d
            if viol > 0:
                factor = 2.0 * penalty * viol
                gx = dx / d
                gy = dy / d
                
                grad[i] += factor * (-gx)
                grad[j] += factor * (gx)
                grad[N+i] += factor * (-gy)
                grad[N+j] += factor * (gy)
                grad[2*N+i] += factor
                grad[2*N+j] += factor
                
    return grad

def _initial_grid(seed=0):
    """Create initial guess: uniform grid with jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cx += rng.uniform(-0.02, 0.02)
            cy += rng.uniform(-0.02, 0.02)
            r = 0.4 / side
            cells.append((cx, cy, r))
            
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    
    return np.concatenate([x, y, r])

def _initial_hexagonal(seed=0):
    """Create initial guess: hexagonal packing pattern."""
    rng = np.random.RandomState(seed)
    
    # Hexagonal packing: rows offset by half spacing
    # For 26 circles, try roughly 5-6 per row
    rows = []
    cols_per_row = [5, 5, 5, 5, 5, 1]  # 26 total
    spacing = 1.0 / (max(cols_per_row) + 0.5)
    
    positions = []
    for row_idx, n_cols in enumerate(cols_per_row):
        y_pos = (row_idx + 0.5) * spacing
        for col_idx in range(n_cols):
            x_offset = (col_idx + 0.5) * spacing
            if row_idx % 2 == 1:
                x_pos = x_offset + spacing * 0.5
            else:
                x_pos = x_offset
            # Keep within bounds
            x_pos = np.clip(x_pos, spacing, 1.0 - spacing)
            y_pos = np.clip(y_pos, spacing, 1.0 - spacing)
            x_pos += rng.uniform(-0.01, 0.01)
            y_pos += rng.uniform(-0.01, 0.01)
            r = spacing * 0.45
            positions.append((x_pos, y_pos, r))
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array([p[2] for p in positions[:N]])
    
    return np.concatenate([x, y, r])

def _initial_random(seed=0):
    """Create initial guess: random positions with small radii."""
    rng = np.random.RandomState(seed)
    
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.02)
    
    return np.concatenate([x, y, r])

def _check_feasibility(x, y, r, tol=1e-5):
    """Check if the configuration is feasible."""
    # Check boundaries
    for i in range(N):
        if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
            return False
    
    # Check circle-circle
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    
    return True

def _repair_solution(v):
    """Try to repair an infeasible solution by reducing radii."""
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:].copy()
    
    # Iteratively reduce the smallest radii that cause violations
    for iteration in range(100):
        if _check_feasibility(x, y, r, tol=1e-6):
            break
        
        # Find the worst violation
        worst_viol = -np.inf
        worst_pair = None
        
        # Check boundaries
        for i in range(N):
            viol = max(r[i] - x[i], x[i] + r[i] - 1.0, r[i] - y[i], y[i] + r[i] - 1.0)
            if viol > worst_viol:
                worst_viol = viol
                worst_pair = ('boundary', i)
        
        # Check circles
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                viol = r[i] + r[j] - d
                if viol > worst_viol:
                    worst_viol = viol
                    worst_pair = ('circle', i, j)
        
        if worst_viol <= 0:
            break
        
        # Reduce radii of involved circles
        if worst_pair[0] == 'boundary':
            i = worst_pair[1]
            r[i] *= 0.9
        else:
            i, j = worst_pair[1], worst_pair[2]
            r[i] *= 0.95
            r[j] *= 0.95
    
    return np.concatenate([x, y, r])

def construct():
    """Pack N circles in unit square maximizing sum of radii."""
    best_value = -np.inf
    best_v = None
    
    # Multiple initialization strategies
    init_strategies = [
        (_initial_grid, 5),
        (_initial_hexagonal, 3),
        (_initial_random, 4),
    ]
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.001, 0.5)] * N
    
    for init_func, n_seeds in init_strategies:
        for seed in range(n_seeds):
            try:
                v0 = init_func(seed)
                
                # Run optimization with L-BFGS-B
                result = minimize(
                    _objective,
                    v0,
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    bounds=bounds,
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-12,
                        'gtol': 1e-9,
                        'maxls': 100,
                        'maxfun': 5000
                    }
                )
                
                v = result.x
                x = v[:N]
                y = v[N:2*N]
                r = v[2*N:]
                
                # Check feasibility
                if _check_feasibility(x, y, r, tol=1e-5):
                    val = np.sum(r)
                    if val > best_value:
                        best_value = val
                        best_v = np.column_stack([x, y, r])
                else:
                    # Try to repair
                    v_repaired = _repair_solution(v)
                    x_r = v_repaired[:N]
                    y_r = v_repaired[N:2*N]
                    r_r = v_repaired[2*N:]
                    
                    if _check_feasibility(x_r, y_r, r_r, tol=1e-5):
                        val = np.sum(r_r)
                        if val > best_value:
                            best_value = val
                            best_v = np.column_stack([x_r, y_r, r_r])
                        
            except Exception:
                continue
    
    if best_v is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_v
# EVOLVE-BLOCK-END
