"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _penalty_objective(v, penalty_weight=100.0):
    """Objective: negative sum of radii plus penalty for constraint violations."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    base = -np.sum(r)
    
    # Boundary violations
    penalty = 0.0
    for i in range(N):
        penalty += max(0, r[i] - x[i])**2
        penalty += max(0, x[i] + r[i] - 1.0)**2
        penalty += max(0, r[i] - y[i])**2
        penalty += max(0, y[i] + r[i] - 1.0)**2
    
    # Circle-circle violations
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-30)
            violation = max(0, r[i] + r[j] - d)
            penalty += violation**2
    
    return base + penalty_weight * penalty


def _penalty_gradient(v, penalty_weight=100.0):
    """Gradient of the penalty objective."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    g = np.zeros(3*N)
    g[2*N:] = -1.0
    
    # Boundary gradients
    for i in range(N):
        # x_i - r_i < 0 violation
        viol = r[i] - x[i]
        if viol > 0:
            g[i] += 2 * penalty_weight * viol
            g[2*N+i] += 2 * penalty_weight * viol
        # x_i + r_i - 1 > 0 violation
        viol = x[i] + r[i] - 1.0
        if viol > 0:
            g[i] += 2 * penalty_weight * viol
            g[2*N+i] += 2 * penalty_weight * viol
        # y_i - r_i < 0 violation
        viol = r[i] - y[i]
        if viol > 0:
            g[N+i] += 2 * penalty_weight * viol
            g[2*N+i] += 2 * penalty_weight * viol
        # y_i + r_i - 1 > 0 violation
        viol = y[i] + r[i] - 1.0
        if viol > 0:
            g[N+i] += 2 * penalty_weight * viol
            g[2*N+i] += 2 * penalty_weight * viol
    
    # Circle-circle gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-30)
            violation = r[i] + r[j] - d
            if violation > 0:
                if d < 1e-15:
                    gx, gy = 0.0, 0.0
                else:
                    gx = dx / d
                    gy = dy / d
                g[i] += 2 * penalty_weight * violation * gx
                g[j] += -2 * penalty_weight * violation * gx
                g[N+i] += 2 * penalty_weight * violation * gy
                g[N+j] += -2 * penalty_weight * violation * gy
                g[2*N+i] += 2 * penalty_weight * violation
                g[2*N+j] += 2 * penalty_weight * violation
    
    return g


def _make_initial_guess(seed=0):
    """Create an initial guess with a better layout for 26 circles."""
    rng = np.random.RandomState(seed)
    
    # Use a hexagonal-like packing pattern for better initial placement
    # For 26 circles, try a 5x5 grid with some removed, or 6x5
    # Let's use a 5x5 grid (25 circles) + 1 extra, or 6x5=30 with 4 removed
    
    # Strategy: use rows with alternating offsets (hexagonal packing)
    # Row 0: 5 circles, Row 1: 5 circles, Row 2: 5 circles, Row 3: 5 circles, Row 4: 5 circles, Row 5: 1 circle
    # That's 26 circles
    
    cells = []
    n_rows = 6
    for row in range(n_rows):
        if row < 5:
            n_in_row = 5
        else:
            n_in_row = 1
        
        if n_in_row == 0:
            continue
            
        row_width = 1.0 / n_in_row
        # Hexagonal offset: odd rows shifted by half
        offset = 0.0 if row % 2 == 0 else 0.5 * row_width
        
        for col in range(n_in_row):
            cx = (col + 0.5) * row_width + offset
            cy = (row + 0.5) / n_rows
            
            # Add small jitter
            cx += rng.uniform(-0.01, 0.01)
            cy += rng.uniform(-0.01, 0.01)
            
            # Estimate radius based on local spacing
            r = min(row_width, 1.0/n_rows) * 0.45
            
            cells.append((cx, cy, r))
    
    # Trim or pad to N
    while len(cells) > N:
        cells.pop()
    while len(cells) < N:
        cells.append((0.5, 0.5, 0.02))
    
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    
    # Ensure initial feasibility: clip positions
    r = np.clip(r, 0.005, 0.15)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    
    return np.concatenate([x, y, r])


def construct():
    """Pack N circles in unit square maximizing sum of radii."""
    best_value = -np.inf
    best_v = None
    
    # Try multiple initial guesses with different penalty weights
    for seed in range(12):
        v0 = _make_initial_guess(seed)
        
        # Start with moderate penalty, increase if needed
        penalty_weights = [50.0, 100.0, 200.0]
        
        for pw in penalty_weights:
            try:
                result = minimize(
                    _penalty_objective,
                    v0,
                    method='L-BFGS-B',
                    jac=lambda v: _penalty_gradient(v, pw),
                    bounds=[(0.0, 1.0)] * (2 * N) + [(0.001, 0.5)] * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                v = result.x
                x = v[:N]
                y = v[N:2*N]
                r = v[2*N:]
                
                # Enforce strict feasibility
                r = np.clip(r, 0.001, 0.5)
                x = np.clip(x, r, 1.0 - r)
                y = np.clip(y, r, 1.0 - r)
                
                # Verify feasibility
                feasible = True
                for i in range(N):
                    if x[i] < r[i] - 1e-8 or x[i] > 1.0 - r[i] + 1e-8:
                        feasible = False
                    if y[i] < r[i] - 1e-8 or y[i] > 1.0 - r[i] + 1e-8:
                        feasible = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-8:
                            feasible = False
                
                if feasible:
                    val = np.sum(r)
                    if val > best_value:
                        best_value = val
                        best_v = np.column_stack([x, y, r])
                        
            except Exception:
                continue
    
    if best_v is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append((cx, cy, r))
        return np.array(cells[:N])
    
    return best_v
# EVOLVE-BLOCK-END
