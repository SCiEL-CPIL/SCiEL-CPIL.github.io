"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _clearance(x, y, r):
    """Distance from center (x,y) to the boundary of [0,1]^2, minus radius."""
    return np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)) - r

def _pairwise_clearance(x, y, r):
    """Minimum pairwise clearance: dist(i,j) - r_i - r_j for all i<j."""
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    # subtract radii sum
    d -= (r[:, None] + r[None, :])
    # set diagonal to large value
    np.fill_diagonal(d, np.inf)
    # return minimum over all pairs
    return d.min()

def _objective(params, x0, y0, r0, penalty=1e6):
    """
    Objective to minimize: -sum(r) + penalty * violation.
    params: [x, y, r] flattened
    """
    n = len(r0)
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Boundary clearance
    bnd = _clearance(x, y, r)
    bnd_viol = np.maximum(0, -bnd)
    
    # Pairwise clearance
    pw = _pairwise_clearance(x, y, r)
    pw_viol = max(0, -pw) if np.isfinite(pw) else 0
    
    # Objective
    obj = -np.sum(r) + penalty * (np.sum(bnd_viol**2) + pw_viol**2)
    return obj

def _hex_init(n):
    """Hexagonal packing initialization."""
    # For n circles in unit square, use hexagonal arrangement
    # Approximate: rows of circles
    # r ~ 1/(2*sqrt(3*n)) for hexagonal
    r_est = 1.0 / (2.0 * np.sqrt(3.0 * n))
    
    # Hexagonal spacing
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    # Number of columns
    cols = int(np.ceil(np.sqrt(n * 2.0 / np.sqrt(3))))
    rows = int(np.ceil(n / cols))
    
    # Scale to fit
    width = (cols - 1) * dx + 2 * r_est
    height = (rows - 1) * dy + 2 * r_est
    scale = min(1.0 / width, 1.0 / height)
    
    r_est *= scale
    dx *= scale
    dy *= scale
    
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x[idx] = r_est + col * dx + (0.5 * dx if row % 2 else 0)
            y[idx] = r_est + row * dy
            r[idx] = r_est
            idx += 1
    
    # Center the arrangement
    cx = (x.max() + x.min()) / 2
    cy = (y.max() + y.min()) / 2
    x += 0.5 - cx
    y += 0.5 - cy
    
    return x, y, r

def _grid_init(n):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x[idx] = (i + 0.5) / side
            y[idx] = (j + 0.5) / side
            r[idx] = 0.5 / side - 1e-3
            idx += 1
    
    return x, y, r

def _random_init(n, seed=42):
    """Random initialization with non-overlapping circles."""
    rng = np.random.RandomState(seed)
    
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    
    # Start with small circles and grow
    r_min = 0.01
    r_max = 0.1
    
    for i in range(n):
        for attempt in range(1000):
            xi = rng.uniform(r_max, 1 - r_max)
            yi = rng.uniform(r_max, 1 - r_max)
            ri = rng.uniform(r_min, r_max)
            
            # Check if it fits
            ok = True
            for j in range(i):
                dx = xi - x[j]
                dy = yi - y[j]
                if np.sqrt(dx*dx + dy*dy) < ri + r[j] + 1e-6:
                    ok = False
                    break
            
            if ok:
                x[i] = xi
                y[i] = yi
                r[i] = ri
                break
        else:
            # If we can't place, use a tiny circle
            x[i] = rng.uniform(0.05, 0.95)
            y[i] = rng.uniform(0.05, 0.95)
            r[i] = r_min * 0.5
    
    return x, y, r

def _push_apart(x, y, r, iterations=100, strength=0.5):
    """Physics-based push-apart to resolve overlaps."""
    n = len(x)
    x = x.copy()
    y = y.copy()
    
    for it in range(iterations):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    force = strength * overlap / dist
                    fx[i] += dx * force
                    fy[i] += dy * force
                    fx[j] -= dx * force
                    fy[j] -= dy * force
                elif dist <= 1e-10:
                    # Random nudge
                    fx[i] += rng.uniform(-0.01, 0.01)
                    fy[i] += rng.uniform(-0.01, 0.01)
                    fx[j] -= fx[i]
                    fy[j] -= fy[i]
        
        # Update positions
        x += fx
        y += fy
        
        # Clamp to boundaries
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
    
    return x, y

def _optimize_from_init(x0, y0, r0, penalty=1e6, max_iter=500, tol=1e-10):
    """Run optimization from a given initialization."""
    n = len(r0)
    x0_opt = x0.copy()
    y0_opt = y0.copy()
    r0_opt = r0.copy()
    
    # Ensure initial feasibility (push apart if needed)
    x0_opt, y0_opt = _push_apart(x0_opt, y0_opt, r0_opt, iterations=200, strength=0.3)
    
    params0 = np.concatenate([x0_opt, y0_opt, r0_opt])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
    for i in range(n):
        bounds.append((0, 1))  # y
    for i in range(n):
        bounds.append((0, 0.5))  # r
    
    # Try multiple penalty values
    best_params = params0.copy()
    best_obj = _objective(params0, x0_opt, y0_opt, r0_opt, penalty)
    
    for pen in [penalty, penalty * 10, penalty * 100]:
        try:
            result = minimize(
                _objective,
                params0,
                args=(x0_opt, y0_opt, r0_opt, pen),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract result
    x_res = best_params[:n]
    y_res = best_params[n:2*n]
    r_res = best_params[2*n:]
    
    # Final polish: ensure strict feasibility
    x_res, y_res = _push_apart(x_res, y_res, r_res, iterations=100, strength=0.1)
    
    # Shrink radii slightly to ensure no overlap
    r_res *= 0.999
    
    # Clamp to boundaries
    x_res = np.clip(x_res, r_res, 1 - r_res)
    y_res = np.clip(y_res, r_res, 1 - r_res)
    r_res = np.clip(r_res, 1e-6, 0.5)
    
    return x_res, y_res, r_res

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    # Try multiple initializations
    inits = []
    
    # Hexagonal
    x_h, y_h, r_h = _hex_init(N)
    inits.append(('hex', x_h, y_h, r_h))
    
    # Grid
    x_g, y_g, r_g = _grid_init(N)
    inits.append(('grid', x_g, y_g, r_g))
    
    # Random with different seeds
    for seed in [42, 123, 456, 789]:
        x_r, y_r, r_r = _random_init(N, seed=seed)
        inits.append((f'rand_{seed}', x_r, y_r, r_r))
    
    best_result = None
    best_score = -np.inf
    
    for name, x0, y0, r0 in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        try:
            x_res, y_res, r_res = _optimize_from_init(x0, y0, r0, penalty=1e6, max_iter=300, tol=1e-10)
            score = np.sum(r_res)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x_res, y_res, r_res])
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
