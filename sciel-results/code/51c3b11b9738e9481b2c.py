"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
PERIMETER = 4.0
MAX_WALL = 150.0  # evaluator limit; we target well below

def _hex_init(w, h, n):
    """Generate hexagonal lattice positions in [0,w]x[0,h]."""
    # Hexagonal packing: rows offset by half spacing
    # Try to fit as many circles as possible in hex pattern
    positions = []
    # Start with a reasonable spacing
    # For hex packing, vertical spacing = sqrt(3)/2 * horizontal spacing
    # We'll try different spacings and pick the best fit
    best_positions = None
    best_count = 0
    
    # Try various base spacings
    for s in np.linspace(0.1, 1.5, 30):
        h_spacing = s
        v_spacing = s * np.sqrt(3) / 2
        
        row = 0
        y = v_spacing / 2
        while y <= h - v_spacing / 2:
            offset = (row % 2) * h_spacing / 2
            x = h_spacing / 2 + offset
            while x <= w - h_spacing / 2:
                if len(positions) < n:
                    positions.append([x, y])
                x += h_spacing
            row += 1
            y += v_spacing
        
        if len(positions) >= n and len(positions) > best_count:
            best_count = len(positions)
            best_positions = positions[:n]
        
        positions = []
    
    # If hex didn't fill enough, fall back to grid
    if best_positions is None or len(best_positions) < n:
        cols = max(1, int(np.ceil(np.sqrt(n * w / h))))
        rows = max(1, int(np.ceil(n / cols)))
        for i in range(n):
            r_idx = i // cols
            c_idx = i % cols
            best_positions = [(c_idx + 0.5) * w / cols, (r_idx + 0.5) * h / rows]
    
    return np.array(best_positions[:n])

def _pack_for_aspect(w, h, max_iters=1000, tol=1e-8):
    """
    Pack N circles in rectangle [0,w]x[0,h], maximize sum of radii.
    Uses vectorized SLSQP with hexagonal initialization.
    """
    # Hexagonal initialization
    centers = _hex_init(w, h, N)
    
    # Initial radii: uniform small value
    r_init = 0.01 * np.ones(N)
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = r_init[i]
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
    
    # Precompute pair indices for non-overlap constraints
    pair_i = []
    pair_j = []
    for i in range(N):
        for j in range(i+1, N):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    
    def nonoverlap_constraints(v):
        """Vectorized non-overlap constraints: dist^2 - (ri+rj)^2 >= 0"""
        xi = v[3*pair_i]
        yi = v[3*pair_i+1]
        ri = v[3*pair_i+2]
        xj = v[3*pair_j]
        yj = v[3*pair_j+1]
        rj = v[3*pair_j+2]
        dx = xi - xj
        dy = yi - yj
        return dx*dx + dy*dy - (ri + rj)**2
    
    def boundary_constraints(v):
        """Vectorized boundary constraints: min(x-r, w-x-r, y-r, h-y-r) >= 0"""
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # For each circle, the constraint is min of 4 values
        # We use the minimum as a single constraint per circle
        cx_left = x - r
        cx_right = w - x - r
        cy_bottom = y - r
        cy_top = h - y - r
        # Return element-wise minimum (4N values, all must be >= 0)
        return np.minimum(np.minimum(cx_left, cx_right), np.minimum(cy_bottom, cy_top))
    
    def radius_constraints(v):
        """Radii must be non-negative"""
        return v[2::3]
    
    constraints = [
        {'type': 'ineq', 'fun': nonoverlap_constraints},
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': radius_constraints}
    ]
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, 0.5 * min(w, h)))
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': max_iters, 'ftol': tol})
    
    if result.success:
        return result.x
    return x0

def _validate_and_score(positions, w, h):
    """Validate packing and compute sum of radii."""
    n = len(positions) // 3
    x = positions[0::3]
    y = positions[1::3]
    r = positions[2::3]
    
    # Check non-negativity
    if np.any(r <= 0):
        return 0.0
    
    # Check boundary
    if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
        return 0.0
    
    # Check non-overlap (vectorized)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    
    return np.sum(r)

def construct():
    start_time = time.time()
    best_value = 0.0
    best_positions = None
    best_w = None
    best_h = None
    
    # Try various aspect ratios
    aspect_ratios = []
    for ar in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        # Perimeter = 2*(w+h) = 4, so w+h = 2
        # w/h = ar => w = ar*h => h*(ar+1) = 2 => h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        aspect_ratios.append((w, h))
    
    # Add some random perturbations around 1:1
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = 2.0 / (1.0 + rng.uniform(0.8, 1.2))
        w = 2.0 - h
        aspect_ratios.append((w, h))
    
    for w, h in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > MAX_WALL - 30:  # safety margin
            break
        
        try:
            positions = _pack_for_aspect(w, h, max_iters=800, tol=1e-7)
            value = _validate_and_score(positions, w, h)
            if value > best_value:
                best_value = value
                best_positions = positions
                best_w = w
                best_h = h
        except Exception as e:
            continue
    
    if best_positions is None:
        # Fallback: simple grid in 2x2 rectangle
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        return np.array(cells)
    
    # Construct output: [x, y, r] for each circle
    x = best_positions[0::3]
    y = best_positions[1::3]
    r = best_positions[2::3]
    
    result = np.column_stack([x, y, r])
    return result
