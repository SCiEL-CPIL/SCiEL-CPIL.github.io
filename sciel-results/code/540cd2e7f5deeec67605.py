import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # safety margin before 150s

def _penalty_objective(v, lam_b, lam_o):
    """v: [x(26), y(26), r(26)]. Minimize -sum(r) + penalties."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Objective: maximize sum(r) → minimize -sum(r)
    obj = -r.sum()
    
    # Boundary penalty
    b = np.maximum(0, r - x) + np.maximum(0, r - (1.0 - x)) + \
        np.maximum(0, r - y) + np.maximum(0, r - (1.0 - y))
    obj += lam_b * b.sum()
    
    # Overlap penalty (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rsum = r[:, None] + r[None, :]
    # Only upper triangle
    overlap = np.maximum(0, rsum - np.sqrt(np.maximum(0, dist2)))
    # Zero out diagonal
    np.fill_diagonal(overlap, 0.0)
    obj += lam_o * (overlap**2).sum()
    
    return obj

def _is_valid(arr, tol=1e-6):
    """Check if arrangement is valid (all constraints satisfied)."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _make_hex_initial():
    """Hexagonal packing initial guess."""
    # Hexagonal pattern: rows offset
    cols = 5
    rows = 6  # 5+5+5+5+4+2 = 26? Let's do 5 cols x 5 rows = 25, plus 1
    # Better: use a hex grid with spacing
    # For hex packing, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # Let's just place them in a hex-like pattern
    pts = []
    # 5 columns, 5 rows, offset every other row
    for row in range(5):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.2
            pts.append([x, y])
    # Add one more
    pts.append([0.5, 0.5])
    # Adjust to fit in [0,1]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Scale to fit
    scale = 0.8 / max(xmax-xmin, ymax-ymin)
    cx = (xmin+xmax)/2
    cy = (ymin+ymax)/2
    for p in pts:
        p[0] = 0.5 + (p[0]-cx)*scale
        p[1] = 0.5 + (p[1]-cy)*scale
    
    # Estimate radius as half min distance
    min_dist = np.inf
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.9
    # Also bound by distance to walls
    for i in range(len(pts)):
        r0 = min(r0, pts[i][0], 1-pts[i][0], pts[i][1], 1-pts[i][1])
    
    arr = np.array(pts)
    radii = np.full(N, r0)
    return np.column_stack([arr[:,0], arr[:,1], radii])

def _make_grid_initial():
    """Grid initial guess."""
    side = 6 # 6x6 = 36 points, enough for 26
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i+0.5)/side, (j+0.5)/side])
    # Shuffle and take first N to get a random subset of the grid? Or just take first N?
    # Taking first N from a grid gives a rectangular block. Better to spread them out.
    # Let's just take every k-th point or random selection.
    # For simplicity and determinism, let's just take the first N.
    pts = pts[:N]
    arr = np.array(pts)
    # Radius: half the cell size, with margin
    r0 = 0.5/side * 0.9
    for i in range(N):
        r0 = min(r0, arr[i,0], 1-arr[i,0], arr[i,1], 1-arr[i,1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:,0], arr[:,1], radii])

def _make_random_initial(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    # Random positions in [0.1, 0.9]
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    # Estimate radius from nearest neighbor
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    r = 0.4 * min_dists
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    r = np.maximum(r, 0.01)
    return np.column_stack([x, y, r])

def _optimize(arr_init, lam_b, lam_o, maxiter=200):
    """Run optimization from initial guess."""
    v0 = arr_init.flatten()
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    res = minimize(
        _penalty_objective, v0,
        args=(lam_b, lam_o),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12}
    )
    
    return res.x.reshape(N, 3)

def construct():
    start_time = time.time()
    
    best_arr = None
    best_val = -np.inf
    
    # Generate initial guesses
    inits = []
    inits.append(('hex', _make_hex_initial()))
    inits.append(('grid', _make_grid_initial()))
    
    # Several random inits
    for seed in [42, 123, 456, 789, 1024, 2048]:
        inits.append((f'rand_{seed}', _make_random_initial(seed)))
    
    # Also try with different lambda values
    lambda_sets = [
        (10.0, 100.0),
        (50.0, 500.0),
        (100.0, 1000.0),
    ]
    
    for lam_b, lam_o in lambda_sets:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for name, arr_init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            try:
                arr = _optimize(arr_init, lam_b, lam_o)
            except Exception:
                continue
            
            # Check validity
            if _is_valid(arr):
                val = arr[:, 2].sum()
                if val > best_val:
                    best_val = val
                    best_arr = arr.copy()
    
    # If no valid solution found, try with looser validity
    if best_arr is None:
        # Try the grid as fallback
        best_arr = _make_grid_initial()
        best_val = best_arr[:, 2].sum()
    
    # Final check and cleanup
    if _is_valid(best_arr):
        return best_arr
    
    # If still invalid, try to fix by reducing radii
    x, y, r = best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]
    # Reduce radii to satisfy boundary
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    r = np.maximum(r, 0.0)
    # Iteratively reduce to fix overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j]:
                    # Reduce both
                    excess = r[i] + r[j] - d
                    r[i] = max(0, r[i] - excess/2)
                    r[j] = max(0, r[j] - excess/2)
                    changed = True
        if not changed:
            break
    best_arr = np.column_stack([x, y, r])
    
    if _is_valid(best_arr):
        return best_arr
    
    # Last resort: return grid
    return _make_grid_initial()
