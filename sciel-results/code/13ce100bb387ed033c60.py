"""Circle packing in unit square - 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi, QhullError

def _is_valid(x, y, r, tol=1e-6):
    """Check if circles are valid (inside unit square, no overlaps)."""
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d[iu] < r_sum[iu] - tol):
        return False
    return True

def _fix_invalid(x, y, r):
    """Shrink radii to fix any violations, then try to grow back."""
    r = np.maximum(r, 0)
    # Iteratively shrink to fix overlaps and boundary violations
    for _ in range(100):
        changed = False
        # Boundary
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] > max_r + 1e-9:
                r[i] = max(0, max_r - 1e-7)
                changed = True
        # Overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        for (i, j) in zip(iu[0], iu[1]):
            if d[i, j] < r[i] + r[j] + 1e-9:
                # Split the violation
                excess = r[i] + r[j] - d[i, j]
                # Shrink the larger one more
                if r[i] >= r[j]:
                    r[i] = max(0, r[i] - excess * 0.6 - 1e-7)
                    r[j] = max(0, r[j] - excess * 0.4 - 1e-7)
                else:
                    r[j] = max(0, r[j] - excess * 0.6 - 1e-7)
                    r[i] = max(0, r[i] - excess * 0.4 - 1e-7)
                changed = True
        if not changed:
            break
    return r

def _voronoi_radii(x, y):
    """Compute max radius for each circle based on Voronoi cell."""
    points = np.column_stack([x, y])
    try:
        vor = Voronoi(points)
        radii = np.zeros(N)
        for i in range(N):
            # Get the Voronoi region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0 or -1 in region:
                # Point on convex hull - use distance to boundary
                radii[i] = min(x[i], 1 - x[i], y[i], 1 - y[i])
            else:
                # Get vertices of the Voronoi cell
                verts = vor.vertices[region]
                # Compute distance from point to each edge of the cell
                min_dist = np.inf
                for k in range(len(verts)):
                    v1 = verts[k]
                    v2 = verts[(k + 1) % len(verts)]
                    # Distance from point to line segment v1-v2
                    # Project point onto line
                    d_vec = v2 - v1
                    d_len2 = np.dot(d_vec, d_vec)
                    if d_len2 < 1e-15:
                        dist = np.linalg.norm(points[i] - v1)
                    else:
                        t = np.dot(points[i] - v1, d_vec) / d_len2
                        t = np.clip(t, 0, 1)
                        proj = v1 + t * d_vec
                        dist = np.linalg.norm(points[i] - proj)
                    min_dist = min(min_dist, dist)
                # Also check distance to unit square boundaries
                min_dist = min(min_dist, x[i], 1 - x[i], y[i], 1 - y[i])
                radii[i] = min_dist - 1e-7
        return radii
    except QhullError:
        # Fallback: just use distance to nearest neighbor / 2
        radii = np.zeros(N)
        for i in range(N):
            dists = np.sqrt((x - x[i])**2 + (y - y[i])**2)
            dists[i] = np.inf
            nearest = np.min(dists)
            radii[i] = min(nearest / 2, x[i], 1 - x[i], y[i], 1 - y[i]) - 1e-7
        return radii

def _objective(params):
    """Objective: minimize negative sum of radii + penalty for violations."""
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    score = -np.sum(r)
    
    # Boundary penalties
    px_neg = np.maximum(0, r - x)
    px_pos = np.maximum(0, r - (1 - x))
    py_neg = np.maximum(0, r - y)
    py_pos = np.maximum(0, r - (1 - y))
    boundary_pen = np.sum(px_neg**2) + np.sum(px_pos**2) + np.sum(py_neg**2) + np.sum(py_pos**2)
    
    # Overlap penalties
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    d = np.sqrt(d2)
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    r_pairs = r_sum[iu]
    violations = r_pairs - d_pairs
    pos_viol = np.maximum(0, violations)
    overlap_pen = np.sum(pos_viol**2)
    
    return score + 50.0 * (boundary_pen + overlap_pen)

def _make_hex_initial():
    """Hexagonal close-packed initial configuration."""
    x = np.zeros(N)
    y = np.zeros(N)
    # Hexagonal grid: rows offset by half spacing
    # Try to fit 26 circles in hex pattern
    # 5 columns x 5 rows = 25, plus 1 extra
    cols = 5
    rows = 5
    spacing = 0.8 / (cols - 1) if cols > 1 else 0.8
    hex_dy = spacing * np.sqrt(3) / 2
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = 0.1 + col * spacing
        y[i] = 0.1 + row * hex_dy
        if row % 2 == 1:
            x[i] += spacing / 2
    # Clamp to [0.1, 0.9]
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    r = np.full(N, 0.08)
    return np.concatenate([x, y, r])

def _make_grid_initial():
    """Grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(0.1, 0.9, side)
    ys = np.linspace(0.1, 0.9, side)
    x, y = np.meshgrid(xs, ys)
    x = x.flatten()
    y = y.flatten()
    x = x[:N]
    y = y[:N]
    r = np.full(N, 0.08)
    return np.concatenate([x, y, r])

def _make_random_initial(seed=None):
    """Random initial configuration."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    return np.concatenate([x, y, r])

def _optimize_positions(params, deadline):
    """Optimize circle positions and radii using L-BFGS-B."""
    bounds = [(0.01, 0.99)] * (2*N) + [(0.01, 0.45)] * N
    res = minimize(_objective, params, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'maxfun': 100000, 'ftol': 1e-12, 'gtol': 1e-8})
    return res.x

def _local_refine(x, y, r, deadline):
    """Refine radii given fixed positions using Voronoi."""
    # Compute Voronoi-based max radii
    max_r = _voronoi_radii(x, y)
    # Use the minimum of current and max
    r_new = np.minimum(r, max_r)
    # Try to grow: set r to max_r
    r_grown = np.maximum(0, max_r)
    # Check validity
    if _is_valid(x, y, r_grown):
        return r_grown
    # If not valid, use current r
    return r_new

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    best_result = None
    best_score = -np.inf
    
    def update_best(x, y, r):
        nonlocal best_result, best_score
        r = np.maximum(0, r)
        if _is_valid(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x, y, r))
    
    # Strategy 1: Start with hexagonal packing, then optimize
    p0 = _make_hex_initial()
    if time.time() < deadline - 5:
        params_opt = _optimize_positions(p0, deadline)
        x, y, r = params_opt[:N], params_opt[N:2*N], params_opt[2*N:3*N]
        r = _fix_invalid(x, y, r)
        update_best(x, y, r)
    
    # Strategy 2: Start with grid, then optimize
    p0 = _make_grid_initial()
    if time.time() < deadline - 5:
        params_opt = _optimize_positions(p0, deadline)
        x, y, r = params_opt[:N], params_opt[N:2*N], params_opt[2*N:3*N]
        r = _fix_invalid(x, y, r)
        update_best(x, y, r)
    
    # Strategy 3: Multiple random restarts with local optimization
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        p0 = _make_random_initial(seed)
        params_opt = _optimize_positions(p0, deadline)
        x, y, r = params_opt[:N], params_opt[N:2*N], params_opt[2*N:3*N]
        r = _fix_invalid(x, y, r)
        update_best(x, y, r)
    
    # Strategy 4: If we have a good result, refine radii using Voronoi
    if best_result is not None and time.time() < deadline - 5:
        x, y, r = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        r_refined = _local_refine(x, y, r, deadline)
        update_best(x, y, r_refined)
    
    # Strategy 5: Perturbation-based local search on best result
    if best_result is not None and time.time() < deadline - 5:
        x, y, r = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Perturb positions slightly
            x_p = x + np.random.normal(0, 0.02, N)
            y_p = y + np.random.normal(0, 0.02, N)
            x_p = np.clip(x_p, 0.05, 0.95)
            y_p = np.clip(y_p, 0.05, 0.95)
            params = np.concatenate([x_p, y_p, r])
            params_opt = _optimize_positions(params, deadline)
            x_o, y_o, r_o = params_opt[:N], params_opt[N:2*N], params_opt[2*N:3*N]
            r_o = _fix_invalid(x_o, y_o, r_o)
            update_best(x_o, y_o, r_o)
    
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
