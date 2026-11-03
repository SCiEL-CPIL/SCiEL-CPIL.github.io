import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def _validate_solution(z):
    """Check if solution is valid: no overlaps, all inside unit square."""
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    # Check boundary
    if np.any(r > x) or np.any(r > 1 - x) or np.any(r > y) or np.any(r > 1 - y):
        return False
    if np.any(r <= 0):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            rr = r[i] + r[j]
            if d2 < rr * rr - 1e-10:
                return False
    return True

def _sum_radii(z):
    return np.sum(z[:, 2])

def _delaunay_init(rng):
    """
    Generate initial configuration using Delaunay triangulation.
    This provides a much better starting configuration than simple grid,
    allowing the local optimizer to converge to a higher quality local maximum.
    """
    # Generate random points in unit square
    pts = rng.uniform(0.05, 0.95, size=(N, 2))
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(pts)
    except Exception:
        # Fallback to grid if Delaunay fails
        side = int(np.ceil(np.sqrt(N)))
        x = np.linspace(0.1, 0.9, side)
        y = np.linspace(0.1, 0.9, side)
        grid_pts = np.array([[xi, yi] for xi in x for yi in y])[:N]
        # Estimate radii
        r = np.full(N, 0.1)
        for i in range(N):
            min_d = min(grid_pts[i, 0], 1 - grid_pts[i, 0], 
                       grid_pts[i, 1], 1 - grid_pts[i, 1])
            for j in range(N):
                if i != j:
                    d = np.hypot(grid_pts[i, 0] - grid_pts[j, 0], 
                                grid_pts[i, 1] - grid_pts[j, 1])
                    min_d = min(min_d, d)
            r[i] = 0.5 * min_d * 0.9
        return np.column_stack((grid_pts[:, 0], grid_pts[:, 1], r))
    
    # For each point, compute minimum distance to neighbors and walls
    r = np.zeros(N)
    for i in range(N):
        min_d = min(pts[i, 0], 1 - pts[i, 0], 
                   pts[i, 1], 1 - pts[i, 1])
        
        # Check Delaunay neighbors
        for simplex in tri.simplices:
            if i in simplex:
                for j in simplex:
                    if i != j:
                        d = np.hypot(pts[i, 0] - pts[j, 0], 
                                    pts[i, 1] - pts[j, 1])
                        min_d = min(min_d, d)
        
        r[i] = 0.5 * min_d * 0.95
    
    return np.column_stack((pts[:, 0], pts[:, 1], r))

def _hex_init(rng, jitter=0.0):
    """Generate hexagonal arrangement as alternative initialization."""
    pts = []
    rows = 0
    while len(pts) < N:
        row = rows
        y = row * 0.8660254
        offset = (0.5 if row % 2 else 0.0)
        col = 0
        while len(pts) < N:
            x = col + offset
            pts.append([x, y])
            col += 1
        rows += 1
    
    pts = np.array(pts[:N])
    # Normalize to unit square
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pts = (pts - [xmin, ymin]) / span
    
    # Center in [0,1]
    cx = (xmax - xmin) / span
    cy = (ymax - ymin) / span
    pts[:, 0] = 0.5 - cx / 2 + pts[:, 0] * cx
    pts[:, 1] = 0.5 - cy / 2 + pts[:, 1] * cy
    
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    
    # Estimate radii
    r = np.full(N, 0.15)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], 
                                         pts[i, 1] - pts[j, 1]))
        r[i] = 0.5 * dmin * 0.98
    
    return np.column_stack((pts[:, 0], pts[:, 1], r))

def _project_feasible(z, iters=50):
    """Heuristic repair: shrink radii / nudge to satisfy constraints."""
    z = z.copy()
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    
    for _ in range(iters):
        # Clamp to walls by shrinking radius and centering
        for i in range(N):
            x_i, y_i, r_i = x[i], y[i], r[i]
            # Wall constraints
            r_i = min(r_i, x_i, 1.0 - x_i, y_i, 1.0 - y_i)
            r_i = max(r_i, 1e-6)
            # Nudge center to be inside if it was outside
            x_i = min(max(x_i, r_i), 1.0 - r_i)
            y_i = min(max(y_i, r_i), 1.0 - r_i)
            x[i], y[i], r[i] = x_i, y_i, r_i
        
        # Resolve overlaps by shrinking the smaller of the two radii
        changed = False
        for _ in range(5):
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = r[i] + r[j]
                    if d < need - 1e-10:
                        # Shrink both proportionally
                        deficit = need - d
                        s = max(0, (d - 1e-10) / (need + 1e-10))
                        r[i] = max(r[i] * s, 1e-6)
                        r[j] = max(r[j] * s, 1e-6)
                        changed = True
            if not changed:
                break
        if not changed:
            break
    
    z = np.column_stack((x, y, r))
    return z

def _soft_objective(z, lam):
    """z = (N, 3) array of (x, y, r). Minimize -sum(r) + lam*(overlap + wall penalties)."""
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    base = -np.sum(r)
    pen = 0.0
    
    # Wall penalties (smooth)
    pen += np.sum(np.maximum(r - x, 0.0) ** 2)
    pen += np.sum(np.maximum(x + r - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(r - y, 0.0) ** 2)
    pen += np.sum(np.maximum(y + r - 1.0, 0.0) ** 2)
    
    # Pairwise overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            need = r[i] + r[j]
            if d2 < need * need:
                d = np.sqrt(max(d2, 1e-12))
                pen += (need - d) ** 2
    
    return base + lam * pen

def _soft_grad(z, lam):
    """Analytical gradient for _soft_objective."""
    x, y, r = z[:, 0], z[:, 1], z[:, 2]
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # Base: d(-sum r)/dr = -1
    gr -= 1.0
    
    # Wall: term (r-x)^2 when r>x
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    
    # Pairwise
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            need = r[i] + r[j]
            if d2 < need * need:
                d = np.sqrt(max(d2, 1e-12))
                f = need - d  # >0
                inv = 1.0 / d
                gx[i] += lam * 2.0 * f * (-dx * inv)
                gx[j] += lam * 2.0 * f * (dx * inv)
                gy[i] += lam * 2.0 * f * (-dy * inv)
                gy[j] += lam * 2.0 * f * (dy * inv)
                gr[i] += lam * 2.0 * f * (1.0)
                gr[j] += lam * 2.0 * f * (1.0)
    
    grad = np.column_stack((gx, gy, gr))
    return grad.flatten()

def _optimize(z0, lam, max_iter=300):
    """Optimize using L-BFGS-B with analytical gradient."""
    z0_flat = z0.flatten()
    
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.4))  # r
    
    try:
        res = minimize(_soft_objective, z0_flat, jac=_soft_grad, 
                      method='L-BFGS-B', bounds=bounds, args=(lam,),
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        z_opt = res.x.reshape(N, 3)
        return z_opt
    except Exception:
        return z0

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = -np.inf
    best_solution = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # Delaunay-based inits (primary strategy)
    for seed in range(15):
        local_rng = np.random.default_rng(seed * 1000)
        z = _delaunay_init(local_rng)
        inits.append(z)
    
    # Hexagonal inits (secondary strategy)
    for jitter in [0.0, 0.02, 0.05, 0.1]:
        z = _hex_init(rng, jitter=jitter)
        inits.append(z)
    
    # Random inits
    for _ in range(5):
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.1)
        z = np.column_stack((x, y, r))
        inits.append(z)
    
    # Process each init
    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout:
            break
        
        z = init.copy()
        
        # Project to feasibility first
        z = _project_feasible(z)
        
        # Optimize with moderate penalty
        z = _optimize(z, lam=10.0, max_iter=200)
        
        # Project again and validate
        z = _project_feasible(z)
        
        if _validate_solution(z):
            score = _sum_radii(z)
            if score > best_score:
                best_score = score
                best_solution = z.copy()
        
        # If time allows, do refinement with lower penalty
        if time.time() - start_time < timeout - 20:
            if best_solution is not None:
                z_refine = best_solution.copy()
                z_refine = _optimize(z_refine, lam=5.0, max_iter=150)
                z_refine = _project_feasible(z_refine)
                
                if _validate_solution(z_refine):
                    score = _sum_radii(z_refine)
                    if score > best_score:
                        best_score = score
                        best_solution = z_refine.copy()
    
    # Final polish if we have a solution
    if best_solution is not None and time.time() - start_time < timeout - 5:
        final = _optimize(best_solution, lam=1.0, max_iter=100)
        final = _project_feasible(final)
        
        if _validate_solution(final):
            score = _sum_radii(final)
            if score > best_score:
                best_score = score
                best_solution = final.copy()
    
    if best_solution is None:
        # Fallback to trivial seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
