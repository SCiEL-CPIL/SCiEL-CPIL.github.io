"""Rewritten program for the 50-circle packing task using SLSQP epigraph formulation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _true_radius(c):
    """Compute the exact packing radius for centers c of shape (N, 2)."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    wall = 0.5 - np.abs(c).max()
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    return float(min(wall, min_pair))

def _hex_init():
    """Create a hexagonal (triangular) lattice initialization."""
    # Hexagonal packing: rows offset by half spacing
    # For 50 circles, use roughly 8 columns x 7 rows with hex offset
    cols = 8
    rows = 7
    # Spacing to fit in [-0.5, 0.5]^2
    # Horizontal spacing: dx, vertical: dy = dx * sqrt(3)/2
    # Total width: (cols-1)*dx <= 1.0 => dx <= 1/(cols-1)
    # Total height: (rows-1)*dy + dy/2 (for offset) ... let's just compute
    dx = 1.0 / (cols - 1) * 0.95  # slight margin
    dy = dx * np.sqrt(3) / 2
    # Center the grid
    x_start = -0.5 + dx / 2
    y_start = -0.5 + dy / 2
    
    centers = []
    for row in range(rows):
        x_offset = 0.0 if row % 2 == 0 else dx / 2
        for col in range(cols):
            x = x_start + col * dx + x_offset
            y = y_start + row * dy
            centers.append([x, y])
    
    # We have rows*cols = 56 points, take first 50
    centers = np.array(centers[:N])
    # Center the whole thing
    centers -= centers.mean(axis=0)
    # Scale to fit in [-0.5, 0.5]
    max_abs = np.abs(centers).max()
    if max_abs > 0.48:
        centers *= 0.48 / max_abs
    return centers

def _grid_init():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init(seed=0):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.45, 0.45, size=(N, 2))
    return c

def _slsqp_optimize(c_init, deadline, maxiter=500):
    """
    Optimize using SLSQP on the epigraph formulation.
    Variables: [x_0, y_0, ..., x_{N-1}, y_{N-1}, r]
    Maximize r subject to:
      ||c_i - c_j||^2 >= (2r)^2 for all i < j
      |c_ik| <= 0.5 - r for all i, k
      r >= 0
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 130:
        return c_init
    
    x0 = np.concatenate([c_init.ravel(), [0.05]])  # start with r=0.05
    
    # Bounds: centers in [-0.5, 0.5], r in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Constraints: 
    # For each pair (i,j), i<j: ||c_i - c_j||^2 - (2r)^2 >= 0
    # For each center component: 0.5 - r - |c_ik| >= 0  =>  0.5 - r - c_ik >= 0 AND 0.5 - r + c_ik >= 0
    
    constraints = []
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            def pair_constraint(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                r = x[-1]
                return np.sum((ci - cj)**2) - (2*r)**2
            constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Wall constraints: 0.5 - r - |c_ik| >= 0
    # This is equivalent to: 0.5 - r - c_ik >= 0 AND 0.5 - r + c_ik >= 0
    for i in range(N):
        for k in range(2):
            def wall_pos(x, i=i, k=k):
                r = x[-1]
                return 0.5 - r - x[2*i + k]
            def wall_neg(x, i=i, k=k):
                r = x[-1]
                return 0.5 - r + x[2*i + k]
            constraints.append({'type': 'ineq', 'fun': wall_pos})
            constraints.append({'type': 'ineq', 'fun': wall_neg})
    
    def objective(x):
        return -x[-1]  # minimize -r = maximize r
    
    try:
        res = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_new = res.x[:2*N].reshape(N, 2)
            c_new = np.clip(c_new, -0.5, 0.5)
            r_new = res.x[-1]
            
            # Validate
            actual_r = _true_radius(c_new)
            if actual_r > 0:
                return c_new
    except Exception:
        pass
    
    return c_init

def _slsqp_optimize_fast(c_init, deadline, maxiter=200):
    """
    Faster SLSQP variant with fewer constraints evaluated per iteration.
    Uses a subset of constraints for speed, then validates.
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 135:
        return c_init
    
    x0 = np.concatenate([c_init.ravel(), [_true_radius(c_init) * 0.9]])
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Only include the most active constraints for speed
    # Compute current distances to identify active constraints
    c = c_init
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    wall = 0.5 - np.abs(c).max()
    r_current = min(wall, min_pair)
    
    # Get the k closest pairs
    k_closest = 100
    idx = np.argsort(dist, axis=None)
    # Filter out diagonal
    valid_idx = []
    for idx_val in idx:
        i, j = np.unravel_index(idx_val, dist.shape)
        if i != j and i < j:
            valid_idx.append((i, j))
        if len(valid_idx) >= k_closest:
            break
    
    constraints = []
    for i, j in valid_idx:
        def pair_constraint(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.sum((ci - cj)**2) - (2*r)**2
        constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Wall constraints for circles closest to walls
    for i in range(N):
        for k in range(2):
            def wall_pos(x, i=i, k=k):
                r = x[-1]
                return 0.5 - r - x[2*i + k]
            def wall_neg(x, i=i, k=k):
                r = x[-1]
                return 0.5 - r + x[2*i + k]
            constraints.append({'type': 'ineq', 'fun': wall_pos})
            constraints.append({'type': 'ineq', 'fun': wall_neg})
    
    def objective(x):
        return -x[-1]
    
    try:
        res = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_new = res.x[:2*N].reshape(N, 2)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            actual_r = _true_radius(c_new)
            if actual_r > 0:
                return c_new
    except Exception:
        pass
    
    return c_init

def _jiggle_and_optimize(c, deadline, scale=0.005):
    """
    Jiggle the configuration slightly and re-optimize to escape local minima.
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 140:
        return c
    
    c_jiggled = c.copy()
    c_jiggled += np.random.randn(*c.shape) * scale
    c_jiggled = np.clip(c_jiggled, -0.5, 0.5)
    
    # Optimize the jiggled version
    c_opt = _slsqp_optimize_fast(c_jiggled, deadline, maxiter=150)
    return c_opt

def _local_refine_slsqp(c, deadline):
    """
    Local refinement using SLSQP with all constraints, starting from good initial point.
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 140:
        return c
    
    r_current = _true_radius(c)
    x0 = np.concatenate([c.ravel(), [r_current * 1.01]])
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    constraints = []
    for i in range(N):
        for j in range(i + 1, N):
            def pair_constraint(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                r = x[-1]
                return np.sum((ci - cj)**2) - (2*r)**2
            constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    for i in range(N):
        for k in range(2):
            def wall_pos(x, i=i, k=k):
                r = x[-1]
                return 0.5 - r - x[2*i + k]
            def wall_neg(x, i=i, k=k):
                r = x[-1]
                return 0.5 - r + x[2*i + k]
            constraints.append({'type': 'ineq', 'fun': wall_pos})
            constraints.append({'type': 'ineq', 'fun': wall_neg})
    
    def objective(x):
        return -x[-1]
    
    try:
        res = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-13, 'disp': False}
        )
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_new = res.x[:2*N].reshape(N, 2)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            actual_r = _true_radius(c_new)
            if actual_r > 0:
                return c_new
    except Exception:
        pass
    
    return c

def construct():
    deadline = [time.time()]
    time_limit = 145.0  # safety margin before 150s
    
    best_c = None
    best_score = 0.0
    
    def try_update(c):
        nonlocal best_c, best_score
        if c is not None and c.shape == (N, 2) and np.all(np.isfinite(c)):
            s = _true_radius(c)
            if s > best_score:
                best_c = c.copy()
                best_score = s
    
    # Phase 1: Try multiple initializations
    initializations = [
        ("hex", _hex_init()),
        ("grid", _grid_init()),
        ("rand0", _random_init(0)),
        ("rand1", _random_init(1)),
        ("rand2", _random_init(42)),
    ]
    
    for name, c_init in initializations:
        elapsed = time.time() - deadline[0]
        if elapsed > 100:
            break
        
        # First do a fast optimization
        c_opt = _slsqp_optimize_fast(c_init, deadline, maxiter=100)
        try_update(c_opt)
        
        # Then do a full optimization on the best result so far
        if best_c is not None:
            c_full = _slsqp_optimize(best_c, deadline, maxiter=200)
            try_update(c_full)
        
        elapsed = time.time() - deadline[0]
        if elapsed > 120:
            break
    
    # Phase 2: Jiggle and re-optimize to escape local minima
    if best_c is not None:
        n_jiggles = 8
        for i in range(n_jiggles):
            elapsed = time.time() - deadline[0]
            if elapsed > 138:
                break
            
            scale = 0.003 + i * 0.002
            c_jig = _jiggle_and_optimize(best_c, deadline, scale=scale)
            try_update(c_jig)
            
            # If improved, do a full polish
            if best_score > 0.07:
                c_polish = _local_refine_slsqp(best_c, deadline)
                try_update(c_polish)
    
    # Phase 3: Final polish with maximum iterations
    if best_c is not None:
        elapsed = time.time() - deadline[0]
        if elapsed < 142:
            c_final = _local_refine_slsqp(best_c, deadline)
            try_update(c_final)
    
    # Ensure validity
    if best_c is None:
        best_c = _grid_init()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
