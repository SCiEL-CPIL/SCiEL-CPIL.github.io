"""Rewritten program for the 50-circle packing task."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(c):
    """Compute the packing radius for centers c of shape (N, 2)."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall constraints
    wall = 0.5 - np.abs(c).max()
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    return float(min(wall, min_pair))

def _negative_score(c):
    return -_score(c)

def _initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _hard_disk_expansion(c, max_iters=500, tol=1e-12):
    """
    Expand circles using a hard-disk style iterative method.
    At each step, find the closest constraint and move centers apart.
    """
    c = c.copy().astype(float)
    for _ in range(max_iters):
        # Compute current radius
        wall = 0.5 - np.abs(c).max()
        diff = c[:, None, :] - c[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = dist.min() / 2.0
        r = min(wall, min_pair)
        
        if r < 1e-15:
            break
            
        # Find which constraint is active
        if wall <= min_pair:
            # Wall is the bottleneck; find the circle closest to the wall
            idx = np.unravel_index(np.abs(c).argmax(), c.shape)
            # Push it inward
            c[idx] -= np.sign(c[idx]) * 1e-6
        else:
            # Pair is the bottleneck
            i, j = np.unravel_index(dist.argmin(), dist.shape)
            if i == j:
                continue
            d = c[i] - c[j]
            norm = np.linalg.norm(d)
            if norm < 1e-15:
                # Random perturbation
                d = np.random.randn(2) * 1e-6
                norm = np.linalg.norm(d)
            d = d / norm
            # Move them apart by half the gap
            gap = (2 * r - norm) / 2.0
            if gap > 0:
                c[i] += d * gap * 0.5
                c[j] -= d * gap * 0.5
        # Clip to square
        c = np.clip(c, -0.5, 0.5)
        
    return c

def _polish(c, deadline):
    """
    Use L-BFGS-B to polish the packing, maximizing the radius.
    We optimize a smooth surrogate: the log-sum-exp of negative distances.
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 110:
        return c
    
    c = c.copy().astype(float)
    
    def smooth_obj(x):
        x = x.reshape(-1, 2)
        # Wall terms
        wall = 0.5 - np.abs(x).max()
        # Pairwise
        diff = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        # Smooth min of pairwise distances / 2
        beta = 100.0
        logsumexp = np.log(np.sum(np.exp(-beta * dist / 2))) / beta
        smooth_min_pair = -logsumexp
        return -(min(wall, smooth_min_pair))
    
    try:
        res = minimize(
            smooth_obj, c.ravel(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if res.success or res.fun < _negative_score(c):
            c = res.x.reshape(-1, 2)
            c = np.clip(c, -0.5, 0.5)
    except Exception:
        pass
    
    return c

def _random_perturbation(c, scale=0.01):
    c = c.copy()
    c += np.random.randn(*c.shape) * scale
    c = np.clip(c, -0.5, 0.5)
    return c

def _discrete_local_search(grid_res=40, max_iters=5000, deadline=None):
    """
    Discrete combinatorial approach:
    1. Create a fine grid of candidate positions
    2. Use local search to find N non-overlapping positions
    3. Refine with continuous optimization
    
    This avoids local minima by exploring the discrete space first.
    """
    if deadline is not None and time.time() - deadline[0] > 120:
        return _initial_grid()
    
    # Create fine grid
    ticks = np.linspace(-0.49, 0.49, grid_res)
    grid_points = np.array([(x, y) for y in ticks for x in ticks])
    n_points = len(grid_points)
    
    # Randomly select N points as initial solution
    rng = np.random.default_rng(42)
    selected_idx = rng.choice(n_points, size=N, replace=False)
    c = grid_points[selected_idx].copy()
    
    best_c = c.copy()
    best_score = _score(c)
    
    # Local search: try swapping or moving points to improve
    for iteration in range(max_iters):
        if deadline is not None and time.time() - deadline[0] > 125:
            break
        
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Try moving to a nearby grid point
        best_new_c = None
        best_new_score = best_score
        
        # Sample some candidate positions near current point
        for _ in range(50):
            # Random perturbation
            new_c = c.copy()
            new_c[i] = new_c[i] + rng.normal(0, 0.02, size=2)
            new_c[i] = np.clip(new_c[i], -0.5, 0.5)
            
            # Check if this improves the score
            s = _score(new_c)
            if s > best_new_score + 1e-12:
                best_new_score = s
                best_new_c = new_c
        
        if best_new_c is not None:
            c = best_new_c
            if best_new_score > best_score:
                best_score = best_new_score
                best_c = c.copy()
    
    return best_c

def _hybrid_refine(c, deadline):
    """
    Combine discrete local search with continuous refinement.
    """
    elapsed = time.time() - deadline[0]
    if elapsed > 115:
        return c
    
    # First do discrete local search
    c_discrete = _discrete_local_search(grid_res=30, max_iters=2000, deadline=deadline)
    s_discrete = _score(c_discrete)
    
    # Then refine with continuous methods
    c_continuous = _hard_disk_expansion(c_discrete, max_iters=100)
    s_continuous = _score(c_continuous)
    
    c_polished = _polish(c_continuous, deadline)
    s_polished = _score(c_polished)
    
    # Return the best
    if s_polished > s_continuous and s_polished > s_discrete:
        return c_polished
    elif s_continuous > s_discrete:
        return c_continuous
    else:
        return c_discrete

def construct():
    deadline = [time.time()]
    total_time_limit = 150.0
    
    best_c = _initial_grid()
    best_score = _score(best_c)
    
    # Phase 1: Hard-disk expansion from grid
    c = _hard_disk_expansion(best_c, max_iters=200)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 2: Polish with L-BFGS-B
    c = _polish(best_c, deadline)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Phase 3: Discrete combinatorial search
    elapsed = time.time() - deadline[0]
    if elapsed < 100:
        c = _discrete_local_search(grid_res=35, max_iters=3000, deadline=deadline)
        s = _score(c)
        if s > best_score:
            best_c = c.copy()
            best_score = s
        
        # Refine the discrete solution
        c = _hard_disk_expansion(c, max_iters=150)
        s = _score(c)
        if s > best_score:
            best_c = c.copy()
            best_score = s
        
        c = _polish(best_c, deadline)
        s = _score(c)
        if s > best_score:
            best_c = c.copy()
            best_score = s
    
    # Phase 4: Multiple hybrid restarts
    n_restarts = 15
    for i in range(n_restarts):
        elapsed = time.time() - deadline[0]
        if elapsed > 135:
            break
        
        # Create a perturbed version of the best solution
        c = _random_perturbation(best_c, scale=0.015 * (1 + i * 0.05))
        
        # Hybrid refinement
        c = _hybrid_refine(c, deadline)
        s = _score(c)
        if s > best_score:
            best_c = c.copy()
            best_score = s
    
    # Final polish pass
    c = _polish(best_c, deadline)
    s = _score(c)
    if s > best_score:
        best_c = c.copy()
        best_score = s
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
