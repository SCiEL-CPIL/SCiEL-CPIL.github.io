# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_radius(pts):
    """Compute the maximum radius for a set of points."""
    pts = np.asarray(pts)
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(pts), axis=1)
    min_wall = np.min(wall_dist)
    # Pairwise distances
    n = len(pts)
    min_pair = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts[i+1:], axis=1)
        if len(d) > 0:
            min_pair = min(min_pair, np.min(d))
    return min(min_wall, min_pair / 2.0)

def _build_bcc_init():
    """
    Generate a Body-Centered Cubic (BCC) like initialization.
    BCC is a dense packing structure. We place points in a grid and add body centers.
    For N=100, a 5x5x4 grid of corners + some centers might work, or just a perturbed grid.
    Let's try a 5x5x4 grid of points, which gives 100 points exactly.
    This is a simple cubic lattice, but we can perturb it slightly.
    """
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    if nx * ny * nz != N:
        # Fallback to random if dimensions don't match
        return np.random.rand(N, 3) - 0.5
    
    x = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    y = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    z = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nz)
    
    grid = np.array([(x[i], y[j], z[k]) for k in range(nz) for j in range(ny) for i in range(nx)])
    # Add small random noise to break symmetry
    grid += np.random.rand(N, 3) * 0.01 - 0.005
    return grid

def _build_random_init():
    """Generate a random initialization."""
    return np.random.rand(N, 3) - 0.5

def _soft_min_objective(params, beta=100.0):
    """
    Objective function for local polishing.
    Minimizes -soft_min of constraints.
    Constraints:
    - Pairwise distances
    - Wall distances * 2
    """
    pts = params.reshape(-1, 3)
    n = len(pts)
    
    # Pairwise distances
    pair_dists = []
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts[i+1:], axis=1)
        pair_dists.append(d)
    pair_dists = np.concatenate(pair_dists)
    
    # Wall constraints: 2 * (0.5 - |c_ik|)
    wall_constraints = 1.0 - 2.0 * np.abs(pts)
    wall_constraints = wall_constraints.flatten()
    
    all_constraints = np.concatenate([pair_dists, wall_constraints])
    
    # Soft min
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    sum_exp = np.sum(exp_terms)
    soft_min_val = max_c - np.log(sum_exp) / beta
    
    return -soft_min_val

def _soft_min_grad(params, beta=100.0):
    """Gradient of the soft-min objective."""
    pts = params.reshape(-1, 3)
    n = len(pts)
    
    pair_dists = []
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts[i+1:], axis=1)
        pair_dists.append(d)
    pair_dists = np.concatenate(pair_dists)
    
    wall_constraints = 1.0 - 2.0 * np.abs(pts)
    wall_constraints_flat = wall_constraints.flatten()
    
    all_constraints = np.concatenate([pair_dists, wall_constraints_flat])
    
    max_c = np.max(all_constraints)
    exp_terms = np.exp(-beta * (all_constraints - max_c))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(params)
    
    # Pairwise
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d_ij = pair_dists[idx]
            w = weights[idx]
            if d_ij > 1e-12:
                diff = pts[i] - pts[j]
                grad[i] += w * diff / d_ij
                grad[j] -= w * diff / d_ij
            idx += 1
            
    # Wall
    for i in range(n):
        for a in range(3):
            k = i * 3 + a
            w = weights[idx + k]
            c = pts[i, a]
            if abs(c) > 1e-12:
                sign_c = 1.0 if c > 0 else -1.0
                grad[i, a] += w * (-2.0 * sign_c)
                
    return grad

def _polish(pts, time_budget=5.0):
    """
    Polish a configuration using L-BFGS-B with a high beta soft-min.
    """
    start_time = time.time()
    params = pts.flatten()
    
    try:
        result = minimize(
            _soft_min_objective,
            params,
            method='L-BFGS-B',
            jac=_soft_min_grad,
            args=(100.0,),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        # Check time
        if time.time() - start_time > time_budget:
             return pts
             
        new_pts = result.x.reshape(-1, 3)
        # Ensure points are inside the cube (clip)
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        return new_pts
    except Exception:
        return pts

def _simulated_annealing(pts, time_budget=10.0):
    """
    Simulated Annealing to improve the packing.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    T = 0.01
    cooling = 0.95
    
    while time.time() - start_time < time_budget:
        # Pick a random point and move it
        i = np.random.randint(N)
        # Move in a random direction
        step_size = 0.01
        delta = (np.random.rand(3) - 0.5) * step_size
        new_pts = current_pts.copy()
        new_pts[i] += delta
        
        # Ensure inside cube
        if np.any(new_pts[i] > 0.5) or np.any(new_pts[i] < -0.5):
            continue
            
        new_r = _compute_radius(new_pts)
        
        if new_r > current_r or np.random.rand() < np.exp((new_r - current_r) / T):
            current_pts = new_pts
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
                
        T *= cooling
        if T < 1e-6:
            break
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_score = -np.inf
    
    # Initializations to try
    inits = []
    # BCC-like grid
    inits.append(_build_bcc_init())
    # Random
    for _ in range(3):
        inits.append(_build_random_init())
        
    for init in inits:
        if time.time() > deadline - 15.0:
            break
            
        # Phase 1: Simulated Annealing
        sa_time = 10.0
        if time.time() > deadline - sa_time - 10.0:
            continue
            
        sa_pts, sa_score = _simulated_annealing(init, time_budget=sa_time)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
        # Phase 2: Polish
        if time.time() > deadline - 10.0:
            continue
            
        polished_pts = _polish(sa_pts, time_budget=5.0)
        polished_score = _compute_radius(polished_pts)
        
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
            
    if best_pts is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_pts
# EVOLVE-BLOCK-END
