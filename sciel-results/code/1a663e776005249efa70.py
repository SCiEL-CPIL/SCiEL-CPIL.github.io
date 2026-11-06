"""Rewritten program for the 200-circle packing task."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(pts):
    """Compute the maximum common radius for given points."""
    # Pairwise distances
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _energy_and_grad(pts, target_r):
    """
    Compute a soft penalty energy and its gradient.
    Penalizes pair distances below 2*target_r and wall violations.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    np.fill_diagonal(dists, 1.0)
    
    # Pair penalty: smooth hinge for dist < 2*target_r
    margin = 2.0 * target_r
    viol_pair = np.maximum(0.0, margin - dists)
    # Gradient contribution from pairs
    grad = np.zeros_like(pts)
    mask = viol_pair > 0
    if np.any(mask):
        # For each violating pair (i,j), gradient on pts[i] is (pts[i]-pts[j])/dist * viol
        # Use symmetric formulation
        factor = viol_pair / dists
        np.fill_diagonal(factor, 0.0)
        grad -= np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    # Wall penalty: push points away from walls
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    wall_viol = np.maximum(0.0, 0.1 - wall_dists)  # Want at least 0.1 from walls
    energy_wall = np.sum(wall_viol ** 2)
    
    # Wall gradient
    grad_wall = np.zeros_like(pts)
    for i in range(n):
        x, y = pts[i]
        # dx/dx for left wall (0.5 - x)
        if 0.5 - x < 0.5 + x:  # closer to right wall
            if 0.5 - x < 0.1:
                grad_wall[i, 0] += 2.0 * (0.5 - x - 0.1) * (-1.0)
        else:  # closer to left wall (-0.5 + x)
            if x + 0.5 < 0.1:
                grad_wall[i, 0] += 2.0 * (x + 0.5 - 0.1) * (1.0)
        
        if 0.5 - y < 0.5 + y:  # closer to top wall
            if 0.5 - y < 0.1:
                grad_wall[i, 1] += 2.0 * (0.5 - y - 0.1) * (-1.0)
        else:  # closer to bottom wall
            if y + 0.5 < 0.1:
                grad_wall[i, 1] += 2.0 * (y + 0.5 - 0.1) * (1.0)
    
    energy_pair = np.sum(viol_pair ** 2)
    total_energy = energy_pair + 10.0 * energy_wall
    total_grad = grad + 10.0 * grad_wall
    
    return total_energy, total_grad

def _hex_grid_init(n):
    """Create a hexagonal grid initialization."""
    # Approximate hexagonal packing
    # For n points, use rows with alternating offsets
    points = []
    row = 0
    while len(points) < n:
        # Hex spacing
        dx = 1.0 / np.ceil(np.sqrt(n / 1.5))
        dy = dx * np.sqrt(3) / 2
        offset = dx / 2 if row % 2 == 1 else 0.0
        x_start = -0.5 + dx / 2 + offset
        col = 0
        while True:
            x = x_start + col * dx
            y = -0.5 + dy / 2 + row * dy
            if x > 0.5 or y > 0.5:
                break
            points.append([x, y])
            col += 1
            if len(points) >= n:
                break
        row += 1
    
    pts = np.array(points[:n])
    return pts

def _random_init(n, seed=None):
    """Random initialization inside the square."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return rng.uniform(-0.45, 0.45, size=(n, 2))

def _grid_init(n):
    """Simple grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].copy()

def _local_search(pts, target_r, max_iters=500, lr=0.001, seed=None):
    """
    Local search using gradient descent on the soft penalty.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    current = pts.copy()
    best = pts.copy()
    best_r = _compute_radius(pts)
    
    for it in range(max_iters):
        energy, grad = _energy_and_grad(current, target_r)
        
        # Adaptive learning rate
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-8:
            break
        
        # Gradient step
        step = lr * grad / (norm_grad + 1e-8)
        new_pts = current - step
        
        # Keep inside bounds
        new_pts = np.clip(new_pts, -0.49, 0.49)
        
        # Check if improved
        new_r = _compute_radius(new_pts)
        if new_r > best_r:
            best_r = new_r
            best = new_pts.copy()
            current = new_pts
            # Decay learning rate
            lr *= 0.99
        else:
            # Try smaller step
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best, best_r

def _nelder_mead_local(pts, max_iters=1000, seed=None):
    """
    Nelder-Mead local refinement.
    """
    from scipy.optimize import minimize
    
    def obj(x):
        p = x.reshape(N, 2)
        r = _compute_radius(p)
        return -r
    
    res = minimize(obj, pts.flatten(), method='Nelder-Mead', 
                   options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10})
    x_opt = res.x.reshape(N, 2)
    r = _compute_radius(x_opt)
    return x_opt, r

def construct():
    start_time = time.time()
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple initializations
    inits = []
    
    # 1. Hexagonal grid
    inits.append(_hex_grid_init(N))
    
    # 2. Regular grid
    inits.append(_grid_init(N))
    
    # 3. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1024, 2048, 4096, 8192]:
        inits.append(_random_init(N, seed))
    
    # Process each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        current_pts = init_pts.copy()
        current_r = _compute_radius(current_pts)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = current_pts.copy()
        
        # Local search with gradient descent
        refined_pts, refined_r = _local_search(
            current_pts, 
            target_r=current_r, 
            max_iters=300, 
            lr=0.005,
            seed=init_idx
        )
        
        if refined_r > best_r:
            best_r = refined_r
            best_pts = refined_pts.copy()
        
        # Nelder-Mead refinement if time permits
        if time.time() - start_time < DEADLINE - 30:
            nm_pts, nm_r = _nelder_mead_local(refined_pts, max_iters=500, seed=init_idx)
            if nm_r > best_r:
                best_r = nm_r
                best_pts = nm_pts.copy()
        
        # Use best result as starting point for next round
        current_pts = best_pts.copy()
        current_r = best_r
    
    # Final refinement on the best result
    if time.time() - start_time < DEADLINE - 10 and best_pts is not None:
        final_pts, final_r = _local_search(
            best_pts, 
            target_r=best_r, 
            max_iters=200, 
            lr=0.002,
            seed=999
        )
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _grid_init(N)
    
    # Final clip to ensure inside square
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    return best_pts
