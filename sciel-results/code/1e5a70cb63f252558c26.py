"""Optimized packing of 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the maximum valid radius for given points in [-1/2, 1/2]^3."""
    if len(pts) < 2:
        return 0.5
    d = pdist(pts)
    min_pair_dist = d.min() / 2.0
    max_coord = np.abs(pts).max(axis=1)
    wall_dist = (0.5 - max_coord).min()
    return min(min_pair_dist, wall_dist)


def _repulsion_energy(pts, sigma=0.05):
    """
    Gaussian-kernel repulsion energy that smoothly aggregates all pair interactions.
    Lower energy = better packing. This is a smooth surrogate that the optimizer can
    minimize to resolve collisions.
    """
    pts = np.array(pts).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return 0.0
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    
    # Gaussian kernel repulsion
    energy = np.sum(np.exp(-dist_sq / (2 * sigma ** 2)))
    
    # Wall repulsion: push points away from boundaries
    wall_dist = 0.5 - np.abs(pts).max(axis=1)
    wall_energy = np.sum(np.exp(-wall_dist / sigma))
    
    return energy + 10.0 * wall_energy


def _repulsion_grad(pts, sigma=0.05):
    """Gradient of the repulsion energy."""
    pts = np.array(pts).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return np.zeros_like(pts)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    
    # Gradient of Gaussian kernel: -diff / sigma^2 * exp(-dist_sq / (2*sigma^2))
    kernel = np.exp(-dist_sq / (2 * sigma ** 2))
    np.fill_diagonal(kernel, 0)
    
    grad = np.zeros_like(pts)
    for i in range(n):
        grad[i] = -np.sum(diff[i] * kernel[i, :] / (sigma ** 2), axis=0)
    
    # Wall gradient
    wall_dist = 0.5 - np.abs(pts).max(axis=1)
    wall_kernel = np.exp(-wall_dist / sigma)
    for i in range(n):
        max_idx = np.argmax(np.abs(pts[i]))
        sign = np.sign(pts[i, max_idx])
        if sign == 0:
            sign = 1.0
        grad[i, max_idx] += 10.0 * wall_kernel[i] * sign / sigma
    
    return grad


def _grid_init(n):
    """Generate a grid initialization with various grid shapes."""
    best_pts = None
    best_r = 0
    
    # Try different grid configurations
    for m1 in range(3, 9):
        for m2 in range(3, 9):
            m3 = max(1, int(round(n / (m1 * m2))))
            if m1 * m2 * m3 < n:
                continue
            
            ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
            
            grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
            if len(grid) >= n:
                subset = grid[:n]
                r = _compute_radius(subset)
                if r > best_r:
                    best_r = r
                    best_pts = subset
    
    if best_pts is None:
        m = int(np.ceil(n ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:n]
        best_r = _compute_radius(best_pts)
    
    return best_pts, best_r


def _bcc_init(n):
    """Generate BCC lattice initialization."""
    best_pts = None
    best_r = 0
    
    for a in np.linspace(0.22, 0.38, 25):
        n_cells = int(1.0 / a)
        pts = []
        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                for k in range(n_cells + 1):
                    x = -0.5 + i * a
                    y = -0.5 + j * a
                    z = -0.5 + k * a
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        pts.append([x, y, z])
                    x2 = -0.5 + (i + 0.5) * a
                    y2 = -0.5 + (j + 0.5) * a
                    z2 = -0.5 + (k + 0.5) * a
                    if -0.5 <= x2 <= 0.5 and -0.5 <= y2 <= 0.5 and -0.5 <= z2 <= 0.5:
                        pts.append([x2, y2, z2])
        
        pts = np.array(pts)
        if len(pts) < n:
            continue
        
        # Select n points with best spacing using greedy approach
        np.random.seed(42)
        for trial in range(20):
            if trial == 0:
                subset = pts[:n]
            else:
                idx = np.random.choice(len(pts), n, replace=False)
                subset = pts[idx]
            
            r = _compute_radius(subset)
            if r > best_r:
                best_r = r
                best_pts = subset
    
    if best_pts is None:
        pts, r = _grid_init(n)
        return pts, r
    
    return best_pts, best_r


def _random_init(n, seed=None):
    """Generate random initialization."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(n, 3))
    r = _compute_radius(pts)
    return pts, r


def _local_polish(pts, deadline, max_iter=500):
    """
    High-precision local polish using exact constraint-based gradient descent.
    Focuses on the tightest constraints with adaptive step sizes.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    lr = 0.002
    stall_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Compute all pair distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Compute wall distances
        wall_dists = 0.5 - np.abs(pts).max(axis=1)
        
        # Find the tightest constraint
        min_pair_dist = dist.min()
        min_wall_dist = wall_dists.min()
        
        if min_pair_dist / 2.0 < min_wall_dist:
            # Tightest constraint is a pair
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            d = pts[i] - pts[j]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                direction_i = d / norm
                direction_j = -d / norm
            else:
                direction_i = np.array([1.0, 0, 0])
                direction_j = -direction_i
        else:
            # Tightest constraint is a wall
            i = np.argmin(wall_dists)
            max_idx = np.argmax(np.abs(pts[i]))
            sign = np.sign(pts[i, max_idx])
            if sign == 0:
                sign = 1.0
            direction_i = np.zeros(3)
            direction_i[max_idx] = -sign
            direction_j = None
        
        # Apply update
        step = lr
        pts[i] += step * direction_i
        if direction_j is not None:
            pts[j] += step * direction_j
        
        # Clip to cube
        pts = np.clip(pts, -0.5, 0.5)
        
        r = _compute_radius(pts)
        if r > best_r + 1e-15:
            best_r = r
            best_pts = pts.copy()
            stall_count = 0
            lr = min(lr * 1.05, 0.01)
        else:
            pts = best_pts.copy()
            lr *= 0.7
            stall_count += 1
            if stall_count > 30:
                break
    
    return best_pts, best_r


def _optimize_repulsion(pts, deadline, sigma=0.05, max_iter=2000):
    """
    Minimize repulsion energy using L-BFGS-B.
    This smooth objective allows the optimizer to resolve all collisions simultaneously.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = np.clip(p, -0.5, 0.5)
        return _repulsion_energy(p, sigma)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = np.clip(p, -0.5, 0.5)
        g = _repulsion_grad(p, sigma)
        return g.flatten()
    
    bounds = [(-0.5, 0.5)] * (n * 3)
    
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    res = minimize(
        objective, x0, method='L-BFGS-B', jac=gradient,
        bounds=bounds, 
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
    )
    
    new_pts = res.x.reshape(-1, 3)
    new_pts = np.clip(new_pts, -0.5, 0.5)
    new_r = _compute_radius(new_pts)
    
    if new_r > best_r:
        best_r = new_r
        best_pts = new_pts
    
    return best_pts, best_r


def _iterative_radius_target(pts, deadline):
    """
    Iteratively increase target radius and minimize repulsion for pairs closer than 2r.
    This escapes local minima by progressively tightening constraints.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Start slightly above current radius and increase
    target_r = current_r * 1.02
    max_target = 0.115  # Upper bound based on known best
    
    while target_r < max_target and time.time() < deadline - 15:
        # Minimize repulsion energy with a sigma tuned to the target radius
        sigma = max(0.01, target_r * 0.3)
        
        try:
            opt_pts, opt_r = _optimize_repulsion(pts, deadline, sigma=sigma, max_iter=800)
            
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
                pts = opt_pts.copy()
                # Increase target
                target_r = min(target_r * 1.01, opt_r * 1.02, max_target)
            else:
                # Try smaller increase
                target_r = min(target_r * 1.005, max_target)
                if target_r - current_r < 1e-6:
                    break
        except Exception:
            break
    
    return best_pts, best_r


def _anneal_and_refine(pts, deadline):
    """
    Simulated annealing with periodic local refinement.
    """
    pts = np.array(pts, dtype=np.float64).copy()
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    T = 0.03
    T_min = 0.001
    cooling = 0.985
    step_size = 0.015
    
    while time.time() < deadline - 10 and T > T_min:
        new_pts = pts.copy()
        idx = np.random.randint(n)
        delta = np.random.randn(3) * step_size
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        delta_r = new_r - best_r
        
        if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        
        T *= cooling
        
        if int((T_min / T) * 100) % 40 == 0 and time.time() < deadline - 15:
            refined_pts, refined_r = _local_polish(pts, deadline)
            if refined_r > best_r:
                best_r = refined_r
                best_pts = refined_pts.copy()
                pts = refined_pts.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Multiple initializations
    inits = []
    
    pts_grid, r_grid = _grid_init(N)
    inits.append((pts_grid, r_grid))
    
    pts_bcc, r_bcc = _bcc_init(N)
    inits.append((pts_bcc, r_bcc))
    
    for seed in [42, 123, 456, 789, 1000, 2000]:
        pts_rand, r_rand = _random_init(N, seed=seed)
        inits.append((pts_rand, r_rand))
    
    inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = inits[0][0]
    best_r = inits[0][1]
    
    # Phase 1: Repulsion-energy optimization from best initializations
    for init_idx, (init_pts, init_r) in enumerate(inits):
        if time.time() > deadline - 50:
            break
        
        if init_r < best_r * 0.80 and init_idx > 0:
            continue
        
        try:
            opt_pts, opt_r = _optimize_repulsion(init_pts, deadline, sigma=0.05, max_iter=1500)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts
        except Exception:
            continue
        
        if time.time() > deadline - 40:
            break
    
    # Phase 2: Iterative radius targeting to escape local minima
    if time.time() < deadline - 30:
        try:
            it_pts, it_r = _iterative_radius_target(best_pts, deadline)
            if it_r > best_r:
                best_r = it_r
                best_pts = it_pts
        except Exception:
            pass
    
    # Phase 3: Simulated annealing for global exploration
    if time.time() < deadline - 25:
        try:
            anneal_pts, anneal_r = _anneal_and_refine(best_pts, deadline)
            if anneal_r > best_r:
                best_r = anneal_r
                best_pts = anneal_pts
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        polished_pts, polished_r = _local_polish(best_pts, deadline, max_iter=800)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts
    
    # Final verification and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
