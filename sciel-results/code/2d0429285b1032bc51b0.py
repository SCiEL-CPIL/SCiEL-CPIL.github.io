"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    # Minimum pairwise distance using cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _energy_and_grad(centres, r_target):
    """
    Compute energy and gradient for L-BFGS-B optimization.
    Energy penalizes pairs closer than 2*r_target and points too close to walls.
    Uses a smooth approximation: for each pair, penalty = max(0, 2*r_target - d)^2
    For walls, penalty = max(0, r_target - wall_dist)^2
    """
    centres = centres.reshape(-1, 2)
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise penalty: only for pairs closer than 2*r_target
    violations = np.maximum(0, 2.0 * r_target - dists)
    energy = np.sum(violations**2)
    
    # Wall distances
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    wall_violations = np.maximum(0, r_target - wall_dists)
    energy += np.sum(wall_violations**2)
    
    # Gradient computation
    grad = np.zeros_like(centres)
    
    # Pairwise gradient
    # d/dx_i [max(0, 2r - d_ij)^2] where d_ij = |x_i - x_j|
    # = -2 * max(0, 2r - d_ij) * (x_i - x_j) / d_ij  for each j != i
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 2.0 * r_target and d > 1e-10:
                v = 2.0 * r_target - d
                grad[i] += -2.0 * v * (centres[i] - centres[j]) / d
    
    # Wall gradient
    for i in range(n):
        # Left wall: 0.5 + x_i
        if wall_dists[i] == (0.5 + centres[i, 0]):
            wv = r_target - (0.5 + centres[i, 0])
            if wv > 0:
                grad[i, 0] += -2.0 * wv
        # Right wall: 0.5 - x_i
        if wall_dists[i] == (0.5 - centres[i, 0]):
            wv = r_target - (0.5 - centres[i, 0])
            if wv > 0:
                grad[i, 0] += 2.0 * wv
        # Bottom wall: 0.5 + y_i
        if wall_dists[i] == (0.5 + centres[i, 1]):
            wv = r_target - (0.5 + centres[i, 1])
            if wv > 0:
                grad[i, 1] += -2.0 * wv
        # Top wall: 0.5 - y_i
        if wall_dists[i] == (0.5 - centres[i, 1]):
            wv = r_target - (0.5 - centres[i, 1])
            if wv > 0:
                grad[i, 1] += 2.0 * wv
    
    return energy, grad


def _hexagonal_init(n_rows=None, n_cols=None):
    """Create an optimized hexagonal packing initialization for N points."""
    if n_rows is None:
        # Estimate optimal rows for 200 points in hexagonal packing
        # Hexagonal packing has ~sqrt(3)/2 * rows * cols points
        # For 200 points, try rows around 15-16
        n_rows = 16
    if n_cols is None:
        # Estimate columns needed
        # Each row has either n_cols or n_cols-1 points
        # Total ≈ n_rows * n_cols - n_rows/2
        # 200 ≈ 16 * n_cols - 8 => n_cols ≈ 13
        n_cols = 13
    
    centres = []
    # Horizontal spacing
    s = 1.0 / n_cols
    # Vertical spacing for hexagonal
    s_vert = s * np.sqrt(3) / 2
    
    # Calculate extents
    y_extent = (n_rows - 1) * s_vert
    y_start = -0.5 + s_vert / 2 + (1.0 - y_extent) / 2
    
    for row in range(n_rows):
        y = y_start + row * s_vert
        if y < -0.5 + 0.01 or y > 0.5 - 0.01:
            continue
        if row % 2 == 0:
            x_extent = (n_cols - 1) * s
            x_start = -0.5 + s / 2 + (1.0 - x_extent) / 2
            for col in range(n_cols):
                x = x_start + col * s
                if -0.5 + 0.01 <= x <= 0.5 - 0.01:
                    centres.append((x, y))
        else:
            # Offset row
            x_extent = (n_cols - 2) * s
            if x_extent > 0:
                x_start = -0.5 + s + (1.0 - x_extent) / 2
                for col in range(n_cols - 1):
                    x = x_start + col * s
                    if -0.5 + 0.01 <= x <= 0.5 - 0.01:
                        centres.append((x, y))
    
    centres = np.array(centres)
    if len(centres) < N:
        # Try to add more by adjusting
        return _grid_init()
    return centres[:N]


def _grid_init():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init(seed):
    """Create a random initialization with some structure."""
    np.random.seed(seed)
    # Start with hexagonal and perturb
    base = _hexagonal_init()
    noise = np.random.uniform(-0.02, 0.02, size=base.shape)
    centres = base + noise
    centres = np.clip(centres, -0.49, 0.49)
    return centres


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1
    
    # Try various numbers of rows
    for n_rows in range(12, 22):
        for start_offset in [0, 1]:
            for n_cols in range(12, 20):
                # Calculate how many points fit in this configuration
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1
                
                if total_points < N:
                    continue
                
                # Calculate spacing
                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0
                
                s = min(s_horiz, s_vert)
                
                # Create the grid
                centres = []
                y_extent = (n_rows - 1) * s * np.sqrt(3) / 2
                y_start = -0.5 + (1.0 - y_extent) / 2 + s * np.sqrt(3) / 4
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_extent = (n_cols - 1) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_extent = (n_cols - 2) * s
                        x_start = -0.5 + (1.0 - x_extent) / 2 + s / 2
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                
                centres = np.array(centres)
                if len(centres) < N:
                    continue
                
                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres.copy()
    
    if best_centres is not None:
        return best_centres
    
    # Fallback to grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _l_bfgs_b_optimization(centres, r_target, deadline):
    """
    Use L-BFGS-B to minimize energy (violations) for a given target radius.
    Returns optimized centres and whether target was achieved.
    """
    def energy_func(x):
        e, g = _energy_and_grad(x, r_target)
        return e
    
    def grad_func(x):
        e, g = _energy_and_grad(x, r_target)
        return g
    
    x0 = centres.reshape(-1).copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            energy_func,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'maxls': 20,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 2000,
                'iprint': -1
            }
        )
        
        new_centres = result.x.reshape(-1, 2)
        # Ensure within bounds
        new_centres = np.clip(new_centres, -0.4999, 0.4999)
        
        # Check if target radius is achievable
        actual_r = _compute_radius(new_centres)
        achieved = actual_r >= r_target - 1e-6
        
        return new_centres, achieved, result.fun
    
    except Exception:
        return centres.copy(), False, np.inf


def _local_optimization(centres, deadline):
    """Local optimization using L-BFGS-B with binary search on radius."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Binary search on radius using L-BFGS-B feasibility
    r_low = best_r
    r_high = min(0.05, best_r * 1.5)  # Upper bound estimate
    
    for iteration in range(30):
        if time.time() > deadline:
            break
        
        r_mid = (r_low + r_high) / 2
        new_centres, achieved, energy = _l_bfgs_b_optimization(best_centres, r_mid, deadline)
        
        if achieved:
            r_low = r_mid
            if _compute_radius(new_centres) > best_r:
                best_r = _compute_radius(new_centres)
                best_centres = new_centres.copy()
        else:
            r_high = r_mid
        
        # If convergence is close, break
        if r_high - r_low < 1e-8:
            break
    
    return best_centres, best_r


def _polish_pass(centres, deadline):
    """High-precision polish using fine-grained L-BFGS-B with tight tolerance."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Try to improve by a small amount
    r_target = best_r * 1.001  # 0.1% improvement target
    new_centres, achieved, energy = _l_bfgs_b_optimization(best_centres, r_target, deadline)
    
    if achieved:
        actual_r = _compute_radius(new_centres)
        if actual_r > best_r:
            best_r = actual_r
            best_centres = new_centres.copy()
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3  # Leave 3s safety margin
    
    best_centres = None
    best_r = -1
    
    # Try multiple initializations for diversity
    initializations = []
    
    # 1. Hexagonal packing (best structured init)
    hex_centres = _hexagonal_packing()
    initializations.append(("hexagonal", hex_centres))
    
    # 2. Different hexagonal configurations
    for nr, nc in [(15, 14), (16, 13), (17, 12), (14, 15)]:
        try:
            h = _hexagonal_init(n_rows=nr, n_cols=nc)
            initializations.append((f"hex_{nr}x{nc}", h))
        except:
            pass
    
    # 3. Random perturbations of hexagonal
    for seed in [42, 123, 456, 789, 101]:
        perturbed = _random_init(seed)
        initializations.append((f"random_{seed}", perturbed))
    
    # 4. Grid initialization
    grid = _grid_init()
    initializations.append(("grid", grid))
    
    for name, centres in initializations:
        if time.time() > deadline - 5:
            break
        
        current_r = _compute_radius(centres)
        
        # L-BFGS-B based local optimization with binary search on radius
        if time.time() < deadline - 10:
            centres, current_r = _local_optimization(centres, deadline)
        
        # High-precision polish pass
        if time.time() < deadline - 3:
            centres, current_r = _polish_pass(centres, deadline)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
        
        # If we've found a good solution, stop early
        if best_r > 0.0365:
            break
    
    # Final check
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > 0:
            return best_centres
    
    # Fallback to a valid grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]
