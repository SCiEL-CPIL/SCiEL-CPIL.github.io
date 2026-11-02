"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, tau):
    """
    Compute the soft-min objective and its gradient.
    We minimize -soft_min(min_pair_dist/2, min_wall_dist)
    which is equivalent to maximizing the soft-min of the constraints.
    """
    centers = centers_flat.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # y values: constraints
    y_pair = dists / 2.0
    y_wall = wall_dists
    
    # Mask for pairwise (i != j)
    mask = ~np.eye(N, dtype=bool)
    
    # Weights
    w_pair = np.exp(-y_pair / tau)
    w_pair[~mask] = 0.0
    
    w_wall = np.exp(-y_wall / tau)
    
    S = np.sum(w_pair) + np.sum(w_wall)
    
    if S == 0:
        return 0.0, np.zeros_like(centers_flat)
    
    # Objective value: -tau * log(S)
    obj_val = -tau * np.log(S)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise contribution
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = np.sum(w_pair[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall contribution
    signs = np.sign(centers)
    grad_wall = w_wall * signs
    
    grad = -(grad_pair + grad_wall) / S
    
    return obj_val, grad.flatten()

def _build_grid():
    """Build a cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _build_bcc_init():
    """Build a Body-Centered Cubic (BCC) initialization."""
    # BCC has 2 atoms per cubic unit.
    # We need 100 spheres. Let's try a 4x4x4 grid of BCC cells?
    # 4*4*4 = 64 cells * 2 = 128 spheres. Too many.
    # 3x4x4 = 48 cells * 2 = 96 spheres. Close.
    # 4x4x3 = 48 cells * 2 = 96.
    # 5x4x4 = 80 cells * 2 = 160.
    
    # Let's just use a simple grid for now, BCC is tricky to fit exactly 100.
    # A better approach: Generate a BCC lattice and take the first 100 points
    # that fit in a cube, then scale/center.
    
    # Lattice vectors for BCC
    # a = 1
    # Points: (0,0,0) and (0.5, 0.5, 0.5) in unit cube
    # Let's generate a grid of unit cubes and collect points.
    
    points = []
    # Estimate grid size: (100/2)^(1/3) ~ 3.6 -> 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner point
                points.append([i, j, k])
                # Center point
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    
    # We have 128 points. We need 100.
    # This is a subset. We should probably just use the grid init or random.
    # For now, return the grid init if BCC is not easily trimmed.
    # Actually, let's just return the grid for simplicity in this structural change,
    # or use a Halton sequence which is more uniform.
    
    return _build_halton_init()

def _build_halton_init():
    """Build a Halton sequence initialization for better uniformity."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index = index // base
        return r
    
    points = []
    # Skip first few points (0,0,0)
    for i in range(1, N + 1):
        points.append([halton(i, 2), halton(i, 3), halton(i, 5)])
    
    points = np.array(points)
    # Scale to fit in [-0.5, 0.5]
    # Halton points are in [0,1]
    # Center and scale
    points = (points - 0.5) * 0.9 # Slight margin from walls
    return points

def _optimize_from_initial(initial_centers, deadline):
    """Run multi-phase optimization from an initial configuration."""
    x0 = initial_centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    x_curr = x0
    
    # Phase 1: Smooth (high tau)
    tau1 = 0.1
    try:
        res1 = minimize(
            lambda x: _soft_min_objective(x, tau1)[0],
            x_curr,
            jac=lambda x: _soft_min_objective(x, tau1)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'gtol': 1e-4, 'maxfun': 5000}
        )
        if time.time() > deadline:
            return x_curr.reshape(N, 3)
        x_curr = res1.x
    except Exception:
        pass
    
    # Phase 2: Medium
    tau2 = 0.02
    try:
        res2 = minimize(
            lambda x: _soft_min_objective(x, tau2)[0],
            x_curr,
            jac=lambda x: _soft_min_objective(x, tau2)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'gtol': 1e-5, 'maxfun': 10000}
        )
        if time.time() > deadline:
            return x_curr.reshape(N, 3)
        x_curr = res2.x
    except Exception:
        pass
    
    # Phase 3: Sharp
    tau3 = 0.005
    try:
        res3 = minimize(
            lambda x: _soft_min_objective(x, tau3)[0],
            x_curr,
            jac=lambda x: _soft_min_objective(x, tau3)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 20000}
        )
        if time.time() > deadline:
            return x_curr.reshape(N, 3)
        x_curr = res3.x
    except Exception:
        pass
    
    # Phase 4: Very Sharp (Polish)
    tau4 = 0.001
    try:
        res4 = minimize(
            lambda x: _soft_min_objective(x, tau4)[0],
            x_curr,
            jac=lambda x: _soft_min_objective(x, tau4)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'gtol': 1e-7, 'maxfun': 30000}
        )
        if time.time() > deadline:
            return x_curr.reshape(N, 3)
        x_curr = res4.x
    except Exception:
        pass
    
    return x_curr.reshape(N, 3)

def _polish(centers, deadline):
    """
    Local search polish: iteratively push apart closest pairs.
    This helps escape local minima of the soft-min.
    """
    centers = centers.copy()
    step_size = 0.01
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Find closest pair
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        # Also check wall constraints
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        # Determine which constraint is tighter
        min_pair_dist = dists[i, j]
        min_wall_dist = wall_dists[min_wall_idx]
        
        if min_pair_dist / 2.0 < min_wall_dist:
            # Push apart i and j
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction /= norm
            centers[i] += direction * step_size
            centers[j] -= direction * step_size
        else:
            # Push away from wall
            i, k = min_wall_idx
            # Wall is at -0.5 or 0.5
            if centers[i, k] > 0:
                centers[i, k] -= step_size
            else:
                centers[i, k] += step_size
        
        # Clip to bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # If no improvement or stuck, reduce step size
        step_size *= 0.95
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_score = 0.0
    best_centers = None
    
    # Initialization strategies
    inits = []
    
    # 1. Grid
    inits.append(_build_grid())
    
    # 2. Halton
    inits.append(_build_halton_init())
    
    # 3. Random
    inits.append(np.random.uniform(-0.45, 0.45, (N, 3)))
    
    # 4. Perturbed Grid
    grid = _build_grid()
    inits.append(grid + np.random.normal(0, 0.02, grid.shape))
    
    # 5. Perturbed Halton
    halton = _build_halton_init()
    inits.append(halton + np.random.normal(0, 0.02, halton.shape))
    
    # Shuffle inits
    np.random.shuffle(inits)
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Optimize
        optimized = _optimize_from_initial(init, deadline)
        
        # Polish
        polished = _polish(optimized, deadline)
        
        # Compute score
        score = _compute_radius(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished
        
        # If we have a very good score, we can stop early?
        # The best known is ~0.11138.
        # If we get above 0.11, it's good.
        if best_score > 0.111:
            # Try one more polish round if time permits
            if time.time() < deadline - 10:
                polished2 = _polish(polished, deadline)
                score2 = _compute_radius(polished2)
                if score2 > best_score:
                    best_score = score2
                    best_centers = polished2
            break
    
    if best_centers is None:
        return _build_grid()
        
    return best_centers

# EVOLVE-BLOCK-END
