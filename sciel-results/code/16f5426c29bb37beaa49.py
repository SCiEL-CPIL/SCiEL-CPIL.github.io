"""
Sphere packing in a unit cube: 50 equal spheres.

Strategy:
1. Multiple diverse initializations (body-centred cubic, face-centred cubic, jittered grid, random).
2. Direct optimization of the exact objective (minimize -radius) using L-BFGS-B with a smooth
   approximation via soft-min, then exact objective with Nelder-Mead.
3. Local neighborhood polish: identify bottleneck pairs and walls, then perform targeted moves.
4. Simulated annealing style perturbations to escape local minima.
"""
import numpy as np
import time
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
import sys

N = 50

def _score(centers):
    """
    Compute the maximum common radius for given centers.
    Uses vectorized operations for speed.
    """
    centers = np.asarray(centers, dtype=np.float64)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _get_bottlenecks(centers, r_current):
    """
    Identify which constraints are active (bottleneck) at the current radius.
    Returns:
        bottleneck_pairs: list of (i, j) pairs that are at minimum distance
        bottleneck_walls: list of (i, axis) where wall constraint is active
    """
    centers = np.asarray(centers, dtype=np.float64)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    bottleneck_pairs = []
    bottleneck_walls = []
    
    tol = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            if abs(dists[i, j] / 2.0 - r_current) < tol:
                bottleneck_pairs.append((i, j))
    
    for i in range(N):
        if abs(wall_dists[i] - r_current) < tol:
            # Find which axis is limiting
            max_abs = np.max(np.abs(centers[i]))
            for k in range(3):
                if abs(abs(centers[i][k]) - max_abs) < tol:
                    bottleneck_walls.append((i, k))
    
    return bottleneck_pairs, bottleneck_walls

def _local_polish(centers, time_limit=10.0):
    """
    Local polish: try small perturbations on bottleneck points to increase the minimum radius.
    Uses a coordinate descent approach on the bottleneck constraints.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    for iteration in range(200):
        if time.time() > start_time + time_limit:
            break
        
        cur = best_centers.copy()
        cur_score = _score(cur)
        
        # Get bottleneck information
        bottleneck_pairs, bottleneck_walls = _get_bottlenecks(cur, cur_score)
        
        # Try moving bottleneck points slightly to increase clearance
        improved = False
        for _ in range(10):
            if time.time() > start_time + time_limit:
                break
            
            new_cur = cur.copy()
            
            # For each bottleneck pair, try to separate them
            for (i, j) in bottleneck_pairs[:5]:  # Limit to top 5 to save time
                d = cur[i] - cur[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                direction = d / dist
                # Move i in direction, j in opposite direction
                step = 0.001
                new_cur[i] += direction * step
                new_cur[j] -= direction * step
            
            # For each bottleneck wall, move the point away from the wall
            for (i, axis) in bottleneck_walls[:5]:
                # Move away from the wall (toward center)
                sign = -np.sign(new_cur[i][axis]) if new_cur[i][axis] != 0 else 1
                new_cur[i][axis] += sign * 0.001
            
            # Clip to cube
            new_cur = np.clip(new_cur, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_score = _score(new_cur)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_centers = new_cur.copy()
                improved = True
        
        if not improved:
            break
    
    return best_centers, best_score

def _soft_min_objective(x_flat, beta):
    """
    Objective function: minimize -soft_min(radius).
    x_flat: flattened centers (N*3)
    beta: softness parameter
    """
    centers = x_flat.reshape(N, 3)
    
    # Wall clearances
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c.max(axis=1)
    
    # Pair clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = 0.5 * dists[i_idx, j_idx]
    
    z = np.concatenate([wall_dists, pair_dists])
    
    # Soft min
    m = np.min(z)
    w = np.exp(-beta * (z - m))
    S_prime = np.sum(w)
    soft_min = m + np.log(S_prime) / beta
    
    return -soft_min

def _soft_min_gradient(x_flat, beta):
    """
    Gradient of the soft-min objective.
    """
    centers = x_flat.reshape(N, 3)
    
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c.max(axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = 0.5 * dists[i_idx, j_idx]
    
    z = np.concatenate([wall_dists, pair_dists])
    
    m = np.min(z)
    w = np.exp(-beta * (z - m))
    S_prime = np.sum(w)
    grad_z = w / S_prime
    
    grad_centers = np.zeros_like(centers)
    
    # Wall terms
    max_abs_val = abs_c.max(axis=1)
    is_max = (abs_c == max_abs_val[:, np.newaxis])
    count_max = is_max.sum(axis=1, keepdims=True)
    count_max[count_max == 0] = 1.0
    
    grad_wall_contrib = grad_z[:N][:, np.newaxis] * (-np.sign(centers) * is_max / count_max)
    grad_centers += grad_wall_contrib
    
    # Pair terms
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        d = dists[i, j]
        if d < 1e-10:
            continue
        g = grad_z[N + k]
        vec = 0.5 * (centers[i] - centers[j]) / d
        grad_centers[i] += g * vec
        grad_centers[j] -= g * vec
    
    return grad_centers.flatten()

def _smooth_optimize(centers, time_limit=15.0):
    """
    Optimize using L-BFGS-B with soft-min approximation, annealing beta.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + time_limit - 2:
            break
        
        def objective(x):
            return _soft_min_objective(x, beta)
        
        def gradient(x):
            return _soft_min_gradient(x, beta)
        
        try:
            res = minimize(
                objective,
                centers.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            new_centers = res.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            new_score = _score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                centers = new_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def _nelder_mead_polish(centers, time_limit=5.0):
    """
    Final polish using Nelder-Mead on the exact objective.
    """
    start_time = time.time()
    
    def exact_obj(x):
        c = x.reshape(N, 3)
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        return -_score(c)
    
    try:
        res = minimize(
            exact_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        new_centers = res.x.reshape(N, 3)
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        if time.time() > start_time + time_limit:
            return centers, _score(centers)
        
        return new_centers, _score(new_centers)
    except Exception:
        return centers, _score(centers)

def _init_bcc(seed=42):
    """
    Body-Centered Cubic initialization.
    Place spheres at corners and center of sub-cubes.
    """
    rng = np.random.RandomState(seed)
    # Use a 3x3x3 grid of BCC units
    # Each BCC unit has 2 points: corner and body center
    # Scale to fit in [-0.5, 0.5]^3
    
    points = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                # Corner point
                x = (ix - 1) * 0.3333
                y = (iy - 1) * 0.3333
                z = (iz - 1) * 0.3333
                points.append([x, y, z])
                # Body center
                x_c = (ix - 1) * 0.3333 + 0.1667
                y_c = (iy - 1) * 0.3333 + 0.1667
                z_c = (iz - 1) * 0.3333 + 0.1667
                points.append([x_c, y_c, z_c])
    
    points = np.array(points)
    # Jitter
    jitter = rng.uniform(-0.02, 0.02, size=points.shape)
    points += jitter
    points = np.clip(points, -0.49, 0.49)
    
    # Select N points
    if len(points) >= N:
        perm = rng.permutation(len(points))
        return points[perm[:N]].copy()
    else:
        # Fill with random points
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        return np.vstack([points, extra])

def _init_fcc(seed=42):
    """
    Face-Centered Cubic initialization.
    """
    rng = np.random.RandomState(seed)
    points = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                # Corner
                x = (ix - 1) * 0.3333
                y = (iy - 1) * 0.3333
                z = (iz - 1) * 0.3333
                points.append([x, y, z])
                # Face centers
                if ix < 2:
                    points.append([x + 0.1667, y, z])
                if iy < 2:
                    points.append([x, y + 0.1667, z])
                if iz < 2:
                    points.append([x, y, z + 0.1667])
    
    points = np.array(points)
    jitter = rng.uniform(-0.02, 0.02, size=points.shape)
    points += jitter
    points = np.clip(points, -0.49, 0.49)
    
    if len(points) >= N:
        perm = rng.permutation(len(points))
        return points[perm[:N]].copy()
    else:
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        return np.vstack([points, extra])

def _init_jittered_grid(seed=42):
    """Jittered grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _init_random(seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centers

def _perturb(centers, scale=0.01, seed=None):
    """Apply random perturbation to escape local minima."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    perturbed = centers + rng.uniform(-scale, scale, size=centers.shape)
    return np.clip(perturbed, -0.5 + 1e-9, 0.5 - 1e-9)

def construct():
    t_start = time.time()
    deadline = t_start + 145  # 5s safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Multiple initializations
    inits = [
        _init_bcc(seed=42),
        _init_bcc(seed=123),
        _init_fcc(seed=42),
        _init_fcc(seed=123),
        _init_jittered_grid(seed=42),
        _init_jittered_grid(seed=123),
        _init_random(seed=101),
        _init_random(seed=202),
    ]
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
        
        # Smooth optimization
        if time.time() > deadline - 25:
            continue
        
        time_left_smooth = min(15.0, deadline - 20 - time.time())
        if time_left_smooth > 5:
            smooth_res, smooth_score = _smooth_optimize(cur, time_limit=time_left_smooth)
            if smooth_score > best_score:
                best_score = smooth_score
                best_centers = smooth_res.copy()
                cur = smooth_res.copy()
        
        # Local polish
        if time.time() > deadline - 15:
            continue
        
        time_left_polish = min(10.0, deadline - 10 - time.time())
        if time_left_polish > 3:
            polish_res, polish_score = _local_polish(cur, time_limit=time_left_polish)
            if polish_score > best_score:
                best_score = polish_score
                best_centers = polish_res.copy()
                cur = polish_res.copy()
        
        # Nelder-Mead polish
        if time.time() > deadline - 5:
            continue
        
        time_left_nm = min(5.0, deadline - 2 - time.time())
        if time_left_nm > 2:
            nm_res, nm_score = _nelder_mead_polish(cur, time_limit=time_left_nm)
            if nm_score > best_score:
                best_score = nm_score
                best_centers = nm_res.copy()
        
        # Perturbation and re-optimize
        if time.time() > deadline - 10:
            continue
        
        for perturb_seed in [1, 2, 3]:
            if time.time() > deadline - 8:
                break
            
            perturbed = _perturb(best_centers, scale=0.02, seed=perturb_seed)
            pert_score = _score(perturbed)
            
            if pert_score > best_score - 0.001:  # Allow slight decrease for exploration
                time_left_smooth = min(5.0, deadline - 5 - time.time())
                if time_left_smooth > 2:
                    smooth_res, smooth_score = _smooth_optimize(perturbed, time_limit=time_left_smooth)
                    if smooth_score > best_score:
                        best_score = smooth_score
                        best_centers = smooth_res.copy()
    
    # Final safety check
    if best_centers is None:
        best_centers = _init_jittered_grid()
    
    # Ensure final result is valid (inside cube)
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

if __name__ == "__main__":
    # Quick self-test
    centers = construct()
    r = _score(centers)
    print(f"Score: {r}")
