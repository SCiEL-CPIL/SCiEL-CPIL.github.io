"""Combined solver: FCC initialization + Voronoi-Lloyd equalization + smooth max-min polish + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _generate_initial_positions():
    """Generate a good initial configuration using FCC lattice points closest to origin."""
    points = []
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])

    points = np.array(points)

    # Center at origin
    points -= np.mean(points, axis=0)

    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)

    # Take N closest
    pts = points[idx[:N]].copy()

    # Scale to fit inside cube with margin
    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale

    return pts


def _generate_bcc_initial():
    """Generate initial positions from BCC lattice."""
    points = []
    max_idx = 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k + 0.5])

    points = np.array(points)
    points -= np.mean(points, axis=0)

    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    pts = points[idx[:N]].copy()

    max_coord = np.max(np.abs(pts))
    if max_coord > 0:
        scale = 0.42 / max_coord
        pts *= scale

    return pts


def _generate_shell_core_initial():
    """Generate initial positions with a core-shell structure."""
    rng = np.random.default_rng(99)
    pts = np.zeros((N, 3))

    # Core: 27 points in a 3x3x3 grid
    core_n = 27
    core_grid = np.array([[i - 1, j - 1, k - 1] for i in range(3)
                          for j in range(3)
                          for k in range(3)], dtype=float)
    core_grid *= 0.18
    pts[:core_n] = core_grid

    # Shell: remaining 23 points on a spherical shell
    shell_n = N - core_n
    shell_pts = rng.normal(0, 1, size=(shell_n, 3))
    norms = np.linalg.norm(shell_pts, axis=1, keepdims=True)
    shell_pts /= norms
    shell_pts *= 0.38
    pts[core_n:] = shell_pts

    # Small perturbation to break symmetry
    pts += rng.normal(0, 0.005, size=pts.shape)
    pts = np.clip(pts, -0.45, 0.45)

    return pts


def _generate_grid_initial():
    """Generate initial positions on a perturbed grid for diversity."""
    # Use a 4x4x4 grid (64 points), take 50
    n_per_side = 4
    total = n_per_side ** 3
    grid = np.array([[i, j, k] for i in range(n_per_side)
                     for j in range(n_per_side)
                     for k in range(n_per_side)], dtype=float)
    grid = (grid / (n_per_side - 1) - 0.5) * 0.85  # map to [-0.425, 0.425]
    
    rng = np.random.default_rng(123)
    # Add small random perturbation
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)
    
    # Take first N points
    return grid[:N].copy()


def _generate_random_initial(seed=42):
    """Generate random initial positions."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.4, 0.4, size=(N, 3))
    return pts


def _lloyd_optimize(c, deadline):
    """Voronoi-Lloyd equalization: move each center toward its Voronoi cell centroid."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)

    for damp in (0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.03):
        if time.time() > deadline:
            break
        steps = max(30, int(200 * damp))
        for _ in range(steps):
            if time.time() > deadline:
                break
            try:
                tri = Delaunay(c)
            except Exception:
                # Fallback: simple repulsion
                d = _pair_dist_matrix(c)
                inv = 1.0 / np.maximum(d, 1e-6) ** 2
                diff = c[:, None, :] - c[None, :, :]
                F = np.sum(diff * inv[:, :, None], axis=1)
                c = np.clip(c + 0.005 * F, -0.49, 0.49)
                continue

            simplices = tri.simplices
            cell_centroids = np.zeros((N, 3))
            cell_counts = np.zeros(N)

            for si in simplices:
                v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
                # Circumcenter
                A = np.array([v1 - v0, v2 - v0, v3 - v0])
                b = np.array([
                    np.dot(v1, v1) - np.dot(v0, v0),
                    np.dot(v2, v2) - np.dot(v0, v0),
                    np.dot(v3, v3) - np.dot(v0, v0)
                ])
                try:
                    cc = np.linalg.solve(2.0 * A, b)
                except np.linalg.LinAlgError:
                    cc = np.mean([v0, v1, v2, v3])

                for vi in si:
                    cell_centroids[vi] += cc
                    cell_counts[vi] += 1

            valid = cell_counts > 0
            if np.any(valid):
                targets = cell_centroids[valid] / cell_counts[valid, None]
                c[valid] = c[valid] + damp * (targets - c[valid])

            c = np.clip(c, -0.49, 0.49)

        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()

    return best, best_r


def _softmin_radius(c, t):
    """Smooth approximation of the true radius via log-sum-exp.
    Uses upper-triangular pairs only to avoid double-counting."""
    d = _pair_dist_matrix(c)
    # Use only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair = d[iu] / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair, wall.ravel()])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) / t))
    return m - t * np.log(s)


def _smooth_obj(x, t):
    """Objective to MINIMIZE: negative smooth radius plus boundary penalty."""
    c = x.reshape(N, 3)
    r = _softmin_radius(c, t)
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    return -r + 1e3 * pen


def _smooth_polish(c, deadline, fine=False):
    """Polish with smooth softmin max-min via L-BFGS-B, annealing temperature."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    x0 = c.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)

    if fine:
        temps = (0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001)
        maxiter = 4000
    else:
        temps = (0.04, 0.015, 0.006, 0.002, 0.0008, 0.0003, 0.0001)
        maxiter = 2000

    for t in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _smooth_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(t,),
                options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-13,
                         'maxls': 80},
            )
            c = np.clip(res.x.reshape(N, 3), -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
            x0 = c.flatten()
        except Exception:
            pass
    return best, best_r


def _repulsion_anneal(c, deadline, steps=2000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)

    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)

        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))

        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)

    return c


def _active_set_polish(c, deadline, max_iter=500):
    """
    Active set method to maximize the minimum distance.
    Identifies binding constraints (pairs and walls) and uses a QP-like step
    to increase the minimum distance among them.
    """
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        d = _pair_dist_matrix(c)
        min_pair_dist = np.min(d)
        wall = 0.5 - np.abs(c)
        min_wall_val = np.min(wall)
        
        current_r = min(min_pair_dist / 2.0, min_wall_val)
        
        # Identify binding constraints (within 1e-5 relative tolerance)
        tol = 1e-5
        binding_pairs = np.argwhere(d < min_pair_dist * (1.0 + tol))
        binding_pairs = [(i, j) for i, j in binding_pairs if i < j]
        
        binding_walls = np.argwhere(wall < min_wall_val * (1.0 + tol))
        
        if not binding_pairs and not binding_walls:
            break
            
        # We want to move points to increase the minimum distance.
        # This is equivalent to finding a direction v that maximizes the minimum dot product
        # with the gradients of the active constraints.
        
        # Gradients for pairs: (c_i - c_j) / |c_i - c_j|
        # Gradients for walls: -sign(c_ik) e_k
        
        # Simple heuristic active set step:
        # For each binding pair, push apart.
        # For each binding wall, push inward.
        
        step = np.zeros_like(c)
        
        # Pair forces
        for i, j in binding_pairs:
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10: continue
            direction = diff / dist
            step[i] += direction
            step[j] -= direction
            
        # Wall forces
        for idx, axis in binding_walls:
            sgn = np.sign(c[idx, axis])
            if sgn == 0: sgn = 1.0
            # Push away from wall (inward)
            step[idx, axis] -= sgn
            
        # Normalize step to have magnitude ~ current_r * 0.1
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-10:
            target_step = current_r * 0.05
            step = step / step_norm * target_step
            
        c_new = c + step
        c_new = np.clip(c_new, -0.499, 0.499)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new
            c = c_new
        else:
            # If we didn't improve, try a smaller step or break
            # Simple backtracking
            found = False
            for shrink in (0.5, 0.1, 0.01):
                c_trial = c + step * shrink
                c_trial = np.clip(c_trial, -0.499, 0.499)
                r_trial = _compute_radius(c_trial)
                if r_trial > best_r:
                    best_r = r_trial
                    best_c = c_trial
                    c = c_trial
                    found = True
                    break
            if not found:
                # If no improvement with smaller steps, we might be stuck or at a local optimum
                # Break to avoid wasting time
                break

    return best_c, best_r

def _local_search(c, deadline, iters=3000):
    """Local search: perturb binding constraints with adaptive step sizes."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    # Track consecutive rejections to adapt step size
    rejections = 0
    max_rejections = 50
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        # Compute distances and walls
        d = _pair_dist_matrix(c)
        min_pair_dist = np.min(d)
        
        wall = 0.5 - np.abs(c)
        min_wall_val = np.min(wall)
        
        # Determine binding constraints (within 2% of minimum)
        threshold = 0.02
        close_pairs = np.argwhere(d < min_pair_dist * (1.0 + threshold))
        close_pairs = [(i, j) for i, j in close_pairs if i < j]
        
        close_walls = np.argwhere(wall < min_wall_val * (1.0 + threshold))
        
        # Adaptive step size: smaller when many rejections
        base_step = 0.002 * (best_r / 0.135)
        base_step *= (1.0 - 0.01 * min(rejections, 50))
        base_step = np.clip(base_step, 0.0005, 0.004)
        
        # Choose which constraint to perturb
        n_close_pairs = len(close_pairs)
        n_close_walls = len(close_walls)
        
        if n_close_pairs > 0 and (n_close_walls == 0 or rng.random() < 0.7):
            # Pick a random close pair
            i, j = close_pairs[rng.integers(0, n_close_pairs)]
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
                # Push apart with some randomness
                step = base_step * rng.uniform(0.5, 1.5)
                c[i] += step * direction
                c[j] -= step * direction
        elif n_close_walls > 0:
            # Pick a random close wall
            idx, axis = close_walls[rng.integers(0, n_close_walls)]
            # Push away from wall
            step = base_step * rng.uniform(0.5, 1.5)
            if c[idx, axis] > 0:
                c[idx, axis] -= step
            else:
                c[idx, axis] += step
        else:
            # No tight constraints: try random nudge to escape
            i = rng.integers(0, N)
            c[i] += rng.normal(0, base_step, size=3)

        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            rejections = 0
        else:
            rejections += 1
            # If stuck, occasionally try a larger perturbation
            if rejections > max_rejections and rng.random() < 0.1:
                i = rng.integers(0, N)
                c[i] += rng.normal(0, base_step * 3, size=3)
                c = np.clip(c, -0.499, 0.499)
                r = _compute_radius(c)
                if r > best_r:
                    best_r = r
                    best_c = c.copy()
                    rejections = 0

    return best_c, best_r


def _metropolis_mc(c, deadline, steps=5000, temp0=0.008, temp_min=0.0005):
    """Metropolis Monte Carlo with adaptive step size and better annealing."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(77)
    r_cur = best_r
    
    # Track acceptance rate to adapt step size
    accept_count = 0
    total_count = 0
    
    for it in range(steps):
        if time.time() > deadline:
            break

        # Anneal temperature with a slower decay
        frac = it / max(steps - 1, 1)
        temp = temp0 * (temp_min / temp0) ** (0.7 * frac)

        # Adapt step size based on recent acceptance rate
        if total_count > 100:
            accept_rate = accept_count / total_count
            # Target ~50% acceptance rate
            if accept_rate > 0.6:
                step_scale = 0.006 * (best_r / 0.135) * 1.3
            elif accept_rate < 0.3:
                step_scale = 0.006 * (best_r / 0.135) * 0.7
            else:
                step_scale = 0.006 * (best_r / 0.135)
        else:
            step_scale = 0.006 * (best_r / 0.135)
            
        # Occasionally try multi-particle moves
        if rng.random() < 0.15:
            # Move 2-3 particles together
            n_move = rng.integers(2, 4)
            indices = rng.choice(N, size=n_move, replace=False)
            c_new = c.copy()
            for idx in indices:
                delta = rng.normal(0, step_scale * 0.7, size=3)
                c_new[idx] += delta
            c_new = np.clip(c_new, -0.49, 0.49)
        else:
            # Single particle move
            i = rng.integers(0, N)
            delta = rng.normal(0, step_scale, size=3)
            c_new = c.copy()
            c_new[i] += delta
            c_new = np.clip(c_new, -0.49, 0.49)

        r_new = _compute_radius(c_new)
        total_count += 1

        # Accept if better or with probability exp((r_new - r_cur) / temp)
        if r_new >= r_cur:
            c = c_new
            r_cur = r_new
            accept_count += 1
        else:
            prob = np.exp((r_new - r_cur) / max(temp, 1e-10))
            if rng.random() < prob:
                c = c_new
                r_cur = r_new
                accept_count += 1

        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
        
        # Reset acceptance counter periodically
        if total_count % 200 == 0:
            accept_count = 0
            total_count = 0

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # 1. Initialize with FCC-like cluster
    best_pts = _generate_initial_positions()
    best_r = _compute_radius(best_pts)

    # Also try grid initialization
    grid_pts = _generate_grid_initial()
    grid_r = _compute_radius(grid_pts)
    if grid_r > best_r:
        best_r = grid_r
        best_pts = grid_pts

    # Also try BCC initialization
    bcc_pts = _generate_bcc_initial()
    bcc_r = _compute_radius(bcc_pts)
    if bcc_r > best_r:
        best_r = bcc_r
        best_pts = bcc_pts

    # Also try shell-core initialization
    shell_pts = _generate_shell_core_initial()
    shell_r = _compute_radius(shell_pts)
    if shell_r > best_r:
        best_r = shell_r
        best_pts = shell_pts

    # 2. Voronoi-Lloyd equalization
    if time.time() < deadline:
        pts, r = _lloyd_optimize(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 3. Repulsion anneal
    if time.time() < deadline:
        pts = _repulsion_anneal(best_pts, deadline, steps=1500)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 4. Smooth max-min polish
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 5. Metropolis MC to escape local optima
    if time.time() < deadline:
        pts, r = _metropolis_mc(best_pts, deadline, steps=5000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 6. Local search on binding constraints
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=4000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 7. Fine smooth polish if time remains
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline, fine=True)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 7.5 Active set polish to snap to exact constraints
    if time.time() < deadline:
        pts, r = _active_set_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 8. Second Metropolis pass with different seed
    if time.time() < deadline:
        pts, r = _metropolis_mc(best_pts, deadline, steps=3000, temp0=0.005, temp_min=0.0003)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 9. Final local search with tighter steps
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=3000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 10. Final fine polish
    if time.time() < deadline:
        pts, r = _smooth_polish(best_pts, deadline, fine=True)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 11. Ultra-fine local search
    if time.time() < deadline:
        pts, r = _local_search(best_pts, deadline, iters=2000)
        if r > best_r:
            best_r = r
            best_pts = pts

    # 12. Final active set polish for maximum precision
    if time.time() < deadline:
        pts, r = _active_set_polish(best_pts, deadline)
        if r > best_r:
            best_r = r
            best_pts = pts

    # Ensure final result is valid
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r

    return best_pts
# EVOLVE-BLOCK-END
