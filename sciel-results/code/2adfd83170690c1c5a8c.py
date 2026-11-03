"""
Pack 200 equal circles in a unit square.

Strategy:
1. Multi-start: generate several initial configurations (hex lattices with
   slightly different spacing/phase, plus a jittered variant).
2. For each start, run a force-directed relaxation to remove overlaps.
3. Then run a smooth soft-min objective with L-BFGS-B, increasing the
   sharpness parameter beta to sharpen the min.
4. Use a projected-gradient-descent / local-search phase to escape local
   minima, with adaptive step size and random perturbations.
5. Track the best radius found so far and return the best configuration.

We keep the public function `construct()` returning an (N, 2) array.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
_TIME_LIMIT = 149  # safety margin before the 150s kill

def _get_radius(pts):
    """
    Compute the maximum common radius for a set of centers:
    min of (distance to walls) and (half of minimum pairwise distance).
    """
    if len(pts) == 0:
        return 0.0

    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0

    return min(min_wall, min_pair)

def _get_radius_fast(pts, tree=None):
    """
    Fast radius computation using KDTree for pairwise distances.
    Returns (radius, tree) where tree can be reused.
    """
    if len(pts) == 0:
        return 0.0, None

    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)

    if tree is None:
        tree = cKDTree(pts)
    
    # Query for nearest neighbors
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    return min(min_wall, min_pair), tree

def _get_radius_kd(pts):
    """
    Radius computation using KDTree (rebuilds tree each call).
    """
    r, _ = _get_radius_fast(pts)
    return r

def _soft_min(vals, beta):
    """
    Soft minimum: -1/beta * log(sum(exp(-beta * x_i))).
    Numerically stable via max shift.
    """
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective for minimization: -soft_min(constraints).
    Constraints:
      - 4 wall distances per point
      - pairwise distances / 2
    Returns (value, gradient).
    """
    pts = x.reshape((N, 2))

    # Wall values
    C_wall = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Pair values
    i_idx, j_idx = np.triu_indices(N, k=1)
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    C_pair = pair_dists / 2.0

    # Combine all constraint values
    C = np.concatenate([C_wall, C_pair])

    # Soft min
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights
    weights = exp_shifted / sum_exp

    # Gradient
    grad_pts = np.zeros((N, 2))

    # Wall gradients
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)

    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top

    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)

    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)

    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)

    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)

    pair_weights = weights[4*N:]

    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)

    grad_pts = -grad_pts / beta

    return obj_val, grad_pts.flatten()

def _generate_hex_initial(spacing, phase_shift=0.0, margin=0.001):
    """
    Generate a hexagonal lattice packing with given spacing and phase shift.
    Returns N points inside [-0.5, 0.5]^2.
    """
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + row_h / 2 + phase_shift
    row_idx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + spacing / 2 + shift + (phase_shift if row_idx % 2 == 0 else 0.0)
        while x < 0.5:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]

    if len(pts) < N:
        # Fallback: random
        return np.random.uniform(-0.4, 0.4, (N, 2))

    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts

def _generate_jittered_hex(spacing, jitter=0.002, seed=0):
    """
    Generate hex lattice with small random jitter to break symmetry.
    """
    rng = np.random.RandomState(seed)
    pts = _generate_hex_initial(spacing, phase_shift=0.0)
    pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)
    pts = np.clip(pts, -0.499, 0.499)
    return pts

def _force_directed_phase(pts, time_limit, iterations=None):
    """
    Force-directed repulsion to spread out overlapping circles.
    Uses KDTree for efficient neighbor lookup and boundary repulsion.
    """
    if iterations is None:
        iterations = max(200, int(time_limit * 50))

    t_start = time.time()
    pts = pts.copy()
    best_r, tree = _get_radius_fast(pts)
    best_pts = pts.copy()

    step = 0.004
    lr_decay = 0.9995
    threshold = 0.12  # only consider pairs closer than this

    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break

        forces = np.zeros_like(pts)

        # Build/update KDTree
        tree = cKDTree(pts)
        
        # For each point, find neighbors within threshold
        # Use query_pairs for efficiency
        pairs = tree.query_pairs(threshold)
        
        # Compute repulsion forces from close pairs
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                # Inverse-square repulsion
                force_mag = 1.0 / (dist * dist + 1e-8)
                direction = diff / dist
                forces[i] += force_mag * direction
                forces[j] -= force_mag * direction

        # Boundary repulsion
        margin = 0.04
        forces[:, 0] += np.where(pts[:, 0] < -0.5 + margin, (margin - (pts[:, 0] + 0.5)) * 5.0, 0.0)
        forces[:, 0] -= np.where(pts[:, 0] > 0.5 - margin, (pts[:, 0] - (0.5 - margin)) * 5.0, 0.0)
        forces[:, 1] += np.where(pts[:, 1] < -0.5 + margin, (margin - (pts[:, 1] + 0.5)) * 5.0, 0.0)
        forces[:, 1] -= np.where(pts[:, 1] > 0.5 - margin, (pts[:, 1] - (0.5 - margin)) * 5.0, 0.0)

        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)

        step *= lr_decay

        r, tree = _get_radius_fast(pts, tree=None)  # rebuild tree
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_r, best_pts

def _directed_repulsion_opt(pts, time_limit):
    """
    Structural change: Instead of soft-min gradient, use a pressure/repulsion model.
    At each step, identify the minimum constraint (bottleneck), then apply forces
    to ALL points to increase the minimum distance. This is like a "pressure" model
    where the bottleneck constraint pushes back.
    
    Key insight: Instead of optimizing a smooth approximation of min, we directly
    work with the active constraints and push points apart using a repulsion force
    that is weighted by how close each pair is to the current minimum.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r, tree = _get_radius_fast(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(777)
    
    step_size = 0.002
    no_improve = 0
    patience = 30
    
    for it in range(5000):
        if time.time() - t_start > time_limit:
            break
        
        # Find current radius and bottleneck
        r_curr, tree = _get_radius_fast(pts, tree=None)
        
        # Build tree and find all pairs within 2*r_curr + small_margin
        tree = cKDTree(pts)
        search_radius = 2.0 * r_curr + 0.01
        pairs = tree.query_pairs(search_radius)
        
        # Compute forces: repulsion weighted by proximity to minimum
        forces = np.zeros_like(pts)
        
        for i, j in pairs:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                # Weight: stronger repulsion when closer to minimum
                # Use a smooth weight: w = 1/(1 + (dist - 2*r_curr)^2 / sigma^2)
                excess = dist - 2.0 * r_curr
                sigma = 0.005
                weight = 1.0 / (1.0 + (excess / sigma) ** 2)
                
                # Also add wall pressure
                force_mag = weight * 0.5
                direction = diff / dist
                forces[i] += force_mag * direction
                forces[j] -= force_mag * direction
        
        # Wall forces: push points away from walls proportionally to proximity
        wall_margin = 0.02
        for axis in range(2):
            lower = pts[:, axis] + 0.5
            upper = 0.5 - pts[:, axis]
            # If close to lower wall, push up; if close to upper wall, push down
            forces[:, axis] += np.where(lower < wall_margin, (wall_margin - lower) * 2.0, 0.0)
            forces[:, axis] -= np.where(upper < wall_margin, (wall_margin - upper) * 2.0, 0.0)
        
        # Apply step
        new_pts = pts + step_size * forces
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r, _ = _get_radius_fast(new_pts)
        
        if new_r > best_r + 1e-15:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            step_size = min(step_size * 1.02, 0.005)
        else:
            no_improve += 1
            step_size *= 0.97
            
            if no_improve >= patience:
                # Random kick to escape local minimum
                kick_size = 0.003
                pts = pts + rng.randn(N, 2) * kick_size
                pts = np.clip(pts, -0.5, 0.5)
                no_improve = 0
                step_size = 0.002
    
    return best_r, best_pts

def _random_restart_hill_climb(pts, time_limit, n_restarts=10):
    """
    Structural change: Fast hill-climbing with random restarts.
    Uses KDTree for O(N log N) radius evaluation instead of O(N^2).
    Each restart: take best config, apply small random perturbation, then
    do greedy hill-climbing by trying to improve the worst constraint.
    """
    t_start = time.time()
    best_r, _ = _get_radius_fast(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(999)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for restart in range(n_restarts):
        if time.time() - t_start > time_limit:
            break
        
        # Start from current best with small perturbation
        cur_pts = best_pts + rng.randn(N, 2) * 0.001
        cur_pts = np.clip(cur_pts, -0.5, 0.5)
        cur_r, tree = _get_radius_fast(cur_pts)
        
        # Hill-climbing: iteratively try to improve the worst constraint
        max_inner = 500
        for inner in range(max_inner):
            if time.time() - t_start > time_limit:
                break
            
            # Find the minimum constraint
            wall_dists = np.array([
                np.min(cur_pts[:, 0]) + 0.5,
                0.5 - np.max(cur_pts[:, 0]),
                np.min(cur_pts[:, 1]) + 0.5,
                0.5 - np.max(cur_pts[:, 1])
            ])
            min_wall = np.min(wall_dists)
            
            # Find closest pair using KDTree
            tree = cKDTree(cur_pts)
            dists, indices = tree.query(cur_pts, k=2)
            min_pair_dist = np.min(dists[:, 1])
            min_pair_r = min_pair_dist / 2.0
            
            cur_r = min(min_wall, min_pair_r)
            
            if cur_r > best_r:
                best_r = cur_r
                best_pts = cur_pts.copy()
            
            # Determine which constraint is the bottleneck
            if min_wall <= min_pair_r:
                # Wall is bottleneck - find the point closest to that wall
                wall_type = np.argmin(wall_dists)
                if wall_type == 0:  # left wall
                    point_idx = np.argmin(cur_pts[:, 0])
                    direction = np.array([1.0, 0.0])
                elif wall_type == 1:  # right wall
                    point_idx = np.argmax(cur_pts[:, 0])
                    direction = np.array([-1.0, 0.0])
                elif wall_type == 2:  # bottom wall
                    point_idx = np.argmin(cur_pts[:, 1])
                    direction = np.array([0.0, 1.0])
                else:  # top wall
                    point_idx = np.argmax(cur_pts[:, 1])
                    direction = np.array([0.0, -1.0])
                
                # Try moving this point in the improving direction
                improved = False
                for step in [0.0005, 0.001, 0.002, 0.003]:
                    new_pts = cur_pts.copy()
                    new_pts[point_idx] += step * direction
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_r, _ = _get_radius_fast(new_pts)
                    if new_r > cur_r:
                        cur_pts = new_pts
                        cur_r = new_r
                        improved = True
                        break
            else:
                # Pair is bottleneck - find the closest pair
                # Get the pair with minimum distance
                min_d_idx = np.argmin(dists[:, 1])
                i = min_d_idx
                j = indices[min_d_idx, 1]
                
                diff = cur_pts[i] - cur_pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                dir_i = diff / dist
                dir_j = -diff / dist
                
                # Try separating them
                improved = False
                for step in [0.0005, 0.001, 0.002, 0.003]:
                    new_pts = cur_pts.copy()
                    new_pts[i] += step * dir_i
                    new_pts[j] += step * dir_j
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_r, _ = _get_radius_fast(new_pts)
                    if new_r > cur_r:
                        cur_pts = new_pts
                        cur_r = new_r
                        improved = True
                        break
            
            if not improved:
                # Try a random perturbation of a random point
                if rng.random() < 0.3:
                    idx = rng.randint(N)
                    new_pts = cur_pts.copy()
                    new_pts[idx] += rng.randn(2) * 0.002
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_r, _ = _get_radius_fast(new_pts)
                    if new_r > cur_r:
                        cur_pts = new_pts
                        cur_r = new_r
                    else:
                        break  # Stop this restart attempt
    
    return best_r, best_pts

def _epigraph_objective_and_grad(x, beta=5000):
    """
    Epigraph formulation: minimize -t subject to t <= c_i for all constraints.
    We use a smooth approximation: minimize -soft_min(c_i) + penalty for t > soft_min.
    Actually, we'll directly maximize the minimum constraint value using a smooth approach.
    
    Alternative: Use the fact that at optimum, t = min(c_i). We can use L-BFGS-B
    with a very high beta to get close to the true min, then do a final local search
    on the active constraints.
    """
    pts = x.reshape((N, 2))
    
    # Wall values
    C_wall = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair values
    i_idx, j_idx = np.triu_indices(N, k=1)
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    C_pair = pair_dists / 2.0
    
    # Combine all constraint values
    C = np.concatenate([C_wall, C_pair])
    
    # Very high beta soft min
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights
    weights = exp_shifted / sum_exp
    
    # Gradient
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)
    
    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top
    
    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)
    
    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _final_polish(pts, time_limit):
    """
    Final high-precision polish using multiple stages:
    1. L-BFGS-B with very high beta (10000+)
    2. Active set identification and local refinement
    3. Tiny random perturbations with re-optimization
    """
    t_start = time.time()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Stage 1: L-BFGS-B with very high beta
    for beta in [5000, 10000, 20000]:
        elapsed = time.time() - t_start
        if elapsed > time_limit * 0.6:
            break
        
        x0 = best_pts.flatten()
        
        def fun_and_grad(x):
            val, g = _epigraph_objective_and_grad(x, beta)
            return val, g
        
        try:
            res = minimize(
                fun_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-18, 'maxls': 200}
            )
            pts_curr = res.x.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_pts = pts_curr.copy()
        except Exception:
            pass
    
    # Stage 2: Identify active constraints and do local refinement
    elapsed = time.time() - t_start
    if elapsed < time_limit * 0.8:
        pts_curr = best_pts.copy()
        r_curr = best_r
        
        # Find the minimum constraint value
        C_wall = np.concatenate([
            pts_curr[:, 0] + 0.5,
            0.5 - pts_curr[:, 0],
            pts_curr[:, 1] + 0.5,
            0.5 - pts_curr[:, 1]
        ])
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        D = cdist(pts_curr, pts_curr)
        pair_dists = D[i_idx, j_idx]
        C_pair = pair_dists / 2.0
        
        C_all = np.concatenate([C_wall, C_pair])
        min_val = np.min(C_all)
        
        # Find all constraints within 1e-6 of the minimum (active set)
        active_tol = max(1e-6, min_val * 1e-4)
        active_mask = C_all < (min_val + active_tol)
        
        # For each active constraint, try small perturbations
        n_active = np.sum(active_mask)
        if n_active > 0 and n_active < 50:
            # Try moving points involved in active constraints
            rng = np.random.RandomState(42)
            for _ in range(20):
                if time.time() - t_start > time_limit:
                    break
                
                # Pick a random active constraint and perturb the involved points
                active_indices = np.where(active_mask)[0]
                chosen = rng.choice(active_indices)
                
                # Determine which points are involved
                if chosen < 4 * N:
                    # Wall constraint
                    wall_idx = chosen // N
                    point_idx = chosen % N
                    # Perturb this point
                    delta = rng.randn(2) * 0.0001
                    new_pts = pts_curr.copy()
                    new_pts[point_idx] += delta
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                else:
                    # Pair constraint
                    pair_idx = chosen - 4 * N
                    i = i_idx[pair_idx]
                    j = j_idx[pair_idx]
                    # Perturb both points
                    delta_i = rng.randn(2) * 0.0001
                    delta_j = rng.randn(2) * 0.0001
                    new_pts = pts_curr.copy()
                    new_pts[i] += delta_i
                    new_pts[j] += delta_j
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                
                new_r = _get_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
    
    return best_r, best_pts

def _run_optimization(pts, time_limit, betas=None):
    """
    Full optimization pipeline for a single initial configuration:
    1. Force-directed relaxation (KDTree-based)
    2. L-BFGS-B with increasing beta
    3. Quick directed repulsion polish
    """
    if betas is None:
        betas = [10, 20, 50, 100, 200, 500, 1000, 2000]

    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    best_r, _ = _get_radius_fast(pts)
    best_x = x0.copy()

    # Phase 1: Force-directed (faster with KDTree)
    fd_time = min(3.0, time_limit * 0.25)
    if fd_time > 1.0:
        r_fd, pts_fd = _force_directed_phase(pts, time_limit=fd_time)
        if r_fd > best_r:
            best_r = r_fd
            best_x = pts_fd.flatten()
            x0 = pts_fd.flatten()

    # Phase 2: L-BFGS-B with increasing beta
    for beta in betas:
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            break
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break

        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g

        try:
            if beta >= 500:
                maxiter = 150
            elif beta >= 100:
                maxiter = 100
            else:
                maxiter = 80
            res = minimize(
                fun_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)

            r, _ = _get_radius_fast(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()

        except Exception:
            pass

    # Phase 3: Quick directed repulsion polish
    elapsed = time.time() - t_start
    remaining = time_limit - elapsed
    if remaining > 2.0:
        r_dr, pts_dr = _directed_repulsion_opt(
            best_x.reshape((N, 2)),
            time_limit=remaining * 0.5
        )
        if r_dr > best_r:
            best_r = r_dr
            best_x = pts_dr.flatten()

    return best_r, best_x.reshape((N, 2))

def _active_set_local_search(pts, time_limit, max_iter=500):
    """
    Local search using KDTree for fast radius evaluation.
    Focuses on the active constraints (bottleneck) and tries to improve them.
    """
    t_start = time.time()
    best_r, _ = _get_radius_fast(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(123)
    
    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        pts_curr = best_pts.copy()
        
        # Find bottleneck using KDTree
        tree = cKDTree(pts_curr)
        wall_dists = np.array([
            np.min(pts_curr[:, 0]) + 0.5,
            0.5 - np.max(pts_curr[:, 0]),
            np.min(pts_curr[:, 1]) + 0.5,
            0.5 - np.max(pts_curr[:, 1])
        ])
        min_wall = np.min(wall_dists)
        
        dists, indices = tree.query(pts_curr, k=2)
        min_pair_dist = np.min(dists[:, 1])
        min_pair_r = min_pair_dist / 2.0
        
        r_curr = min(min_wall, min_pair_r)
        
        if r_curr > best_r:
            best_r = r_curr
            best_pts = pts_curr.copy()
        
        # Try to improve the bottleneck
        improved = False
        
        # If wall is bottleneck
        if min_wall <= min_pair_r:
            wall_type = np.argmin(wall_dists)
            if wall_type == 0:
                point_idx = np.argmin(pts_curr[:, 0])
                direction = np.array([1.0, 0.0])
            elif wall_type == 1:
                point_idx = np.argmax(pts_curr[:, 0])
                direction = np.array([-1.0, 0.0])
            elif wall_type == 2:
                point_idx = np.argmin(pts_curr[:, 1])
                direction = np.array([0.0, 1.0])
            else:
                point_idx = np.argmax(pts_curr[:, 1])
                direction = np.array([0.0, -1.0])
            
            for step in [0.0002, 0.0005, 0.001, 0.002]:
                new_pts = pts_curr.copy()
                new_pts[point_idx] += step * direction
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_r, _ = _get_radius_fast(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    improved = True
                    break
        else:
            # Pair is bottleneck
            min_d_idx = np.argmin(dists[:, 1])
            i = min_d_idx
            j = indices[min_d_idx, 1]
            
            diff = pts_curr[i] - pts_curr[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                dir_i = diff / dist
                dir_j = -diff / dist
                
                for step in [0.0002, 0.0005, 0.001, 0.002]:
                    new_pts = pts_curr.copy()
                    new_pts[i] += step * dir_i
                    new_pts[j] += step * dir_j
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_r, _ = _get_radius_fast(new_pts)
                    if new_r > best_r:
                        best_r = new_r
                        best_pts = new_pts.copy()
                        improved = True
                        break
        
        if not improved and it > 50:
            # Random perturbation to escape
            pts_curr = best_pts + rng.randn(N, 2) * 0.001
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            r_curr, _ = _get_radius_fast(pts_curr)
            if r_curr > best_r:
                best_r = r_curr
                best_pts = pts_curr.copy()
    
    return best_r, best_pts

def construct():
    t0 = time.time()
    np.random.seed(42)

    best_r = 0.0
    best_pts = None

    # Generate multiple initial configurations
    inits = []

    # Init 1: Standard hex with spacing ~0.0732
    pts1 = _generate_hex_initial(0.0732, phase_shift=0.0)
    inits.append(pts1)

    # Init 2: Shifted hex
    pts2 = _generate_hex_initial(0.0732, phase_shift=0.01)
    inits.append(pts2)

    # Init 3: Slightly denser hex (will have overlaps, optimizer fixes)
    pts3 = _generate_hex_initial(0.070, phase_shift=0.0)
    inits.append(pts3)

    # Init 4: Jittered hex
    pts4 = _generate_jittered_hex(0.0732, jitter=0.002, seed=0)
    inits.append(pts4)

    # Init 5: Jittered hex with different seed
    pts5 = _generate_jittered_hex(0.0732, jitter=0.003, seed=1)
    inits.append(pts5)

    num_inits = len(inits)
    # Reserve time: 10s for force-directed, 40s for directed repulsion,
    # 30s for random restart hill-climb, 10s for active set search, 5s buffer
    time_per_init = 8.0  # ~8s per init for force-directed + L-BFGS

    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        remaining_for_phase = _TIME_LIMIT - 85 - elapsed  # reserve 85s for later phases
        if remaining_for_phase <= 3:
            break

        t_limit = min(time_per_init, remaining_for_phase)
        r_opt, pts_opt = _run_optimization(init_pts, time_limit=t_limit)

        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt

    # Phase 2: Directed repulsion optimization (structural change)
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 40 - elapsed
        if remaining_time > 10:
            r_dr, pts_dr = _directed_repulsion_opt(
                best_pts,
                time_limit=remaining_time * 0.5
            )
            if r_dr > best_r:
                best_r = r_dr
                best_pts = pts_dr

    # Phase 3: Random restart hill-climbing (structural change - uses KDTree)
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 20 - elapsed
        if remaining_time > 10:
            r_hc, pts_hc = _random_restart_hill_climb(
                best_pts,
                time_limit=remaining_time * 0.5,
                n_restarts=15
            )
            if r_hc > best_r:
                best_r = r_hc
                best_pts = pts_hc

    # Phase 4: Active set local search with KDTree
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 5 - elapsed
        if remaining_time > 5:
            r_active, pts_active = _active_set_local_search(
                best_pts,
                time_limit=remaining_time * 0.6
            )
            if r_active > best_r:
                best_r = r_active
                best_pts = pts_active

    # Phase 5: Final high-precision polish with L-BFGS
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 3 - elapsed
        if remaining_time > 5:
            betas_polish = [5000, 10000]
            r_polish, pts_polish = _run_optimization(
                best_pts,
                time_limit=remaining_time * 0.7,
                betas=betas_polish
            )
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish

    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))

    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
