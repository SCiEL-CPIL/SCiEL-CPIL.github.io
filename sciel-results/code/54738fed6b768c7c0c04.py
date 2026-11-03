"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _get_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return max(r, 0.0)

def _soft_min(values, beta):
    """Soft minimum using log-sum-exp trick."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if beta > 1e6:
        return m
    exp_vals = np.exp(-beta * (values - m))
    return m - np.log(np.sum(exp_vals)) / beta

def _objective_and_grad(centers, beta):
    """
    Objective: maximize min over all constraints.
    Constraints: for each pair (i,j), dist_ij >= 2r; for each i, wall_dist_i >= r.
    We minimize -r where r = min(all_constraints/2 or wall_dist).
    
    Using epigraph: maximize t subject to all constraints >= t.
    Soft surrogate: minimize -soft_min(constraints) / 2 for pair distances,
    and wall distances directly.
    
    Actually, let's define:
    - For pairs: constraint_ij = dist_ij (we need dist_ij >= 2t, so effective is dist_ij/2)
    - For walls: constraint_i = wall_dist_i (we need wall_dist_i >= t)
    
    r = min(min_pair/2, min_wall)
    
    We want to maximize r. Using soft-min:
    f = soft_min([all pair_dists/2, all wall_dists], beta)
    grad = -df/dcenters
    """
    n = centers.shape[0]
    
    # Wall distances
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    
    # Pair distances using KDTree
    tree = cKDTree(centers)
    # Get all pairs within a reasonable distance
    # For efficiency, use k-nearest neighbors
    k = min(10, n)
    dists, idxs = tree.query(centers, k=k)
    
    # Collect all relevant distances (pairs and walls)
    # For the soft-min, we need all pair distances / 2 and wall distances
    # But we can approximate by using only nearest neighbors for the gradient
    
    # Effective constraints for each point:
    # - nearest pair distance / 2
    # - wall distance
    pair_dists_half = dists[:, 1:] / 2.0  # shape (n, k-1)
    wall_d = wall_dists  # shape (n,)
    
    # Combine all constraints
    # For gradient computation, we need d(constraint)/d(centers)
    
    # Let's compute the soft-min over all constraints
    all_constraints = np.concatenate([pair_dists_half.ravel(), wall_d])
    
    # Soft min
    m = np.min(all_constraints)
    if beta > 1e6:
        soft_min_val = m
    else:
        exp_vals = np.exp(-beta * (all_constraints - m))
        soft_min_val = m - np.log(np.sum(exp_vals)) / beta
    
    # Gradient of soft_min with respect to each constraint
    if beta > 1e6:
        # Hard min: gradient is -1 at the minimum, 0 elsewhere
        min_mask = (all_constraints == m)
        weights = np.zeros_like(all_constraints)
        weights[min_mask] = 1.0 / np.sum(min_mask)
    else:
        weights = exp_vals / np.sum(exp_vals)
    
    # Now compute gradient with respect to centers
    grad = np.zeros_like(centers)
    
    # Wall gradient
    # wall_dist_i = min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5)
    # d(wall_dist_i)/dx_i = -1 if 0.5 - x_i is min, +1 if x_i + 0.5 is min
    for i in range(n):
        w = weights[n * (k - 1) + i]  # weight for wall constraint i
        if w < 1e-15:
            continue
        # Determine which wall is closest
        dx_pos = 0.5 - centers[i, 0]
        dx_neg = centers[i, 0] + 0.5
        dy_pos = 0.5 - centers[i, 1]
        dy_neg = centers[i, 1] + 0.5
        min_val = min(dx_pos, dx_neg, dy_pos, dy_neg)
        if abs(dx_pos - min_val) < 1e-12:
            grad[i, 0] += w * (-1.0)
        elif abs(dx_neg - min_val) < 1e-12:
            grad[i, 0] += w * (1.0)
        if abs(dy_pos - min_val) < 1e-12:
            grad[i, 1] += w * (-1.0)
        elif abs(dy_neg - min_val) < 1e-12:
            grad[i, 1] += w * (1.0)
    
    # Pair distance gradient
    # For pair (i, j): constraint = |c_i - c_j| / 2
    # d(constraint)/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(constraint)/dc_j = (c_j - c_i) / (2 * |c_i - c_j|)
    for i in range(n):
        for j_idx in range(1, k):
            j = idxs[i, j_idx]
            w = weights[i * (k - 1) + (j_idx - 1)]
            if w < 1e-15:
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-15:
                continue
            unit = diff / dist
            # d(|c_i - c_j|/2)/dc_i = unit / 2
            grad[i] += w * unit / 2.0
            # d(|c_i - c_j|/2)/dc_j = -unit / 2
            grad[j] -= w * unit / 2.0
    
    # We want to maximize soft_min, so gradient of objective is -grad
    # (since we'll do gradient ascent, or equivalently minimize -objective)
    return -soft_min_val, -grad

def _generate_hex_initial(n_rows=None):
    """Generate hexagonal lattice initial configuration."""
    if n_rows is None:
        n_rows = int(np.sqrt(N * 2 / np.sqrt(3)))
    
    best_centers = None
    best_r = 0.0
    
    for nr in range(max(8, n_rows - 4), min(30, n_rows + 5)):
        nc = int(np.ceil(N / nr))
        
        r1 = 1.0 / (2.0 * nc)
        r2 = 1.0 / ((nr - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        centers = []
        for i in range(nr):
            for j in range(nc):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _get_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers

def _generate_jittered_hex(centers, jitter_scale=0.005, seed=42):
    """Add small random jitter to break symmetry."""
    rng = np.random.RandomState(seed)
    jittered = centers.copy()
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=jittered.shape)
    jittered += jitter
    jittered[:, 0] = np.clip(jittered[:, 0], -0.5, 0.5)
    jittered[:, 1] = np.clip(jittered[:, 1], -0.5, 0.5)
    return jittered

def _force_directed_phase(centers, max_iter=500, time_budget=20.0):
    """Force-directed relaxation to spread circles apart."""
    centers = centers.copy()
    start_time = time.time()
    
    for it in range(max_iter):
        if _time_left() < 5.0 or (time.time() - start_time) > time_budget:
            break
        
        tree = cKDTree(centers)
        n = centers.shape[0]
        
        # Compute forces: repulsion between close pairs, attraction to center
        forces = np.zeros_like(centers)
        
        # Wall forces: push away from walls
        for i in range(n):
            for dim in range(2):
                if centers[i, dim] < -0.4:
                    forces[i, dim] += 1.0 / (0.1 - (centers[i, dim] + 0.5)) ** 2
                elif centers[i, dim] > 0.4:
                    forces[i, dim] -= 1.0 / (0.1 - (0.5 - centers[i, dim])) ** 2
        
        # Pair repulsion
        dists, idxs = tree.query(centers, k=6)
        for i in range(n):
            for j_idx in range(1, 6):
                j = idxs[i, j_idx]
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Repulsive force ~ 1/dist^2
                force_mag = 0.01 / (dist ** 2)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Apply forces with damping
        lr = 0.001 * (1.0 - it / max_iter)
        centers += lr * forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    return centers

def _projected_gradient_descent(centers, max_iter=2000, beta=100.0, time_budget=30.0):
    """Projected gradient ascent on the soft-min objective."""
    centers = centers.copy()
    start_time = time.time()
    best_r = _get_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        if _time_left() < 5.0 or (time.time() - start_time) > time_budget:
            break
        
        obj, grad = _objective_and_grad(centers, beta)
        
        # Adaptive learning rate
        lr = 0.005 * (1.0 + 0.5 * np.cos(it / 100.0))
        
        # Gradient ascent
        trial = centers + lr * grad
        trial[:, 0] = np.clip(trial[:, 0], -0.5, 0.5)
        trial[:, 1] = np.clip(trial[:, 1], -0.5, 0.5)
        
        r_trial = _get_radius(trial)
        if r_trial > best_r + 1e-14:
            best_r = r_trial
            best_centers = trial.copy()
            centers = trial.copy()
        
        # Increase beta over time for sharper min
        if it % 100 == 0 and beta < 1000:
            beta *= 1.5
    
    return best_centers, best_r

def _epigraph_objective_and_grad(centers, t, beta):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    Smooth surrogate: minimize sum of soft-violations + penalty on t.
    
    f(centers, t) = -t + (1/beta) * sum(log(1 + exp(beta * max(0, t - constraint))))
    """
    n = centers.shape[0]
    
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    
    tree = cKDTree(centers)
    k = min(10, n)
    dists, idxs = tree.query(centers, k=k)
    pair_dists_half = dists[:, 1:] / 2.0
    
    all_constraints = np.concatenate([pair_dists_half.ravel(), wall_dists])
    
    # Violations: max(0, t - constraint)
    violations = np.maximum(0.0, t - all_constraints)
    
    # Soft sum of violations using log-sum-exp
    if np.max(violations) < 1e-15:
        soft_viol = 0.0
        viol_grad = np.zeros_like(all_constraints)
    else:
        m = np.max(violations)
        exp_vals = np.exp(beta * (violations - m))
        soft_viol = (m + np.log(np.sum(exp_vals))) / beta
        viol_grad = exp_vals / np.sum(exp_vals)
    
    obj = -t + soft_viol
    
    # Gradient
    grad_centers = np.zeros_like(centers)
    grad_t = -1.0 + np.sum(viol_grad)
    
    # Only for violated constraints
    violated_mask = violations > 1e-15
    if np.any(violated_mask):
        # Wall gradients
        for i in range(n):
            w = viol_grad[n * (k - 1) + i]
            if w < 1e-15:
                continue
            if t > wall_dists[i]:
                dx_pos = 0.5 - centers[i, 0]
                dx_neg = centers[i, 0] + 0.5
                dy_pos = 0.5 - centers[i, 1]
                dy_neg = centers[i, 1] + 0.5
                min_val = min(dx_pos, dx_neg, dy_pos, dy_neg)
                if abs(dx_pos - min_val) < 1e-12:
                    grad_centers[i, 0] += w * (-1.0)
                elif abs(dx_neg - min_val) < 1e-12:
                    grad_centers[i, 0] += w * (1.0)
                if abs(dy_pos - min_val) < 1e-12:
                    grad_centers[i, 1] += w * (-1.0)
                elif abs(dy_neg - min_val) < 1e-12:
                    grad_centers[i, 1] += w * (1.0)
        
        # Pair gradients
        for i in range(n):
            for j_idx in range(1, k):
                j = idxs[i, j_idx]
                w = viol_grad[i * (k - 1) + (j_idx - 1)]
                if w < 1e-15:
                    continue
                if t > pair_dists_half[i, j_idx - 1]:
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-15:
                        continue
                    unit = diff / dist
                    grad_centers[i] += w * unit / 2.0
                    grad_centers[j] -= w * unit / 2.0
    
    return obj, grad_centers, grad_t

def _final_polish(centers, max_iter=5000, time_budget=40.0):
    """High-precision polish using epigraph formulation with increasing beta."""
    centers = centers.copy()
    start_time = time.time()
    best_r = _get_radius(centers)
    best_centers = centers.copy()
    
    t = best_r
    beta = 50.0
    
    for it in range(max_iter):
        if _time_left() < 5.0 or (time.time() - start_time) > time_budget:
            break
        
        obj, gc, gt = _epigraph_objective_and_grad(centers, t, beta)
        
        # Simple gradient descent with line search
        lr_c = 0.002
        lr_t = 0.001
        
        trial_c = centers - lr_c * gc
        trial_t = t - lr_t * gt
        
        trial_c[:, 0] = np.clip(trial_c[:, 0], -0.5, 0.5)
        trial_c[:, 1] = np.clip(trial_c[:, 1], -0.5, 0.5)
        trial_t = max(trial_t, 0.0)
        
        # Check if this improves the actual radius
        r_trial = _get_radius(trial_c)
        if r_trial > best_r + 1e-14:
            best_r = r_trial
            best_centers = trial_c.copy()
            centers = trial_c.copy()
            t = trial_t
        
        # Increase beta periodically
        if it % 200 == 0 and beta < 5000:
            beta *= 1.8
    
    return best_centers, best_r

def _run_optimization(centers, time_budget=120.0):
    """Run the full optimization pipeline."""
    start_time = time.time()
    
    # Phase 1: Force-directed relaxation
    if _time_left() > 60.0:
        centers = _force_directed_phase(centers, max_iter=300, time_budget=25.0)
    
    # Phase 2: Projected gradient descent
    if _time_left() > 30.0:
        centers, r = _projected_gradient_descent(centers, max_iter=1500, beta=100.0, time_budget=35.0)
    
    # Phase 3: Final polish with epigraph
    if _time_left() > 15.0:
        centers, r = _final_polish(centers, max_iter=3000, time_budget=40.0)
    
    # Final clip
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Generate multiple initial configurations
    best_centers = None
    best_r = 0.0
    
    configs = [
        _generate_hex_initial(14),
        _generate_hex_initial(16),
        _generate_hex_initial(18),
        _generate_jittered_hex(_generate_hex_initial(15), jitter_scale=0.003, seed=1),
        _generate_jittered_hex(_generate_hex_initial(15), jitter_scale=0.003, seed=2),
    ]
    
    for i, cfg in enumerate(configs):
        if _time_left() < 30.0:
            break
        
        r = _get_radius(cfg)
        if r > best_r:
            best_r = r
            best_centers = cfg.copy()
        
        # Run optimization on this config if time permits
        if _time_left() > 40.0:
            opt_centers = _run_optimization(cfg, time_budget=min(40.0, _time_left() - 10.0))
            opt_r = _get_radius(opt_centers)
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()
    
    if best_centers is None:
        best_centers = _generate_hex_initial(15)
    
    return best_centers
# EVOLVE-BLOCK-END
