"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

DEADLINE = 140.0  # Safety margin before 150s limit
_start_time = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start_time)


def _compute_radius(centers):
    """Compute the score: largest common radius for given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are within the square
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    # Min distance between pairs / 2
    dists = np.zeros((N, N))
    for i in range(N):
        diff = centers[i+1:] - centers[i]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists[i, i+1:] = d
        dists[i+1:, i] = d
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    if len(pair_dists) == 0:
        return 0.0
    min_pair = np.min(pair_dists)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _pair_dists(centers):
    """Compute all pair distances efficiently."""
    n = len(centers)
    dists = np.zeros((n, n))
    for i in range(n):
        diff = centers[i+1:] - centers[i]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists[i, i+1:] = d
        dists[i+1:, i] = d
    return dists


def _wall_dists(centers):
    """Compute distance from each center to nearest wall."""
    return np.min(0.5 - np.abs(centers), axis=1)


def _softmin_obj_and_grad(x, beta):
    """
    Epigraph formulation with auxiliary variable t.
    x = [centers_flat, t] where t is the radius.
    
    Objective: minimize -t + penalty for constraints violated.
    Constraints: 
      - For all i<j: |c_i - c_j|/2 >= t  =>  |c_i - c_j| - 2t >= 0
      - For all i: 0.5 - |c_ik| >= t  =>  0.5 - |c_ik| - t >= 0
    
    We use a smooth penalty: sum of soft-min of violations.
    """
    n = N
    centers = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    # Compute pair distances
    dists = _pair_dists(centers)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    wall_dists = _wall_dists(centers)
    
    # Violations: we want pair_dists/2 >= t and wall_dists >= t
    # violation_pair = t - pair_dists/2 (positive if violated)
    # violation_wall = t - wall_dists (positive if violated)
    
    # Use soft-max to create smooth penalty
    # penalty = softmax(violations) / beta
    viol_pair = t - pair_dists / 2.0
    viol_wall = t - wall_dists
    
    all_viol = np.concatenate([viol_pair, viol_wall])
    
    # Soft-max approximation of max(0, violations)
    # Use: softplus(v) = log(1 + exp(v/beta)) * beta
    # But we want sum of positive parts, so use:
    # sum(softplus(violations))
    
    max_v = np.max(all_viol)
    if max_v < -beta * 10:
        penalty = 0.0
        grad_penalty = np.zeros_like(x)
    else:
        # softplus(v) = log(1 + exp(v/beta)) * beta
        # grad softplus(v) = exp(v/beta) / (1 + exp(v/beta))
        
        shifted = all_viol / beta
        # Numerical stability
        shifted = np.where(shifted > 50, 50, shifted)
        shifted = np.where(shifted < -50, -50, shifted)
        
        exp_shifted = np.exp(shifted)
        one_plus_exp = 1.0 + exp_shifted
        
        # softplus value
        softplus_vals = beta * np.log(one_plus_exp)
        penalty = np.sum(softplus_vals)
        
        # Gradient of softplus w.r.t. v: exp(v/beta) / (1 + exp(v/beta))
        grad_softplus = exp_shifted / one_plus_exp
        
        # Now build gradient w.r.t. x
        grad_penalty = np.zeros_like(x)
        
        # Gradient w.r.t. t: sum of grad_softplus
        grad_t = np.sum(grad_softplus)
        grad_penalty[2*n] = grad_t
        
        # Gradient w.r.t. centers
        # For pair (i,j): violation = t - dist(i,j)/2
        # d(violation)/d(c_i) = -(c_i - c_j) / dist(i,j) * (1/2) * 2 = -(c_i-c_j)/dist
        # Wait: d(dist)/d(c_i) = (c_i - c_j)/dist
        # d(violation)/d(c_i) = -d(dist)/d(c_i) * (1/2) = -(c_i-c_j)/(2*dist)
        # d(violation)/d(c_j) = -d(dist)/d(c_j) * (1/2) = (c_i-c_j)/(2*dist)
        
        # For wall: violation = t - (0.5 - |c_ik|)
        # d(violation)/d(c_ik) = -d(0.5-|c_ik|)/d(c_ik) = sign(c_ik)
        
        # Process pair violations
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if idx >= len(viol_pair):
                    break
                g = grad_softplus[idx]
                if g != 0.0:
                    diff = centers[i] - centers[j]
                    d = np.linalg.norm(diff)
                    if d > 1e-10:
                        # d(violation)/d(c_i) = -diff / (2*d)
                        grad_penalty[2*i] -= g * diff[0] / (2*d)
                        grad_penalty[2*i+1] -= g * diff[1] / (2*d)
                        # d(violation)/d(c_j) = diff / (2*d)
                        grad_penalty[2*j] += g * diff[0] / (2*d)
                        grad_penalty[2*j+1] += g * diff[1] / (2*d)
                idx += 1
        
        # Process wall violations
        for i in range(n):
            if idx >= len(viol_pair) + len(viol_wall):
                break
            g = grad_softplus[idx]
            if g != 0.0:
                # d(violation)/d(c_ik) = sign(c_ik)
                for k in range(2):
                    if centers[i, k] > 0:
                        grad_penalty[2*i+k] += g * 1.0
                    elif centers[i, k] < 0:
                        grad_penalty[2*i+k] -= g * 1.0
                    else:
                        grad_penalty[2*i+k] += 0.0  # subgradient, use 0
            idx += 1
    
    # Objective: -t + penalty
    obj = -t + penalty
    
    # Gradient: [-grad_penalty_centers, -1 + grad_penalty_t]
    grad_obj = np.zeros_like(x)
    grad_obj[:2*n] = grad_penalty[:2*n]
    grad_obj[2*n] = -1.0 + grad_penalty[2*n]
    
    return obj, grad_obj


def _anneal_optimize(centers, t_init, beta_start=1.0, beta_end=0.01, max_iter=2000):
    """
    Anneal the soft-min temperature while optimizing.
    """
    x = np.concatenate([centers.flatten(), [t_init]])
    current_obj, _ = _softmin_obj_and_grad(x, beta_start)
    current_r = _compute_radius(centers)
    
    best_x = x.copy()
    best_r = current_r
    best_obj = current_obj
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Anneal beta
        progress = it / max_iter
        beta = beta_start * (beta_end / beta_start) ** progress
        
        try:
            obj, grad = _softmin_obj_and_grad(x, beta)
        except:
            break
        
        # Gradient descent with line search
        lr = 0.01
        improved = False
        for ls in range(20):
            new_x = x - lr * grad
            # Clip centers to square
            new_centers = new_x[:2*N].reshape(N, 2)
            new_t = new_x[2*N]
            
            if np.any(np.abs(new_centers) > 0.5):
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_x[:2*N] = new_centers.flatten()
            
            try:
                new_obj, _ = _softmin_obj_and_grad(new_x, beta)
            except:
                break
            
            if new_obj < obj - 1e-10:
                x = new_x
                obj = new_obj
                improved = True
                break
            else:
                lr *= 0.5
        
        if not improved:
            # Try smaller step
            lr = 1e-4
            for ls in range(10):
                new_x = x - lr * grad
                new_centers = new_x[:2*N].reshape(N, 2)
                if np.any(np.abs(new_centers) > 0.5):
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    new_x[:2*N] = new_centers.flatten()
                try:
                    new_obj, _ = _softmin_obj_and_grad(new_x, beta)
                except:
                    break
                if new_obj < obj - 1e-10:
                    x = new_x
                    obj = new_obj
                    improved = True
                    break
                else:
                    lr *= 0.5
        
        if not improved:
            break
        
        # Update best
        new_centers = x[:2*N].reshape(N, 2)
        new_t = x[2*N]
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r + 1e-10:
            best_r = new_r
            best_x = x.copy()
            best_obj = obj
    
    return best_x[:2*N].reshape(N, 2), best_r


def _polish_exact(centers, max_iter=5000):
    """
    High-precision polish: optimize only the active constraints.
    Find the minimum radius, identify which constraints are active,
    and solve a QP-like problem on those.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    
    for outer in range(20):
        if _time_left() < 5.0:
            break
        
        # Find active constraints
        dists = _pair_dists(centers)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        wall_dists = _wall_dists(centers)
        
        min_pair = np.min(pair_dists)
        min_wall = np.min(wall_dists)
        current_r = min(min_pair / 2.0, min_wall)
        
        # Active pair constraints: those within 1e-6 of min
        active_pairs = []
        for idx in range(len(pair_dists)):
            if pair_dists[idx] < min_pair + 1e-6:
                # Find which pair
                i = idx // (N - idx - 1) if idx < N*(N-1)//2 else None
                # More careful index mapping
                count = 0
                for i in range(N):
                    for j in range(i+1, N):
                        if count == idx:
                            active_pairs.append((i, j))
                            break
                        count += 1
                    if count == idx:
                        break
        
        # Active wall constraints
        active_walls = []
        for i in range(N):
            if wall_dists[i] < min_wall + 1e-6:
                active_walls.append(i)
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Try to move centers to increase the minimum
        # Simple approach: for each active constraint, try to increase the distance
        improved = False
        for (i, j) in active_pairs[:10]:  # Limit to first 10 active pairs
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            # Move i and j apart
            step = diff / d * 1e-5
            new_centers = centers.copy()
            new_centers[i] += step
            new_centers[j] -= step
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_r = _compute_radius(new_centers)
            if new_r > current_r + 1e-10:
                centers = new_centers
                current_r = new_r
                improved = True
                break
        
        if not improved:
            for i in active_walls[:10]:
                # Move center i away from nearest wall
                for k in range(2):
                    if centers[i, k] > 0:
                        # Move left
                        new_centers = centers.copy()
                        new_centers[i, k] -= 1e-5
                    else:
                        # Move right
                        new_centers = centers.copy()
                        new_centers[i, k] += 1e-5
                    new_centers = np.clip(new_centers, -0.5, 0.5)
                    new_r = _compute_radius(new_centers)
                    if new_r > current_r + 1e-10:
                        centers = new_centers
                        current_r = new_r
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return centers, current_r


def _hex_initial():
    """Hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    # Hexagonal lattice
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    spacing = 1.0 / (rows * 0.866)
    
    idx = 0
    for row in range(rows):
        if idx >= N:
            break
        y = -0.5 + spacing * (row + 0.5) * 0.866
        if abs(y) > 0.5:
            continue
        offset = (row % 2) * spacing * 0.5
        max_cols = int((1.0 - 2 * spacing * 0.25) / spacing) + 1
        for col in range(max_cols):
            if idx >= N:
                break
            x = -0.5 + spacing * (col + 0.5) + offset
            if abs(x) <= 0.5:
                centers[idx] = [x, y]
                idx += 1
        if idx >= N:
            break
    
    if idx < N:
        for i in range(idx, N):
            centers[i] = [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)]
    
    return centers[:N]


def _grid_initial(m):
    """Grid initialization with m columns."""
    centers = []
    n_rows = int(np.ceil(N / m))
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n_rows) + 0.5) / n_rows - 0.5
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    return np.array(centers[:N])


def _random_initial():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 2))


def construct():
    """
    Construct the centers of 200 equal circles inside [-1/2, 1/2]^2
    to maximize the common radius.
    """
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_r = -1.0
    
    # Try multiple initializations
    initializations = []
    
    for m in [10, 11, 12, 13, 14]:
        initializations.append(_grid_initial(m))
    
    initializations.append(_hex_initial())
    
    for _ in range(5):
        initializations.append(_random_initial())
    
    # Process each initialization
    for init_idx, init_centers in enumerate(initializations):
        if _time_left() < 10.0:
            break
        
        r = _compute_radius(init_centers)
        if r > best_r:
            best_r = r
            best_centers = init_centers.copy()
        
        # Use annealed soft-min optimization
        try:
            t_init = r * 1.01  # Slightly above current radius
            opt_centers, opt_r = _anneal_optimize(init_centers, t_init, 
                                                   beta_start=0.5, beta_end=0.01,
                                                   max_iter=1000)
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Final polish: exact local optimization on best
    if best_centers is not None and _time_left() > 15.0:
        try:
            final_centers, final_r = _polish_exact(best_centers, max_iter=5000)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Second round of polish if time permits
    if best_centers is not None and _time_left() > 10.0:
        try:
            final_centers, final_r = _polish_exact(best_centers, max_iter=3000)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
