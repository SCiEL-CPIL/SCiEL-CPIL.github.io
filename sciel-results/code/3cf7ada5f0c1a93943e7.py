"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_lattice_init(n):
    """Generate n points in a hexagonal lattice within the unit disk."""
    spacing = 0.15
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= n:
            # Sort by distance from origin, take closest n
            dist = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dist, kind='stable')[:n]
            return points[idx]
        spacing *= 0.95

def _calc_score(points):
    """Calculate the packing score: min of pair distances/2 and container margins."""
    n = len(points)
    # Pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Container margins
    norms = np.sqrt(np.sum(points * points, axis=1))
    min_margin = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_margin)

def _objective_and_grad(points, beta):
    """
    Compute soft-min objective and gradient.
    We want to maximize r where r = min(min_pair_dist/2, min_margin).
    Using soft-min: r_soft = -1/beta * log(sum(exp(-beta*x)))
    where x_i are the constraints (pair distances/2 and margins).
    """
    n = len(points)
    
    # Build constraint vector
    # Pair constraints: dist(i,j)/2 for i<j
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_constr = dists[iu] / 2.0
    
    # Container constraints: 1 - |p_i|
    norms = np.sqrt(np.sum(points * points, axis=1))
    container_constr = 1.0 - norms
    
    # Combine all constraints
    all_constr = np.concatenate([pair_constr, container_constr])
    
    # Soft-min: -1/beta * log(sum(exp(-beta*x)))
    # For numerical stability, use log-sum-exp trick
    min_val = np.min(all_constr)
    shifted = -beta * (all_constr - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_val - log_sum / beta
    
    # Gradient of soft-min w.r.t. points
    # d(soft_min)/d(constr_k) = -exp(-beta*(constr_k - min_val)) / sum(exp(-beta*(constr_j - min_val)))
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient w.r.t. pair distances
    # d(dist(i,j))/d(p_i) = (p_i - p_j) / dist(i,j)
    # d(pair_constr)/d(p_i) = 0.5 * (p_i - p_j) / dist(i,j)
    
    grad = np.zeros_like(points)
    
    # Pair contributions
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = dists[i, j]
        if d > 1e-12:
            diff = points[i] - points[j]
            grad[i] += 0.5 * w * diff / d
            grad[j] -= 0.5 * w * diff / d
    
    # Container contributions
    for i in range(n):
        w = weights[len(pair_constr) + i]
        if norms[i] > 1e-12:
            grad[i] += w * points[i] / norms[i]
    
    return soft_min, grad

def _polish(points, max_iter=2000, tol=1e-12):
    """Final polish using direct optimization of the nonsmooth objective."""
    n = len(points)
    points = points.copy()
    
    best_score = _calc_score(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        # Calculate current score and identify critical constraints
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        norms = np.sqrt(np.sum(points * points, axis=1))
        margins = 1.0 - norms
        
        # Find the minimum constraint
        iu = np.triu_indices(n, k=1)
        pair_vals = dists[iu] / 2.0
        all_vals = np.concatenate([pair_vals, margins])
        min_val = np.min(all_vals)
        
        # Identify active constraints (within tolerance)
        active_pairs = np.where(pair_vals <= min_val + 1e-10)[0]
        active_containers = np.where(margins <= min_val + 1e-10)[0]
        
        if len(active_pairs) == 0 and len(active_containers) == 0:
            break
        
        # Simple gradient ascent on the active constraints
        grad = np.zeros_like(points)
        
        for idx in active_pairs:
            i, j = iu[0][idx], iu[1][idx]
            d = dists[i, j]
            if d > 1e-12:
                diff = points[i] - points[j]
                grad[i] += diff / d
                grad[j] -= diff / d
        
        for i in active_containers:
            if norms[i] > 1e-12:
                grad[i] += points[i] / norms[i]
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        grad /= gnorm
        
        # Try a small step
        step_size = 1e-4
        new_points = points + step_size * grad
        
        new_score = _calc_score(new_points)
        
        if new_score > best_score + 1e-15:
            points = new_points
            best_score = new_score
            best_points = points.copy()
        else:
            # Try smaller step
            step_size *= 0.1
            new_points = points + step_size * grad
            new_score = _calc_score(new_points)
            if new_score > best_score + 1e-15:
                points = new_points
                best_score = new_score
                best_points = points.copy()
            else:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    n = N
    
    # Initialize with hexagonal lattice
    points = _hex_lattice_init(n)
    best_points = points.copy()
    best_score = _calc_score(points)
    
    # Phase 1: Smooth optimization with soft-min
    beta = 50.0
    lr = 0.01
    
    while time.time() < deadline - 30.0:  # Leave 30s for polish
        # Compute objective and gradient
        obj, grad = _objective_and_grad(points, beta)
        
        # Gradient ascent
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        
        points = points + lr * grad / gnorm
        
        # Project back into valid region (simple: just keep as is, constraints handled by objective)
        current_score = _calc_score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Adapt learning rate
        if current_score > obj:
            lr *= 0.9
        else:
            lr *= 0.5
        
        # Increase beta over time for sharper approximation
        beta *= 1.05
    
    # Phase 2: Polish with direct optimization
    if time.time() < deadline - 5.0:
        polished, polished_score = _polish(best_points, max_iter=5000, tol=1e-12)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    # Final validation
    final_score = _calc_score(best_points)
    if final_score <= 0:
        # Fallback: return initial lattice
        return _hex_lattice_init(n)
    
    return best_points
# EVOLVE-BLOCK-END
