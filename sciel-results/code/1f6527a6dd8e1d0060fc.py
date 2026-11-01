"""Combined program: hex/ring init + explicit-gradient soft-min annealing + high-precision polish."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return max(0.0, 1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Pairwise distances using upper triangle only
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    min_pair_dist = np.min(dists) / 2.0
    
    # Boundary distance
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair_dist, min_boundary)

def _soft_min_and_grad(points, beta):
    """
    Compute soft-min of all constraints (pair distances and boundary distances)
    and its gradient with respect to points.
    
    Constraints:
      - For each pair (i,j): d_ij = ||p_i - p_j||  (we want to maximize min d_ij)
      - For each point i: b_i = 1 - ||p_i||       (we want to maximize min b_i)
    
    We define the objective as soft-min of {d_ij/2, b_i} over all i,j,i.
    """
    n = len(points)
    iu = np.triu_indices(n, k=1)
    
    # Pairwise distances
    diff = points[iu[0]] - points[iu[1]]  # (m, 2)
    d2 = np.sum(diff**2, axis=-1)  # (m,)
    d = np.sqrt(d2 + 1e-30)  # (m,)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)  # (n,)
    b = 1.0 - norms  # (n,)
    
    # All constraints: d_ij / 2 and b_i
    m = len(iu[0])
    constraints = np.concatenate([d / 2.0, b])  # (m + n,)
    
    # Soft-min: -logsumexp(-beta * x) / beta
    # For numerical stability, use logsumexp trick
    scaled = -beta * constraints
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft-min with respect to constraints
    # d(soft_min)/d(constraints[k]) = exp(-beta * constraints[k]) / sum(exp(-beta * constraints))
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # (m + n,)
    
    # Gradient with respect to points
    grad = np.zeros_like(points)
    
    # Pairwise part: for pair (i,j), constraint = d_ij / 2
    # d(d_ij/2)/d(p_i) = (p_i - p_j) / (2 * d_ij)
    # d(d_ij/2)/d(p_j) = (p_j - p_i) / (2 * d_ij)
    pair_weights = weights[:m]
    for idx in range(m):
        i = iu[0][idx]
        j = iu[1][idx]
        w = pair_weights[idx]
        if d[idx] < 1e-15:
            continue
        # Gradient contribution to point i
        grad[i] += w * (diff[idx] / (2.0 * d[idx]))
        # Gradient contribution to point j
        grad[j] += w * (-diff[idx] / (2.0 * d[idx]))
    
    # Boundary part: for point i, constraint = 1 - ||p_i||
    # d(1 - ||p_i||)/d(p_i) = -p_i / ||p_i||
    boundary_weights = weights[m:]
    for idx in range(n):
        w = boundary_weights[idx]
        if norms[idx] < 1e-15:
            continue
        grad[idx] -= w * (points[idx] / norms[idx])
    
    return soft_min_val, grad

def _objective_and_grad(points, beta):
    """
    Objective: maximize soft-min of constraints.
    We minimize the negative: f(x) = -soft_min
    """
    val, grad = _soft_min_and_grad(points, beta)
    return -val, -grad

def _gradient_optimize(points, time_limit, beta_schedule=None):
    """
    Gradient-based optimization with adaptive step size.
    Uses a schedule of beta values (from low to high) for progressive sharpening.
    """
    if beta_schedule is None:
        beta_schedule = [10, 50, 200, 500, 2000, 20000, 100000]
    
    current_pts = points.copy()
    current_score = _compute_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    start_time = time.time()
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        # Sub-optimization at this beta
        sub_start = time.time()
        sub_limit = min(5.0, max(0.5, time_limit - (time.time() - start_time)))
        
        lr = 1e-4
        min_lr = 1e-8
        lr_decay = 0.999
        max_stall = 100
        stall = 0
        prev_val = None
        
        while time.time() - sub_start < sub_limit:
            val, grad = _objective_and_grad(current_pts, beta)
            
            # Adaptive step size
            step = lr * grad
            new_pts = current_pts + step
            
            # Ensure points stay within unit circle (with small margin)
            norms = np.linalg.norm(new_pts, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                new_pts[mask] *= (0.999 / norms[mask])[:, None]
            
            new_score = _compute_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
                stall = 0
                # Increase learning rate slightly if improving
                lr = min(lr * 1.05, 1e-3)
            else:
                stall += 1
                lr *= lr_decay
                if lr < min_lr:
                    break
                if stall > max_stall:
                    break
        
        # After each beta phase, recompute best
        final_score = _compute_score(current_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = current_pts.copy()
    
    return best_pts, best_score

def _hex_lattice_init(n):
    """Initialize points in a hexagonal lattice clipped to the unit circle."""
    best_pts = None
    best_score = -1.0
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        spacing = 2.0 / (np.sqrt(n) * scale)
        
        nx = int(2.0 / spacing) + 5
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 5
        
        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0
        
        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                    pts_list.append([px, py])
        
        if len(pts_list) < n:
            continue
            
        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
        
    return best_pts

def _ring_decomposition_init(n):
    """Initialize points in concentric rings with optimized radii."""
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Determine number of rings based on area
    # Approximate: ring k has radius ~ k/sqrt(n)
    num_rings = int(np.sqrt(n) / 1.5) + 2
    
    # Distribute points roughly proportional to circumference
    radii = np.linspace(0.1, 0.95, num_rings)
    circumferences = 2 * np.pi * radii
    total_circ = np.sum(circumferences)
    
    counts = np.round(remaining * circumferences / total_circ).astype(int)
    counts = np.maximum(counts, 1)
    
    # Adjust to match remaining
    diff = remaining - np.sum(counts)
    if diff > 0:
        for i in range(diff):
            counts[i % len(counts)] += 1
    elif diff < 0:
        for i in range(abs(diff)):
            if counts[i % len(counts)] > 1:
                counts[i % len(counts)] -= 1
    
    for k, (count, r) in enumerate(zip(counts, radii)):
        if count == 0:
            continue
        d_theta = 2 * np.pi / count
        offset = np.pi / count * (k % 2)
        for j in range(count):
            theta = j * d_theta + offset
            points.append([r * np.cos(theta), r * np.sin(theta)])
            
    points = np.array(points[:n])
    return points

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate diverse initializations
    inits = []
    
    hex_pts = _hex_lattice_init(N)
    inits.append(hex_pts)
    
    ring_pts = _ring_decomposition_init(N)
    inits.append(ring_pts)
    
    # Random init
    rand_pts = np.random.rand(N, 2) * 2 - 1
    norms = np.linalg.norm(rand_pts, axis=1)
    rand_pts = rand_pts * (0.95 / norms)[:, None]
    inits.append(rand_pts)
    
    best_overall_pts = None
    best_overall_score = -1.0
    
    # Allocate time for gradient optimization on each init
    # ~145s total. Give each init ~40s, then polish best with remaining time.
    grad_time_per_init = 35.0
    
    for init in inits:
        if time.time() > deadline - 20:  # Leave 20s for final polish
            break
        
        pts, score = _gradient_optimize(init, grad_time_per_init)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts
    
    # Final high-precision polish with very high beta
    if best_overall_pts is not None and time.time() < deadline - 3:
        # Use very high beta for precise local optimization
        high_beta_schedule = [50000, 100000, 200000, 500000]
        pts, score = _gradient_optimize(best_overall_pts, time_limit=15.0, beta_schedule=high_beta_schedule)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts
    
    # Final validation and clipping
    if best_overall_pts is None:
        best_overall_pts = hex_pts
        
    best_overall_pts = best_overall_pts.reshape(-1, 2)
    norms = np.linalg.norm(best_overall_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_overall_pts[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
        
    if not np.all(np.isfinite(best_overall_pts)):
        return hex_pts
        
    return best_overall_pts

# EVOLVE-BLOCK-END
