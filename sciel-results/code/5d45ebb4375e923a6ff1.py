"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    min_pair = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_pair:
                min_pair = d
    return min(min_pair / 2.0, min_boundary)

def _softmin_logsumexp(a, b):
    """Soft minimum of a and b using log-sum-exp trick."""
    m = np.maximum(a, b)
    return m - np.log(np.exp(a - m) + np.exp(b - m))

def _grad_softmin_neg(pos, beta):
    """
    Gradient of -softmin(log(dist_pair), log(dist_boundary)) 
    using soft-min approximation.
    """
    n = len(pos)
    grad = np.zeros_like(pos)
    
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(pos), axis=1)
    
    # Use soft-min: we want to maximize min(min_pair/2, min_boundary)
    # Equivalent to maximizing log(min_pair/2) and log(min_boundary)
    # Use soft-min on the log values
    
    log_pair = np.log(dist + 1e-15)
    log_bound = np.log(dist_boundary + 1e-15)
    
    # Soft-min over all pairs and boundaries
    # We approximate: maximize the soft-min of all constraints
    
    # For each pair (i,j), contribution to gradient
    # softmin(a,b) = -logsumexp(-a, -b)
    # d/dx softmin(a,b) = (exp(-a)*da/dx + exp(-b)*db/dx) / (exp(-a) + exp(-b))
    
    # Work with negative logs for soft-min
    neg_log_pair = -log_pair  # shape (n, n)
    neg_log_bound = -log_bound  # shape (n,)
    
    # Combine all constraints: we want softmin of all (pair distances and boundary distances)
    # For efficiency, compute gradient contributions
    
    # Pairwise gradients
    # For pair (i,j): d/dpos_i softmin = ?
    # Let's use a simpler approach: gradient ascent on soft-min
    
    # Actually, let's use a direct gradient computation
    # Objective: f(pos) = softmin(min_pair/2, min_boundary)
    # We'll use: f = -logsumexp(-log(min_pair/2), -log(min_boundary))
    
    # Simplify: work with the log of the minimum
    # Use soft-min: s = -logsumexp(-log(d_ij/2) for all i<j, -log(b_i) for all i)
    
    # Compute weights
    # For pairs
    w_pair = np.exp(-log_pair + np.log(2))  # weight for pair (i,j)
    # For boundaries  
    w_bound = np.exp(-log_bound)
    
    total_w = np.sum(w_pair) + np.sum(w_bound)
    
    if total_w < 1e-30:
        return grad
    
    # Gradient from pairs
    # d/dpos_i [softmin] involves sum over j of w_pair[i,j] * (pos_i - pos_j) / dist[i,j]
    # But we need to be careful with the soft-min derivative
    
    # Use a simpler gradient: for each constraint, weight by its "tightness"
    # and push points apart
    
    # Pairwise repulsion weighted by soft-min weights
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if d < 1e-10:
                continue
            # Weight based on how tight this constraint is
            # Tighter constraints get higher weight in soft-min
            w = w_pair[i, j] / total_w
            # Gradient: push i and j apart
            direction = (pos[i] - pos[j]) / d
            grad[i] += w * direction
            grad[j] -= w * direction
    
    # Boundary attraction: push points toward center if near boundary
    for i in range(n):
        w = w_bound[i] / total_w
        # Gradient of -log(boundary_dist): push toward center
        # d/dpos_i log(0.5 - max(|x_i|, |y_i|)) 
        # If |x_i| >= |y_i|: d/dx = -1/(0.5-|x_i|) * sign(x_i)
        # If |y_i| > |x_i|: d/dy = -1/(0.5-|y_i|) * sign(y_i)
        abs_x = abs(pos[i, 0])
        abs_y = abs(pos[i, 1])
        if abs_x >= abs_y:
            if abs_x > 0.49:
                grad[i, 0] -= w * np.sign(pos[i, 0]) / (0.5 - abs_x + 1e-15)
        else:
            if abs_y > 0.49:
                grad[i, 1] -= w * np.sign(pos[i, 1]) / (0.5 - abs_y + 1e-15)
    
    return grad

def _initial_hex_grid(n):
    """Create initial hexagonal lattice packing."""
    # Hexagonal lattice with spacing 1
    # For n points, use approximately sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n * 2 / 3)))
    cols = int(np.ceil(n / rows))
    
    points = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col * 1.0
            y = row * np.sqrt(3) / 2
            if row % 2 == 1:
                x += 0.5
            points.append([x, y])
            count += 1
        if count >= n:
            break
    
    points = np.array(points[:n])
    
    # Center the lattice
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    points -= center
    
    # Scale to fit in [-0.5, 0.5]^2 with margin
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        scale = 0.45 / max_coord
        points *= scale
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Start with hexagonal lattice
    pos = _initial_hex_grid(N)
    
    # Ensure valid initial position
    pos = np.clip(pos, -0.49, 0.49)
    
    best_pos = pos.copy()
    best_r = _compute_radius(pos)
    
    # Gradient ascent with adaptive step size
    step_size = 0.01
    beta = 100.0  # Sharpness of soft-min
    
    # Precompute for efficiency
    n = N
    
    while time.time() < deadline - 2:
        # Compute gradient
        grad = _grad_softmin_neg(pos, beta)
        
        # Adaptive step size
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-10:
            break
        
        # Take step
        new_pos = pos + step_size * grad
        
        # Keep within bounds
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        # Check improvement
        new_r = _compute_radius(new_pos)
        
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pos.copy()
            pos = new_pos
            # Slightly increase step size if improving
            step_size = min(step_size * 1.1, 0.05)
        else:
            # Decrease step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
        
        # Occasionally re-center to avoid drift
        if int((time.time() - start_time) * 10) % 10 == 0:
            center = (pos.max(axis=0) + pos.min(axis=0)) / 2
            pos -= center
            pos = np.clip(pos, -0.49, 0.49)
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 1:
        pos = best_pos.copy()
        small_step = 0.001
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            grad = _grad_softmin_neg(pos, beta)
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-12:
                break
            new_pos = pos + small_step * grad
            new_pos = np.clip(new_pos, -0.49, 0.49)
            new_r = _compute_radius(new_pos)
            if new_r > _compute_radius(pos):
                pos = new_pos
            else:
                small_step *= 0.5
    
    # Return best position found
    return best_pos
# EVOLVE-BLOCK-END
