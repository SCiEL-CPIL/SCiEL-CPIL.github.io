"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    Uses a smooth soft-min based penalty that is C1 continuous.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    # 1. Boundary Violations - smooth via soft-min of distance to each wall
    # dist_to_left = pos_x - r, dist_to_right = 1 - pos_x - r, etc.
    # We want penalty when any dist < 0. Use -soft_min of the 4 distances.
    d_left = pos[:, 0] - radii
    d_right = 1.0 - pos[:, 0] - radii
    d_bottom = pos[:, 1] - radii
    d_top = 1.0 - pos[:, 1] - radii
    
    # soft_min(a, b, c, d) via log-sum-exp of -d_i (i.e., soft-min of d_i)
    # soft_min(d) = -logsumexp(-d_i) / beta  for large beta
    beta = 20.0
    d_all = np.stack([d_left, d_right, d_bottom, d_top], axis=-1)  # (N, 4)
    # soft_min = -logsumexp(-d)/beta
    neg_d = -d_all * beta
    neg_d_max = np.max(neg_d, axis=-1, keepdims=True)
    logsumexp_neg = neg_d_max + np.log(np.sum(np.exp(neg_d - neg_d_max), axis=-1, keepdims=True))
    soft_min_d = -logsumexp_neg.squeeze(-1) / beta  # (N,)
    
    # Penalty: for each circle, penalty = max(0, -soft_min_d)^2
    # Since soft_min_d < min(d_i), when all d_i > 0, soft_min_d > 0 → no penalty
    # When some d_i < 0, soft_min_d < 0 → penalty
    boundary_viol = np.maximum(0.0, -soft_min_d)
    boundary_penalty = np.sum(boundary_viol ** 2)
    
    # 2. Pairwise Overlap Violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    # gap = dist - min_dist; negative when overlapping
    gap = dist - min_dist
    
    # smooth penalty: use soft version of max(0, -gap)^2
    # For pairs where gap < 0: penalty = gap^2 (since -gap > 0)
    # Use: penalty = (soft_max(0, -gap))^2 where soft_max(0, a) = log(1+exp(beta*a))/beta
    beta_pair = 30.0
    neg_gap = -gap * beta_pair
    # soft_max(0, -gap) = log(1 + exp(neg_gap)) / beta_pair  (numerically stable)
    neg_gap_clipped = np.clip(neg_gap, -50, 50)
    soft_viol = np.log1p(np.exp(neg_gap_clipped)) / beta_pair  # (N, N)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_penalty = np.sum(soft_viol[mask] ** 2)
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(300):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r - 1e-10:
                radii[i] = max(max_r * 0.99995, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    # Scale both radii to make them just touch with small gap
                    k = dist / (sum_r + 1e-12) * 0.99995
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        # Sample 3000 random points for better coverage
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        for _ in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _force_directed_anneal(pos, radii, steps=500, start_temp=0.05, end_temp=0.001, rng=None):
    """
    Force-directed annealing to escape local optima.
    Applies repulsive forces between overlapping circles and attractive forces
    to pull circles towards the center of available space.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Calculate forces
        forces = np.zeros((N, 2))
        
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r and dist > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = sum_r - dist
                    force_mag = overlap * 10.0
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attractive forces to center (weak)
        for i in range(N):
            forces[i, 0] += (0.5 - pos[i, 0]) * 0.1
            forces[i, 1] += (0.5 - pos[i, 1]) * 0.1
        
        # Update positions with noise
        new_pos = pos + forces * 0.01 + rng.normal(0, temp, (N, 2))
        
        # Keep within bounds
        for i in range(N):
            new_pos[i, 0] = np.clip(new_pos[i, 0], radii[i], 1.0 - radii[i])
            new_pos[i, 1] = np.clip(new_pos[i, 1], radii[i], 1.0 - radii[i])
        
        # Try to expand radii
        new_radii = radii.copy()
        for i in range(N):
            # Max radius allowed by walls
            max_r_wall = min(new_pos[i, 0], 1.0-new_pos[i, 0], new_pos[i, 1], 1.0-new_pos[i, 1])
            
            # Max radius allowed by other circles
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = new_pos[i, 0] - new_pos[j, 0]
                dy = new_pos[i, 1] - new_pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - new_radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            # Expand slightly if possible
            if max_r_circ > new_radii[i]:
                new_radii[i] = new_radii[i] * 0.99 + max_r_circ * 0.01
        
        # Check validity
        valid = True
        for i in range(N):
            if new_pos[i, 0] < new_radii[i] or new_pos[i, 0] > 1.0 - new_radii[i]:
                valid = False
                break
            if new_pos[i, 1] < new_radii[i] or new_pos[i, 1] > 1.0 - new_radii[i]:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = new_pos[i, 0] - new_pos[j, 0]
                    dy = new_pos[i, 1] - new_pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
            pos = new_pos
            radii = new_radii
        else:
            # Reject move
            pass
            
    return best_pos, best_radii

def _vectorized_force_anneal(pos, radii, steps=500, start_temp=0.04, end_temp=0.0005, rng=None):
    """Vectorized force-directed annealing - much faster than loop version."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(steps):
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Vectorized pairwise distances
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy + 1e-12)
        
        # Repulsive forces (vectorized)
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0, sum_r - dist)
        force_mag = overlap * 15.0 / (dist + 1e-10)
        
        forces = np.zeros((N, 2))
        forces[:, 0] = np.sum(force_mag * dx, axis=1)
        forces[:, 1] = np.sum(force_mag * dy, axis=1)
        
        # Weak centering force
        forces[:, 0] += (0.5 - pos[:, 0]) * 0.05
        forces[:, 1] += (0.5 - pos[:, 1]) * 0.05
        
        # Update positions
        new_pos = pos + forces * 0.005 + rng.normal(0, temp, (N, 2))
        new_pos[:, 0] = np.clip(new_pos[:, 0], radii, 1.0 - radii)
        new_pos[:, 1] = np.clip(new_pos[:, 1], radii, 1.0 - radii)
        
        # Try to expand radii (vectorized max radius computation)
        new_radii = radii.copy()
        # Vectorized: compute max allowed radius for each circle
        # Wall constraint
        wall_max = np.minimum(np.minimum(new_pos[:, 0], 1.0 - new_pos[:, 0]),
                              np.minimum(new_pos[:, 1], 1.0 - new_pos[:, 1]))
        # Circle constraint: for each i, min over j != i of (dist_ij - r_j)
        dist_mat = np.sqrt((new_pos[:, 0, np.newaxis] - new_pos[np.newaxis, :, 0])**2 +
                           (new_pos[:, 1, np.newaxis] - new_pos[np.newaxis, :, 1])**2 + 1e-12)
        # Set diagonal to inf so it doesn't affect min
        np.fill_diagonal(dist_mat, np.inf)
        circ_allowed = dist_mat - radii[np.newaxis, :]  # (N, N): dist_ij - r_j
        circ_max = np.min(circ_allowed, axis=1)
        
        max_allowed = np.minimum(wall_max, circ_max)
        max_allowed = np.maximum(max_allowed, 1e-6)
        
        # Blend toward max allowed
        expand_mask = max_allowed > new_radii
        new_radii[expand_mask] = new_radii[expand_mask] * 0.97 + max_allowed[expand_mask] * 0.03
        
        # Quick validity check
        valid = True
        if np.any(new_pos[:, 0] < new_radii - 1e-8) or np.any(new_pos[:, 0] > 1.0 - new_radii + 1e-8):
            valid = False
        elif np.any(new_pos[:, 1] < new_radii - 1e-8) or np.any(new_pos[:, 1] > 1.0 - new_radii + 1e-8):
            valid = False
        else:
            dx2 = new_pos[:, 0, np.newaxis] - new_pos[np.newaxis, :, 0]
            dy2 = new_pos[:, 1, np.newaxis] - new_pos[np.newaxis, :, 1]
            dist2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + 1e-12)
            min_d = new_radii[:, np.newaxis] + new_radii[np.newaxis, :]
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            if np.any(dist2[mask] < min_d[mask] - 1e-7):
                valid = False
        
        if valid:
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
            pos = new_pos
            radii = new_radii
    
    return best_pos, best_radii

def _local_pair_optimize(pos, radii, tol=1e-10, max_iters=500):
    """
    High-precision local optimization focusing on pairs that are nearly touching.
    For each pair (i,j) that is close, solve the 2D subproblem of maximizing
    r_i + r_j subject to dist >= r_i + r_j and wall constraints, then iterate.
    This is a final polish that exploits the active-set structure.
    """
    pos = pos.copy()
    radii = radii.copy()
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pairs = np.argwhere(mask)
    
    for iteration in range(max_iters):
        improved = False
        
        for i, j in pairs:
            xi, yi = pos[i]
            xj, yj = pos[j]
            ri, rj = radii[i], radii[j]
            
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Check if this pair is "active" (nearly touching)
            if dist < ri + rj + 0.01:
                # Try to increase both radii while maintaining feasibility
                # Direction: move circles apart along their connecting line
                if dist > 1e-10:
                    ux, uy = dx / dist, dy / dist
                else:
                    ux, uy = 1.0, 0.0
                
                # How much can we push apart?
                # Circle i can move in direction (ux, uy), circle j in (-ux, -uy)
                # Max push for i: limited by walls
                push_i_max = min(xi + ux*10 - ri, 1.0 - (xi + ux*10) - ri,
                                 yi + uy*10 - ri, 1.0 - (yi + uy*10) - ri)
                # More carefully: parametric
                # x_i(t) = xi + t*ux, y_i(t) = yi + t*uy
                # Need: ri <= x_i(t) <= 1-ri, ri <= y_i(t) <= 1-ri
                # t >= ri - xi (if ux > 0) or t <= (1-ri) - xi (if ux > 0)
                t_min_i, t_max_i = -1e10, 1e10
                if ux > 1e-10:
                    t_min_i = max(t_min_i, (ri - xi) / ux)
                    t_max_i = min(t_max_i, (1.0 - ri - xi) / ux)
                elif ux < -1e-10:
                    t_min_i = max(t_min_i, (1.0 - ri - xi) / ux)
                    t_max_i = min(t_max_i, (ri - xi) / ux)
                if uy > 1e-10:
                    t_min_i = max(t_min_i, (ri - yi) / uy)
                    t_max_i = min(t_max_i, (1.0 - ri - yi) / uy)
                elif uy < -1e-10:
                    t_min_i = max(t_min_i, (1.0 - ri - yi) / uy)
                    t_max_i = min(t_max_i, (ri - yi) / uy)
                
                # Circle j moves in direction (-ux, -uy)
                t_min_j, t_max_j = -1e10, 1e10
                if -ux > 1e-10:
                    t_min_j = max(t_min_j, (rj - xj) / (-ux))
                    t_max_j = min(t_max_j, (1.0 - rj - xj) / (-ux))
                elif -ux < -1e-10:
                    t_min_j = max(t_min_j, (1.0 - rj - xj) / (-ux))
                    t_max_j = min(t_max_j, (rj - xj) / (-ux))
                if -uy > 1e-10:
                    t_min_j = max(t_min_j, (rj - yj) / (-uy))
                    t_max_j = min(t_max_j, (1.0 - rj - yj) / (-uy))
                elif -uy < -1e-10:
                    t_min_j = max(t_min_j, (1.0 - rj - yj) / (-uy))
                    t_max_j = min(t_max_j, (rj - yj) / (-uy))
                
                # Total separation after pushing by t: dist + 2*t
                # New radii can be at most (dist + 2*t) / 2 each if we want them to touch
                # But we want to maximize r_i + r_j, so set both to (dist + 2*t)/2
                # Actually, we can distribute: r_i_new + r_j_new = dist + 2*t
                # To maximize sum, we just need the max t.
                t_max = min(t_max_i, t_max_j)
                t_max = max(t_max, 0.0)
                
                if t_max > 1e-8:
                    new_dist = dist + 2 * t_max
                    new_ri = new_dist / 2.0
                    new_rj = new_dist / 2.0
                    
                    # Also check wall constraints for new radii
                    new_xi = xi + t_max * ux
                    new_yi = yi + t_max * uy
                    new_xj = xj - t_max * ux
                    new_yj = yj - t_max * uy
                    
                    wall_i = min(new_xi, 1.0 - new_xi, new_yi, 1.0 - new_yi)
                    wall_j = min(new_xj, 1.0 - new_xj, new_yj, 1.0 - new_yj)
                    
                    new_ri = min(new_ri, wall_i)
                    new_rj = min(new_rj, wall_j)
                    
                    # Check against other circles
                    feasible = True
                    for k in range(N):
                        if k == i or k == j:
                            continue
                        d_ik = np.sqrt((new_xi - pos[k,0])**2 + (new_yi - pos[k,1])**2)
                        if d_ik < new_ri + radii[k] - tol:
                            feasible = False
                            break
                        d_jk = np.sqrt((new_xj - pos[k,0])**2 + (new_yj - pos[k,1])**2)
                        if d_jk < new_rj + radii[k] - tol:
                            feasible = False
                            break
                    
                    if feasible and (new_ri + new_rj) > (ri + rj) + tol:
                        pos[i] = [new_xi, new_yi]
                        pos[j] = [new_xj, new_yj]
                        radii[i] = new_ri
                        radii[j] = new_rj
                        improved = True
        
        if not improved:
            break
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Strategy: Multi-start with different initializations + vectorized force annealing
    inits = []
    
    # 1. Best Fit
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    # 2. Lattice
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    # 3. Random (4 variations)
    for i in range(4):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.5))
        
    # Multi-stage optimization with more stages
    penalty_stages = [20.0, 100.0, 500.0, 2000.0, 5000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 40:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        for p_w in penalty_stages:
            if time.time() > deadline - 25:
                break
                
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-9})
                
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
                    
            except Exception:
                pass
            
            # Check validity and update best
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
        # Apply vectorized force-directed annealing
        if time.time() < deadline - 30:
            pos_fa, rad_fa = _vectorized_force_anneal(pos_pp, radii_pp, steps=350, start_temp=0.03, end_temp=0.0004)
            pos_fa, rad_fa = _post_process(pos_fa, rad_fa)
            score_fa = np.sum(rad_fa)
            
            if _is_valid(pos_fa, rad_fa) and score_fa > best_score:
                best_score = score_fa
                best_pos = pos_fa.copy()
                best_radii = rad_fa.copy()
                
        # Local pair optimization for active-set refinement
        if time.time() < deadline - 15:
            pos_lp, rad_lp = _local_pair_optimize(best_pos, best_radii)
            pos_lp, rad_lp = _post_process(pos_lp, rad_lp)
            score_lp = np.sum(rad_lp)
            if _is_valid(pos_lp, rad_lp) and score_lp > best_score:
                best_score = score_lp
                best_pos = pos_lp.copy()
                best_radii = rad_lp.copy()

    # Final aggressive polish: high penalty + local pair optimization
    if time.time() < deadline - 10:
        x_final = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result = minimize(_objective, x_final, method='L-BFGS-B', bounds=bounds,
                              args=(20000.0,),
                              options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12})
            pos_f = result.x[:2*N].reshape(N, 2)
            rad_f = np.maximum(result.x[2*N:], 1e-4)
            pos_f, rad_f = _post_process(pos_f, rad_f)
            score_f = np.sum(rad_f)
            if _is_valid(pos_f, rad_f) and score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = rad_f.copy()
        except Exception:
            pass

    # Final local pair optimization pass
    if time.time() < deadline - 5:
        pos_final, rad_final = _local_pair_optimize(best_pos, best_radii, tol=1e-12, max_iters=200)
        pos_final, rad_final = _post_process(pos_final, rad_final)
        score_final = np.sum(rad_final)
        if _is_valid(pos_final, rad_final) and score_final > best_score:
            best_score = score_final
            best_pos = pos_final.copy()
            best_radii = rad_final.copy()

    # Final validation and post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Fallback if invalid
    if not _is_valid(best_pos, best_radii):
         pos = np.zeros((N, 2))
         radii = np.zeros(N)
         k = 0
         for i in range(5):
             for j in range(5):
                 if k < N-1:
                     pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                     radii[k] = 0.095
                     k += 1
         pos[k] = [0.5, 0.02]
         radii[k] = 0.01
         best_pos, best_radii = pos, radii
         best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
