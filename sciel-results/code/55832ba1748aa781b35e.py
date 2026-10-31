"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii, n_iter=300):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(n_iter):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _objective(x, penalty_weight=100.0):
    """
    Objective: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=500.0, beta=40.0):
    """
    Objective with softplus-based smooth penalties.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    # Smooth boundary
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    
    # softplus for max(0, v)
    boundary_pen = np.sum(np.where(radii - soft_min_d > 20.0/beta, 
                                   radii - soft_min_d, 
                                   np.log1p(np.exp(beta * (radii - soft_min_d))) / beta))

    # Smooth overlap
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist - dist
    overlap_pen = np.sum(np.where(viol_raw[mask] > 20.0/beta, 
                                  viol_raw[mask], 
                                  np.log1p(np.exp(beta * viol_raw[mask])) / beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _slsqp_constraints():
    """Generate constraint functions for SLSQP with hard non-overlap constraints."""
    n = N
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    
    return constraints

def _lp_radius_optimize(pos):
    """
    Solve for maximum radii given fixed positions using Linear Programming.
    """
    n = len(pos)
    c = -np.ones(n)
    A_rows = []
    b_vals = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
            
    for i in range(n):
        w = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(w)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
            return radii
    except Exception:
        pass
    return None

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(8000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
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

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(300):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _physics_simulate(pos, radii, n_steps=5000, dt=0.002, deadline=None):
    """
    Physics-based simulation with repulsive forces.
    Circles exert forces proportional to overlap, and we grow radii when possible.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros((n, 2))
        
        # Boundary forces
        for i in range(n):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - pos[i, 0]) * 100.0
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i])) * 100.0
            if pos[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - pos[i, 1]) * 100.0
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i])) * 100.0
        
        # Overlap repulsion forces
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0.0, sum_r - dist)
        
        # Force magnitude: strong repulsion when overlapping
        force_mag = overlap * 50.0 / (dist + 1e-6)
        
        fx_matrix = force_mag * dx
        fy_matrix = force_mag * dy
        
        forces[:, 0] += np.sum(fx_matrix, axis=1)
        forces[:, 1] += np.sum(fy_matrix, axis=1)
        
        # Update positions with damping
        damping = 0.95
        pos[:, 0] += dt * forces[:, 0] * damping
        pos[:, 1] += dt * forces[:, 1] * damping
        
        # Clamp to unit square
        pos[:, 0] = np.clip(pos[:, 0], 0.001, 0.999)
        pos[:, 1] = np.clip(pos[:, 1], 0.001, 0.999)
        
        # Try to grow radii slightly
        grow_factor = 1.0 + 0.0001
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                max_r = min(max_r, d - radii[j])
            radii[i] = min(radii[i] * grow_factor, max_r * 0.999)
            radii[i] = max(radii[i], 0.001)
        
        # Periodically check and update best
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _gradient_descent(pos, radii, n_steps=3000, lr_init=0.003, lr_decay=0.9995, seed=0, deadline=None):
    """
    Vectorized gradient descent with pairwise force computation.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    lr = lr_init
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces vectorized
        forces = np.zeros((n, 2))
        r_forces = np.zeros(n)
        
        # Boundary forces
        left_viol = np.maximum(0.0, radii - pos[:, 0])
        forces[:, 0] += left_viol * 10.0
        r_forces += left_viol * 5.0
        right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
        forces[:, 0] -= right_viol * 10.0
        r_forces += right_viol * 5.0
        bottom_viol = np.maximum(0.0, radii - pos[:, 1])
        forces[:, 1] += bottom_viol * 10.0
        r_forces += bottom_viol * 5.0
        top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
        forces[:, 1] -= top_viol * 10.0
        r_forces += top_viol * 5.0
        
        # Overlap forces
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        dist = np.sqrt(dist_sq + 1e-12)
        overlap = np.maximum(0.0, sum_r - dist)
        
        force_mag = overlap * 5.0 / (dist + 1e-12)
        
        fx_matrix = force_mag * dx
        fy_matrix = force_mag * dy
        
        forces[:, 0] += np.sum(fx_matrix, axis=1)
        forces[:, 1] += np.sum(fy_matrix, axis=1)
        
        r_forces += 2.0 * np.sum(overlap, axis=1)
        
        # Update
        pos[:, 0] += lr * forces[:, 0]
        pos[:, 1] += lr * forces[:, 1]
        radii -= lr * r_forces
        
        pos[:, 0] = np.clip(pos[:, 0], 0.001, 0.999)
        pos[:, 1] = np.clip(pos[:, 1], 0.001, 0.999)
        radii = np.clip(radii, 0.001, 0.4)
        
        total_force = np.sum(forces ** 2) + np.sum(r_forces ** 2)
        if total_force < 1e-6:
            lr *= 0.9
        else:
            lr *= lr_decay
        
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _simulated_annealing(pos, radii, n_steps=20000, t_init=0.01, t_min=1e-6, cooling=0.9997, seed=0, deadline=None):
    """Simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    t = t_init
    step = 0
    while step < n_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.randint(n)
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        
        scale = 0.01 * t / t_init + 0.001
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        dr = rng.normal(0, scale * 0.5)
        
        new_x = np.clip(old_x + dx, 0.001, 0.999)
        new_y = np.clip(old_y + dy, 0.001, 0.999)
        new_r = np.clip(old_r + dr, 0.001, 0.4)
        
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        # Compute energy change
        old_bv = np.maximum(0.0, old_r - old_x) + np.maximum(0.0, old_x - (1.0 - old_r)) + \
                 np.maximum(0.0, old_r - old_y) + np.maximum(0.0, old_y - (1.0 - old_r))
        old_overlap = 0.0
        for j in range(n):
            if j == i:
                continue
            dx2 = old_x - pos[j, 0]
            dy2 = old_y - pos[j, 1]
            dist_sq2 = dx2 * dx2 + dy2 * dy2
            min_dist2 = old_r + radii[j]
            viol2 = max(0.0, min_dist2 * min_dist2 - dist_sq2)
            old_overlap += viol2 * viol2
        old_contrib = 50.0 * (old_bv * old_bv + old_overlap)
        
        new_bv = np.maximum(0.0, new_r - new_x) + np.maximum(0.0, new_x - (1.0 - new_r)) + \
                 np.maximum(0.0, new_r - new_y) + np.maximum(0.0, new_y - (1.0 - new_r))
        new_overlap = 0.0
        for j in range(n):
            if j == i:
                continue
            dx3 = new_x - pos[j, 0]
            dy3 = new_y - pos[j, 1]
            dist_sq3 = dx3 * dx3 + dy3 * dy3
            min_dist3 = new_r + radii[j]
            viol3 = max(0.0, min_dist3 * min_dist3 - dist_sq3)
            new_overlap += viol3 * viol3
        new_contrib = 50.0 * (new_bv * new_bv + new_overlap)
        
        delta_energy = new_contrib - old_contrib
        delta_score = new_r - old_r
        delta_total = delta_energy - delta_score
        
        if delta_total < 0 or rng.random() < np.exp(-delta_total / max(t, 1e-10)):
            pass
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        t *= cooling
        t = max(t, t_min)
        step += 1
        
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP, LP, and local search."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # SLSQP with hard constraints
    if time.time() < deadline - 15:
        try:
            cons = _slsqp_constraints()
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.success:
                x_new = result.x
                pos_slq = x_new[:2*N].reshape(N, 2)
                rad_slq = np.maximum(x_new[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_slq, rad_slq)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # LP-based local search
    if time.time() < deadline - 5:
        for iter in range(150):
            if time.time() > deadline - 5:
                break
            
            new_radii = _lp_radius_optimize(best_pos)
            if new_radii is None:
                break
                
            new_radii_safe = np.maximum(new_radii * 0.99999, 1e-5)
            
            if _is_valid(best_pos, new_radii_safe):
                score = np.sum(new_radii_safe)
                if score > best_score:
                    best_score = score
                    best_radii = new_radii_safe.copy()
                else:
                    break
            else:
                break
            
            # Identify tight constraints and apply forces
            dx = best_pos[:, 0, np.newaxis] - best_pos[np.newaxis, :, 0]
            dy = best_pos[:, 1, np.newaxis] - best_pos[np.newaxis, :, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            
            forces = np.zeros((N, 2))
            for i in range(N):
                for j in range(i+1, N):
                    sum_r = best_radii[i] + best_radii[j]
                    d = dists[i, j]
                    if d > 0 and sum_r > d * 0.95:
                        strength = (sum_r / d) ** 4
                        fx = strength * dx[i, j]
                        fy = strength * dy[i, j]
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                        
            for i in range(N):
                if best_pos[i, 0] < best_radii[i] * 1.1:
                    forces[i, 0] += 1.0
                if best_pos[i, 0] > 1.0 - best_radii[i] * 1.1:
                    forces[i, 0] -= 1.0
                if best_pos[i, 1] < best_radii[i] * 1.1:
                    forces[i, 1] += 1.0
                if best_pos[i, 1] > 1.0 - best_radii[i] * 1.1:
                    forces[i, 1] -= 1.0
            
            lr = 0.005
            best_pos += lr * forces
            best_pos = np.clip(best_pos, 0.01, 0.99)
            
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [50.0, 200.0, 1000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective with escalating penalties
        for p_w in penalty_stages:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Smooth objective
        if time.time() < deadline - 15:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 10:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 50.0),
                                      options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10})
                    if result.fun < _objective_smooth(current_x, p_w, 50.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Physics simulation
        if time.time() < deadline - 12:
            try:
                pos_phys, rad_phys = _physics_simulate(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=3000, dt=0.002,
                    deadline=deadline - 10
                )
                pos_pp, radii_pp = _post_process(pos_phys, rad_phys)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 4: Gradient descent
        if time.time() < deadline - 10:
            try:
                pos_gd, rad_gd = _gradient_descent(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=3000, lr_init=0.003, lr_decay=0.9995,
                    seed=idx * 100, deadline=deadline - 8
                )
                pos_pp, radii_pp = _post_process(pos_gd, rad_gd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 5: Simulated annealing
        if time.time() < deadline - 10:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=15000, t_init=0.008, t_min=1e-6, cooling=0.9998,
                    seed=idx * 200 + 50, deadline=deadline - 8
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 8:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Second polish attempt with perturbation
    if time.time() < deadline - 5:
        try:
            rng = np.random.RandomState(12345)
            perturbed_pos = best_pos + rng.normal(0, 0.003, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            pos_pol2, rad_pol2 = _polish(perturbed_pos, best_radii.copy(), deadline)
            pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
            score2 = np.sum(radii_pp2)
            if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = rad_pol2.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
