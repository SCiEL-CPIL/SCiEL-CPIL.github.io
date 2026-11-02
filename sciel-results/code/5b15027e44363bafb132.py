"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    # Check bounds
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
        
    # Check pairwise non-overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    
    triu = np.triu_indices(n, 1)
    overlaps = sum_r[triu] - dist[triu]
    
    if np.any(overlaps > 1e-7):
        return False
        
    return True

def _calculate_penalty(xyz):
    """Calculate total penalty for invalidity (overlaps and boundary violations)."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    penalty = 0.0
    
    lb_x = np.maximum(0, r - x)
    penalty += np.sum(lb_x ** 2)
    ub_x = np.maximum(0, x + r - 1)
    penalty += np.sum(ub_x ** 2)
    lb_y = np.maximum(0, r - y)
    penalty += np.sum(lb_y ** 2)
    ub_y = np.maximum(0, y + r - 1)
    penalty += np.sum(ub_y ** 2)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    
    triu = np.triu_indices(n, 1)
    overlaps = np.maximum(0, sum_r[triu] - dist[triu])
    penalty += np.sum(overlaps ** 2)
    
    return penalty

def _score(xyz):
    """
    Objective function for Simulated Annealing.
    Score = Sum(r) - Weight * Penalty
    """
    penalty = _calculate_penalty(xyz)
    sum_r = np.sum(xyz[:, 2])
    weight = 10.0 
    return sum_r - weight * penalty

def _initial_grid():
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_hex():
    """Create a hexagonal initial configuration."""
    positions = []
    side = 5
    for j in range(side):
        for i in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.5 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.1])
    return np.array(positions[:N])

def _initial_random(seed):
    """Create a random initial configuration."""
    np.random.seed(seed)
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.05, 0.12, N)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    return np.column_stack([x, y, r])

def _simulated_annealing(xyz_init, time_budget):
    """
    Run Simulated Annealing on the given initial configuration.
    """
    start_time = time.time()
    xyz = xyz_init.copy()
    
    T = 1.0
    alpha = 0.999
    min_T = 1e-5
    max_steps = 100000
    
    best_xyz = xyz.copy()
    best_score = _score(xyz)
    best_valid_score = -np.inf
    best_valid_xyz = xyz.copy() if _validate(xyz) else None
    
    sigma_pos = 0.05
    sigma_r = 0.02
    
    n_accepted = 0
    n_rejected = 0
    window = 1000
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        xyz_new = xyz.copy()
        
        move_type = np.random.random()
        
        if move_type < 0.7:
            idx = np.random.randint(N)
            dx = np.random.normal(0, sigma_pos)
            dy = np.random.normal(0, sigma_pos)
            xyz_new[idx, 0] += dx
            xyz_new[idx, 1] += dy
            xyz_new[idx, 0] = np.clip(xyz_new[idx, 0], 0, 1)
            xyz_new[idx, 1] = np.clip(xyz_new[idx, 1], 0, 1)
        elif move_type < 0.9:
            idx = np.random.randint(N)
            dr = np.random.normal(0, sigma_r)
            xyz_new[idx, 2] += dr
            xyz_new[idx, 2] = np.clip(xyz_new[idx, 2], 1e-4, 0.5)
        else:
            indices = np.random.choice(N, size=3, replace=False)
            for idx in indices:
                dx = np.random.normal(0, sigma_pos * 0.5)
                dy = np.random.normal(0, sigma_pos * 0.5)
                dr = np.random.normal(0, sigma_r * 0.5)
                xyz_new[idx, 0] += dx
                xyz_new[idx, 1] += dy
                xyz_new[idx, 2] += dr
                xyz_new[idx, 0] = np.clip(xyz_new[idx, 0], 0, 1)
                xyz_new[idx, 1] = np.clip(xyz_new[idx, 1], 0, 1)
                xyz_new[idx, 2] = np.clip(xyz_new[idx, 2], 1e-4, 0.5)
        
        score_new = _score(xyz_new)
        score_old = _score(xyz)
        
        delta_score = score_new - score_old
        
        if delta_score > 0:
            xyz = xyz_new
            n_accepted += 1
        else:
            if T > 0:
                prob = np.exp(delta_score / T)
                if np.random.random() < prob:
                    xyz = xyz_new
                    n_accepted += 1
                else:
                    n_rejected += 1
            else:
                n_rejected += 1
        
        if score_new > best_score:
            best_score = score_new
            best_xyz = xyz_new.copy()
            
        if step % 10 == 0 or score_new > best_valid_score + 0.01:
             if _validate(xyz_new):
                 val_score = np.sum(xyz_new[:, 2])
                 if val_score > best_valid_score:
                     best_valid_score = val_score
                     best_valid_xyz = xyz_new.copy()
        
        if step % window == 0 and step > 0:
            accept_rate = n_accepted / (n_accepted + n_rejected + 1e-9)
            if accept_rate > 0.5:
                sigma_pos *= 1.1
                sigma_r *= 1.1
            elif accept_rate < 0.2:
                sigma_pos *= 0.9
                sigma_r *= 0.9
            n_accepted = 0
            n_rejected = 0
        
        T *= alpha
        if T < min_T:
            T = min_T
    
    if _validate(best_xyz):
        return best_xyz
    elif best_valid_xyz is not None:
        return best_valid_xyz
    else:
        xyz_fix = best_xyz.copy()
        for _ in range(100):
            if _validate(xyz_fix):
                return xyz_fix
            xyz_fix[:, 2] *= 0.95
            xyz_fix[:, 0] = np.clip(xyz_fix[:, 0], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
            xyz_fix[:, 1] = np.clip(xyz_fix[:, 1], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
        return _initial_grid()

def _lbfgs_polish(xyz, time_budget):
    """
    Use L-BFGS-B to polish a valid configuration.
    """
    start_time = time.time()
    params = xyz.flatten()
    
    def obj(p):
        n = N
        x = p[0::3]
        y = p[1::3]
        r = p[2::3]
        
        penalty = 0.0
        lb_x = np.maximum(0, r - x)
        penalty += np.sum(lb_x ** 2)
        ub_x = np.maximum(0, x + r - 1)
        penalty += np.sum(ub_x ** 2)
        lb_y = np.maximum(0, r - y)
        penalty += np.sum(lb_y ** 2)
        ub_y = np.maximum(0, y + r - 1)
        penalty += np.sum(ub_y ** 2)
        neg_r = np.maximum(0, -r)
        penalty += np.sum(neg_r ** 2)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        sum_r = r[:, None] + r[None, :]
        overlap = np.maximum(0, sum_r - dist)
        triu = np.triu_indices(n, 1)
        penalty += np.sum(overlap[triu] ** 2)
        
        return -np.sum(r) + 100.0 * penalty
    
    bounds = [(0, 1), (0, 1), (1e-4, 0.5)] * N
    
    try:
        result = minimize(obj, params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        xyz_res = result.x.reshape(N, 3)
        if _validate(xyz_res):
            if np.sum(xyz_res[:, 2]) > np.sum(xyz[:, 2]):
                return xyz_res
    except Exception:
        pass
        
    return xyz

def _pbd_polish(xyz, time_budget):
    """
    Position Based Dynamics (PBD) style polish.
    Iteratively resolves overlaps and pushes circles outward to maximize radii.
    """
    start_time = time.time()
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Parameters
    max_iters = 2000
    damping = 0.9
    growth_rate = 0.005
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # 1. Resolve overlaps (Gauss-Seidel)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        dx = 0.1
                        dy = 0.1
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    dx_unit = dx / dist
                    dy_unit = dy / dist
                    overlap = min_dist - dist
                    push = overlap * 0.5 * damping
                    
                    x[i] -= dx_unit * push
                    y[i] -= dy_unit * push
                    x[j] += dx_unit * push
                    y[j] += dy_unit * push
        
        # 2. Enforce boundaries
        for i in range(N):
            # If circle goes out of bounds, push it back and shrink radius if necessary
            if x[i] - r[i] < 0:
                x[i] = r[i]
            if x[i] + r[i] > 1:
                x[i] = 1 - r[i]
            if y[i] - r[i] < 0:
                y[i] = r[i]
            if y[i] + r[i] > 1:
                y[i] = 1 - r[i]
            
            # Ensure position is strictly inside [0,1]
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            
            # Max possible radius based on walls
            max_r_wall = min(x[i], y[i], 1-x[i], 1-y[i])
            if r[i] > max_r_wall:
                r[i] = max_r_wall
            
            # Max possible radius based on neighbors
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r_neigh = dist - r[j]
                if r[i] > max_r_neigh:
                    r[i] = max_r_neigh
            
            r[i] = max(0, r[i])
        
        # 3. Grow radii slightly (explore larger radii)
        # This is the "pressure" that tries to expand circles
        for i in range(N):
            # Check if we can grow
            can_grow = True
            # Walls
            if x[i] - r[i] < 0 or x[i] + r[i] > 1 or y[i] - r[i] < 0 or y[i] + r[i] > 1:
                can_grow = False
            # Neighbors
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] + growth_rate:
                    can_grow = False
                    break
            
            if can_grow:
                r[i] += growth_rate
            else:
                # Try to move towards empty space? 
                # Simple heuristic: move towards center of mass of free space? 
                # Or just rely on the overlap resolution to push things around.
                # Let's add a small random jitter to escape local minima in PBD
                if np.random.random() < 0.1:
                    x[i] += np.random.normal(0, 0.01)
                    y[i] += np.random.normal(0, 0.01)
                    x[i] = np.clip(x[i], 0, 1)
                    y[i] = np.clip(y[i], 0, 1)
        
        # 4. Re-enforce validity after growth
        # If growth caused overlap, the next iteration's step 1 will fix it,
        # but we might have invalid intermediate states.
        # Let's just check validity at the end of each iteration? 
        # No, PBD guarantees validity if we do it right.
        # But the "grow" step might make it invalid.
        # So we should only grow if it remains valid? 
        # The check above `if dist < r[i] + r[j] + growth_rate` ensures we don't overlap.
        # But it doesn't check walls for the NEW radius.
        # Let's re-check walls for new radius.
        # Actually, the logic `if x[i] - r[i] < 0 ... can_grow = False` checks CURRENT radius.
        # If we grow, we need to check if NEW radius fits.
        # Let's refine the growth check.
        
        # Actually, simpler approach: 
        # 1. Resolve overlaps.
        # 2. Clamp to walls.
        # 3. Calculate max possible radius for each circle.
        # 4. Set radius to max possible radius * (1 - small_epsilon) or similar?
        # No, that converges to local optimum where all circles are touching.
        # That's actually what we want!
        
        # Let's restart the logic for PBD polish to be more robust.
        
        # Revised PBD Loop:
        # 1. Resolve overlaps (push apart).
        # 2. Clamp positions to [0,1].
        # 3. For each circle, calculate max radius allowed by walls and neighbors.
        # 4. Set radius to that max value.
        # This converges to a "tight" packing.
        
        pass # The loop above was a bit messy. Let's implement the cleaner version below.

    # Clean PBD Implementation
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
            
        # 1. Resolve overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        dx = 0.1
                        dy = 0.1
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    dx_unit = dx / dist
                    dy_unit = dy / dist
                    overlap = min_dist - dist
                    push = overlap * 0.5
                    
                    x[i] -= dx_unit * push
                    y[i] -= dy_unit * push
                    x[j] += dx_unit * push
                    y[j] += dy_unit * push
        
        # 2. Clamp to walls
        for i in range(N):
            x[i] = np.clip(x[i], 0, 1)
            y[i] = np.clip(y[i], 0, 1)
            
            # Shrink radius if it violates walls
            max_r_wall = min(x[i], y[i], 1-x[i], 1-y[i])
            if r[i] > max_r_wall:
                r[i] = max_r_wall
        
        # 3. Maximize radii given positions
        for i in range(N):
            # Start with wall constraint
            max_r = min(x[i], y[i], 1-x[i], 1-y[i])
            
            # Check neighbors
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # r_i <= dist - r_j
                max_r = min(max_r, dist - r[j])
            
            r[i] = max(0, max_r)
            
            # Re-clamp position if radius change affected wall constraint? 
            # No, max_r was calculated based on position.
            # But if r[i] was shrunk due to walls, position is still valid.
            # If r[i] was shrunk due to neighbors, position is still valid.
            
    xyz_out = np.column_stack([x, y, r])
    
    # Final validation and cleanup
    if _validate(xyz_out):
        if np.sum(xyz_out[:, 2]) > np.sum(xyz[:, 2]):
            return xyz_out
    
    return xyz

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of radii.
    """
    global_start = time.time()
    deadline = 145.0  # Safety margin
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(456),
        _initial_random(789),
        _initial_random(101),
        _initial_random(2024)
    ]
    
    for i, xyz0 in enumerate(inits):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        time_budget = min(12.0, remaining - 2.0)
        if time_budget < 2.0:
            break
            
        xyz_opt = _simulated_annealing(xyz0, time_budget)
        score = np.sum(xyz_opt[:, 2])
        
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Polish with L-BFGS
    elapsed = time.time() - global_start
    if elapsed < deadline - 10.0:
        remaining = deadline - elapsed - 5.0
        if remaining > 2.0:
            xyz_polished = _lbfgs_polish(best_xyz, remaining)
            score = np.sum(xyz_polished[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_polished

    # Polish with PBD
    elapsed = time.time() - global_start
    if elapsed < deadline - 5.0:
        remaining = deadline - elapsed - 1.0
        if remaining > 1.0:
            xyz_pbd = _pbd_polish(best_xyz, remaining)
            score = np.sum(xyz_pbd[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_pbd

    # Final validation
    if not _validate(best_xyz):
        xyz_fix = best_xyz.copy()
        for _ in range(300):
            if _validate(xyz_fix):
                break
            xyz_fix[:, 2] *= 0.95
            xyz_fix[:, 0] = np.clip(xyz_fix[:, 0], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
            xyz_fix[:, 1] = np.clip(xyz_fix[:, 1], xyz_fix[:, 2], 1 - xyz_fix[:, 2])
        if _validate(xyz_fix):
            best_xyz = xyz_fix
        else:
            best_xyz = _initial_grid()
            
    return best_xyz

# EVOLVE-BLOCK-END
