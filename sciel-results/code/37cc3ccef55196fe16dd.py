"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, penalty_weight=5000.0):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    radii = np.maximum(radii, 1e-4)
    
    penalty = 0.0
    # Boundary penalties (smooth)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    penalty += np.sum(left_viol**2) + np.sum(right_viol**2) + np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Pairwise overlap penalties (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap = np.maximum(0.0, min_dist_sq - dist_sq)
    penalty += np.sum(overlap[mask] ** 2 / (4.0 * min_dist_sq[mask]))
    
    return -np.sum(radii) + penalty_weight * penalty


def _initial_hexagonal():
    """Create a hexagonal packing pattern - typically optimal for circle packing."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5 rows of 5, with staggered rows of 4
    # Row pattern: 5, 4, 5, 4, 4 (to get 26)
    # Actually let's do: 5, 4, 5, 4, 4 = 22... need 26
    # Let's do: 5, 5, 5, 5, 5, 1 = 26 but that's not hexagonal
    # Better: 4, 5, 4, 5, 4, 4 = 26
    # Or: 5, 4, 5, 4, 5, 3 = 26
    
    # Let's use a proper hexagonal arrangement
    # 5 circles in rows 0, 2, 4 and 4 in rows 1, 3, and 1 extra
    # Actually: rows of 5,4,5,4,4,4 = 26
    
    r = 0.092
    y_spacing = r * np.sqrt(3)
    
    # 6 rows: 5, 4, 5, 4, 4, 4
    row_sizes = [5, 4, 5, 4, 4, 4]
    idx = 0
    for row_idx, size in enumerate(row_sizes):
        y = (row_idx + 0.5) * (1.0 / 6.0)
        if row_idx % 2 == 0:
            x_start = (1.0 - (size - 1) * 2 * r) / 2.0
        else:
            x_start = (1.0 - (size - 1) * 2 * r) / 2.0 + r
        
        for i in range(size):
            if idx < N:
                pos[idx, 0] = x_start + i * 2 * r
                pos[idx, 1] = y
                radii[idx] = r
                idx += 1
    
    # Ensure we have exactly N circles
    while idx < N:
        pos[idx, 0] = 0.5
        pos[idx, 1] = 0.5
        radii[idx] = r * 0.5
        idx += 1
    
    return pos[:N], radii[:N]


def _initial_grid():
    """Create a 5x5 grid with one extra circle."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.095
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = r
    
    # 26th circle in a gap
    pos[25, 0] = 0.5
    pos[25, 1] = 0.05
    radii[25] = 0.04
    
    return pos, radii


def _initial_corner_heavy():
    """Place circles heavily in corners to exploit boundary space."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corner circles
    corner_positions = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corner_positions):
        pos[i, 0] = cx
        pos[i, 1] = cy
        radii[i] = 0.11
    
    # 8 edge circles
    edge_positions = [
        (0.5, 0.07), (0.5, 0.93), (0.07, 0.5), (0.93, 0.5),
        (0.3, 0.07), (0.7, 0.07), (0.3, 0.93), (0.7, 0.93)
    ]
    for i, (cx, cy) in enumerate(edge_positions):
        idx = 4 + i
        pos[idx, 0] = cx
        pos[idx, 1] = cy
        radii[idx] = 0.085
    
    # 14 interior circles
    rng = np.random.RandomState(77)
    for i in range(14):
        idx = 12 + i
        pos[idx, 0] = rng.uniform(0.25, 0.75)
        pos[idx, 1] = rng.uniform(0.25, 0.75)
        radii[idx] = 0.075
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.06, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity, gently and proportionally."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(800):
        valid = True
        
        # Fix boundary violations
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r * 0.9999:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # Fix pairwise overlaps - shrink proportionally to radius
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-12:
                    # Shrink both proportionally to their radii
                    total_r = radii[i] + radii[j]
                    if total_r < 1e-12:
                        continue
                    # Scale factor to make them just touch
                    scale = dist / min_dist * 0.9999
                    # Shrink proportionally: smaller circle shrinks more in absolute terms
                    # but proportionally both shrink by the same factor
                    radii[i] = max(radii[i] * scale, 1e-5)
                    radii[j] = max(radii[j] * scale, 1e-5)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _grow_and_push(pos, radii, deadline, n_iter=5000):
    """
    Grow radii and push circles apart using force-directed dynamics.
    Adaptive step sizes and weights toward smaller circles.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(999)
    
    t = 0.03
    cool_rate = (0.0001 / t) ** (1.0 / max(n_iter, 1))
    
    for iteration in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        forces = np.zeros((N, 2))
        
        # Vectorized pairwise forces
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, 1e12)
        
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap forces (repulsive) - stronger for smaller circles
        overlap = np.maximum(0.0, min_dist - dist)
        # Weight: smaller circles get pushed more
        weight = np.where(radii[:, np.newaxis] < radii[np.newaxis, :], 
                         1.5, 0.5)
        f_mag = 50.0 * overlap * weight / np.maximum(dist, 1e-12)
        
        fx = f_mag * dx
        fy = f_mag * dy
        
        forces[:, 0] = np.sum(fx, axis=1)
        forces[:, 1] = np.sum(fy, axis=1)
        
        # Boundary forces
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += 50.0 * (radii[i] - pos[i, 0])
            elif pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= 50.0 * (pos[i, 0] - (1.0 - radii[i]))
            
            if pos[i, 1] < radii[i]:
                forces[i, 1] += 50.0 * (radii[i] - pos[i, 1])
            elif pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= 50.0 * (pos[i, 1] - (1.0 - radii[i]))
        
        step_size = t * 0.3
        noise = rng.normal(0, t * 0.01, size=(N, 2))
        pos += forces * step_size + noise
        pos = np.clip(pos, 0.01, 0.99)
        
        # Grow radii where possible - adaptive growth
        for i in range(N):
            min_clear = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if j == i:
                    continue
                dx_ij = pos[i, 0] - pos[j, 0]
                dy_ij = pos[i, 1] - pos[j, 1]
                dist_ij = np.sqrt(dx_ij * dx_ij + dy_ij * dy_ij)
                min_clear = min(min_clear, dist_ij - radii[j])
            
            # Smaller circles grow faster
            growth_rate = t * 0.005 * (0.1 / max(radii[i], 0.01))
            if min_clear > radii[i]:
                new_r = min(radii[i] + growth_rate, min_clear * 0.98)
                if new_r > radii[i]:
                    radii[i] = new_r
            elif min_clear < radii[i] * 0.95:
                radii[i] = max(min_clear * 0.98, 0.01)
        
        current_score = np.sum(radii)
        
        # Quick validity check
        valid = True
        for i in range(N):
            if pos[i, 0] < radii[i] - 1e-9 or pos[i, 0] > 1.0 - radii[i] + 1e-9:
                valid = False
                break
            if pos[i, 1] < radii[i] - 1e-9 or pos[i, 1] > 1.0 - radii[i] + 1e-9:
                valid = False
                break
        if valid:
            dx2 = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
            dy2 = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
            dist2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
            min_d2 = radii[:, np.newaxis] + radii[np.newaxis, :]
            if np.any(np.triu(dist2, k=1) < np.triu(min_d2, k=1) - 1e-8):
                valid = False
        
        if valid and current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
            best_radii = radii.copy()
        elif not valid:
            pos = best_pos.copy()
            radii = best_radii.copy()
        
        t *= cool_rate
    
    pos_pp, radii_pp = _post_process(best_pos, best_radii)
    score = np.sum(radii_pp)
    
    if _is_valid(pos_pp, radii_pp) and score > 0:
        return pos_pp, radii_pp, score
    
    return best_pos, best_radii, best_score


def _differential_evolution_run(pos_init, radii_init, deadline, pop_size=40, n_iter=200):
    """
    Differential evolution for global optimization of circle positions and radii.
    """
    best_pos = pos_init.copy()
    best_radii = radii_init.copy()
    best_score = np.sum(radii_init)
    
    rng = np.random.RandomState(555)
    
    # Create population
    population = np.zeros((pop_size, 2 * N + N))
    for i in range(pop_size):
        population[i, :2*N] = pos_init.flatten() + rng.normal(0, 0.05, size=2*N)
        population[i, 2*N:] = radii_init + rng.normal(0, 0.02, size=N)
        population[i, :2*N] = np.clip(population[i, :2*N], 0.05, 0.95).reshape(-1)
        population[i, 2*N:] = np.clip(population[i, 2*N:], 0.03, 0.25)
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        x = population[i]
        pos = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            fitness[i] = np.sum(radii_pp)
        else:
            fitness[i] = 0.0
    
    # Track best
    best_idx = np.argmax(fitness)
    if fitness[best_idx] > best_score:
        best_score = fitness[best_idx]
        best_pos = population[best_idx, :2*N].reshape(N, 2).copy()
        best_radii = population[best_idx, 2*N:].copy()
    
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    for gen in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        for i in range(pop_size):
            if time.time() > deadline - 3:
                    break
            
            # Select three distinct random indices
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            while len(set([r1, r2, r3])) < 3:
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            
            # Create trial vector
            trial = population[r1].copy()
            for j in range(len(trial)):
                if rng.random() < CR or j == rng.randint(len(trial)):
                    trial[j] = population[r1, j] + F * (population[r2, j] - population[r3, j])
            
            # Clip to bounds
            trial[:2*N] = np.clip(trial[:2*N], 0.05, 0.95)
            trial[2*N:] = np.clip(trial[2*N:], 0.03, 0.25)
            
            # Evaluate trial
            pos = trial[:2*N].reshape(N, 2)
            radii = trial[2*N:]
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                trial_fitness = np.sum(radii_pp)
            else:
                trial_fitness = 0.0
            
            # Selection
            if trial_fitness > fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness > best_score:
                    best_score = trial_fitness
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii, best_score


def _polish(pos, radii, deadline):
    """High-precision local search with multiple passes and tight tolerances."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.4))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple L-BFGS-B passes with decreasing tolerances
    tolerances = [(1e-12, 1e-8), (1e-14, 1e-10), (1e-16, 1e-12), (1e-18, 1e-14)]
    for ftol, gtol in tolerances:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 5000, 'ftol': ftol, 'gtol': gtol})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # SLSQP pass with higher penalty
    if time.time() < deadline - 4:
        try:
            result = minimize(lambda x: _objective(x, 10000.0), x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14})
            
            if result.fun < _objective(x0, 10000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Final Powell pass for fine-tuning
    if time.time() < deadline - 2:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='Powell', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-15})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Third polish pass with even tighter settings
    if time.time() < deadline - 1.5:
        try:
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            result = minimize(lambda x: _objective(x, 20000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            
            if result.fun < _objective(x0, 20000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-5)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration with time-aware allocation."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.3))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # L-BFGS-B first (fast gradient-based)
    if time.time() < deadline - 15:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Powell (derivative-free, good for fine-tuning)
    if time.time() < deadline - 12:
        try:
            result = minimize(lambda x: _objective(x, 5000.0), x0, method='Powell', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-13})
            
            if result.fun < _objective(x0, 5000.0) - 1e-15:
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Grow and push pass
    if time.time() < deadline - 20:
        try:
            gp_pos, gp_radii, gp_score = _grow_and_push(best_pos, best_radii, deadline, n_iter=5000)
            if gp_score > best_score + 1e-12:
                best_score = gp_score
                best_pos = gp_pos.copy()
                best_radii = gp_radii.copy()
        except Exception:
            pass
    
    # Differential evolution pass
    if time.time() < deadline - 30:
        try:
            de_pos, de_radii, de_score = _differential_evolution_run(
                best_pos, best_radii, deadline, pop_size=30, n_iter=150)
            if de_score > best_score + 1e-12:
                best_score = de_score
                best_pos = de_pos.copy()
                best_radii = de_radii.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 4:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
        
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Build initial configurations
    init_configs = []
    
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_grid()
    init_configs.append((pos, radii))
    
    pos, radii = _initial_corner_heavy()
    init_configs.append((pos, radii))
    
    for seed in [42, 123, 456, 789]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii))
    
    # Perturbed hexagonal
    rng = np.random.RandomState(99)
    pos, radii = _initial_hexagonal()
    pos_pert = pos + rng.normal(0, 0.03, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed grid
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.03, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Perturbed corner heavy
    pos, radii = _initial_corner_heavy()
    pos_pert = pos + rng.normal(0, 0.03, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii))
    
    # Run optimization from each initial config with time-aware allocation
    # Prioritize hexagonal (typically best for circle packing)
    config_order = [0, 5, 1, 6, 2, 7, 3, 4, 8]
    for idx in config_order:
        if time.time() > deadline - 25:
            break
        if idx >= len(init_configs):
            continue
        
        pos, radii = init_configs[idx]
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score + 1e-12:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # Fallback if nothing valid found
    if best_pos is None or best_score < 0:
        pos, radii = _initial_grid()
        pos_pp, radii_pp = _post_process(pos, radii)
        best_pos = pos_pp
        best_radii = radii_pp
        best_score = np.sum(radii_pp)
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        best_pos, best_radii, best_score = _polish(best_pos, best_radii, deadline)
    
    # Final post-processing to ensure validity
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
