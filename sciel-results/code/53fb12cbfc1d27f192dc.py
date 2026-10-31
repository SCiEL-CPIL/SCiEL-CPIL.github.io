"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective: minimize negative sum of radii + penalty for violations.
    x = [x_1..x_N, y_1..y_N, r_1..r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    # 1. Boundary violations (smooth squared penalty)
    left_viol = np.maximum(0.0, radii - pos[:, 0])
    right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bottom_viol = np.maximum(0.0, radii - pos[:, 1])
    top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_penalty = (np.sum(left_viol**2) + np.sum(right_viol**2) +
                        np.sum(bottom_viol**2) + np.sum(top_viol**2))

    # 2. Pairwise overlap violations (vectorized, smoother)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy

    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    violation = np.maximum(0.0, min_dist_sq - dist_sq)
    denom = min_dist_sq + 1e-8
    smooth_overlap = (violation ** 2) / denom

    overlap_penalty = np.sum(smooth_overlap[mask])

    total_penalty = boundary_penalty + overlap_penalty
    return -np.sum(radii) + penalty_weight * total_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(100):
        valid = True

        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.999, 1e-5)
                valid = False

        # 2. Overlap fix
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False

        if valid:
            break

    return pos, radii

def _initial_best_fit():
    """Greedy best-fit: place a large central circle, then fill."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i)

        for _ in range(1500):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r and max_r_circ > 0.01:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if not found or best_r < 0.01:
            best_x, best_y, best_r = 0.5, 0.5, 0.01

        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.99

    return pos, radii

def _initial_lattice():
    """5x5 lattice init (25 cells) + one extra; gives a strong dense start."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.095
                k += 1
    # N=26: k reaches 25, so the 26th stays at origin with radius 0 -> post_process fixes
    return pos, radii

def _initial_random(seed):
    """Random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _polish(x0, bounds, deadline, penalty_weight=500.0):
    """L-BFGS-B then high-precision Powell polish. Returns best (pos, radii, score)."""
    best_pos = x0[:2*N].reshape(N, 2).copy()
    best_radii = np.maximum(x0[2*N:], 1e-4).copy()
    best_score = np.sum(best_radii)
    best_pos, best_radii = _post_process(best_pos, best_radii)

    def _try(x_opt):
        nonlocal best_pos, best_radii, best_score
        p = x_opt[:2*N].reshape(N, 2)
        r = np.maximum(x_opt[2*N:], 1e-4)
        p, r = _post_process(p, r)
        if _is_valid(p, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p.copy()
                best_radii = r.copy()

    # L-BFGS-B
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                       args=(penalty_weight,),
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() < deadline:
            _try(res.x)
    except Exception:
        pass

    # High-precision Powell polish on current best
    if time.time() < deadline - 5:
        x_cur = np.concatenate([best_pos.flatten(), best_radii])
        try:
            res = minimize(_objective, x_cur, method='Powell', bounds=bounds,
                           args=(penalty_weight,),
                           options={'maxiter': 2000, 'ftol': 1e-14, 'xtol': 1e-12})
            _try(res.x)
        except Exception:
            pass

    return best_pos, best_radii, best_score

def _force_directed_anneal(pos, radii, deadline, n_iter=5000, seed=42):
    """
    Vectorized force-directed annealing:
    - Repulsion between overlapping circles
    - Attraction toward center of available space
    - Boundary forces
    Uses simulated annealing schedule for temperature.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    T0 = 0.1
    T_end = 0.001
    n_steps = n_iter
    
    best_score = np.sum(radii) if _is_valid(pos, radii) else -1.0
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(n_steps):
        if time.time() > deadline - 2:
            break
        
        # Temperature schedule
        t = step / max(n_steps - 1, 1)
        T = T0 * (T_end / T0) ** t
        
        # Compute forces (vectorized)
        # 1. Pairwise repulsion for overlaps
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0.0, min_dist - dist)
        
        # Force direction: push apart
        force_x = np.zeros((N, N))
        force_y = np.zeros((N, N))
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        # Only apply force where there's overlap
        overlap_mask = mask & (overlap > 0)
        
        for i in range(N):
            for j in range(i + 1, N):
                if overlap[i, j] > 0:
                    d = dist[i, j] + 1e-12
                    fx = (pos[i, 0] - pos[j, 0]) / d * overlap[i, j] * 2.0
                    fy = (pos[i, 1] - pos[j, 1]) / d * overlap[i, j] * 2.0
                    force_x[i, j] += fx
                    force_x[j, i] -= fx
                    force_y[i, j] += fy
                    force_y[j, i] -= fy
        
        # 2. Boundary forces
        # Push circles that are too close to walls back
        boundary_fx = np.zeros(N)
        boundary_fy = np.zeros(N)
        
        # Left wall
        left_dist = pos[:, 0] - radii
        left_viol = np.maximum(0.0, -left_dist)
        boundary_fx += left_viol * 2.0
        
        # Right wall
        right_dist = (1.0 - pos[:, 0]) - radii
        right_viol = np.maximum(0.0, -right_dist)
        boundary_fx -= right_viol * 2.0
        
        # Bottom wall
        bottom_dist = pos[:, 1] - radii
        bottom_viol = np.maximum(0.0, -bottom_dist)
        boundary_fy += bottom_viol * 2.0
        
        # Top wall
        top_dist = (1.0 - pos[:, 1]) - radii
        top_viol = np.maximum(0.0, -top_dist)
        boundary_fy -= top_viol * 2.0
        
        # 3. Gentle attraction to maximize radii (grow circles slightly)
        # This is handled implicitly by not shrinking radii
        
        # Total force
        total_fx = force_x.sum(axis=1) + boundary_fx
        total_fy = force_y.sum(axis=1) + boundary_fy
        
        # Update positions with noise
        step_size = 0.01
        noise = T * 0.05
        pos[:, 0] += step_size * total_fx + rng.normal(0, noise, N)
        pos[:, 1] += step_size * total_fy + rng.normal(0, noise, N)
        
        # Clamp to valid range
        pos[:, 0] = np.clip(pos[:, 0], 0.005, 0.995)
        pos[:, 1] = np.clip(pos[:, 1], 0.005, 0.995)
        
        # Try to grow radii if no overlaps
        # Check current validity
        if _is_valid(pos, radii):
            # Try growing each radius slightly
            for i in range(N):
                # Max possible radius given boundaries and other circles
                max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                max_r_circ = max_r_wall
                for j in range(N):
                    if i == j:
                        continue
                    d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                    possible_r = d - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                # Grow toward max_r_circ with some margin
                target_r = max_r_circ * 0.995
                if target_r > radii[i]:
                    radii[i] = min(target_r, radii[i] * 1.001 + 0.0001)
        
        # Track best
        if _is_valid(pos, radii):
            s = np.sum(radii)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin before 150s kill

    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.45))

    penalty_weight = 500.0

    # Build candidate initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    pos_la, rad_la = _initial_lattice()
    inits.append((pos_la, rad_la))
    for s in (100, 200, 300, 400, 500):
        p, r = _initial_random(s)
        inits.append((p, r))

    best_pos = inits[0][0].copy()
    best_radii = inits[0][1].copy()
    best_pos, best_radii = _post_process(best_pos, best_radii)
    best_score = np.sum(best_radii) if _is_valid(best_pos, best_radii) else -1.0

    # 1) Global exploration with differential_evolution (bounded, robust)
    if time.time() < deadline - 35:
        try:
            # DE bounds: positions and radii
            de_bounds = [(0.005, 0.995)] * (2 * N) + [(0.005, 0.45)] * N
            de_res = differential_evolution(
                _objective, de_bounds, args=(penalty_weight,),
                seed=42, maxiter=30, tol=1e-8, popsize=15,
                polish=False, update_interval=5,
                init='sobol')
            if de_res is not None and de_res.x is not None:
                p, r, s = _polish(de_res.x, bounds, deadline - 25, penalty_weight)
                if s > best_score:
                    best_score = s
                    best_pos = p
                    best_radii = r
        except Exception:
            pass

    # 2) Local multi-start polishing from each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        try:
            p, r, s = _polish(x0, bounds, deadline - 15, penalty_weight)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r
        except Exception:
            pass

    # 3) Force-directed annealing from current best
    if time.time() < deadline - 10:
        try:
            p, r, s = _force_directed_anneal(best_pos, best_radii, deadline - 5, n_iter=8000, seed=123)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r
        except Exception:
            pass

    # 4) Final high-precision polish on the global best
    if time.time() < deadline - 3:
        x_best = np.concatenate([best_pos.flatten(), best_radii])
        try:
            p, r, s = _polish(x_best, bounds, deadline - 2, penalty_weight)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r
        except Exception:
            pass

    # Final validation + post-processing
    best_pos, best_radii = _post_process(best_pos, best_radii)

    # Fallback if invalid
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
