"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines force-directed annealing with structural parameterization and
high-precision polish. The annealing phase explores topological arrangements
while the structural search reduces dimensionality for global optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty. Vectorized."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations (vectorized) - smooth quadratic
    left = np.maximum(0.0, radii - pos[:, 0])
    right = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bot = np.maximum(0.0, radii - pos[:, 1])
    top = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    bpen = np.sum(left * left) + np.sum(right * right) + np.sum(bot * bot) + np.sum(top * top)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = np.maximum(0.0, min_dist - dist)
    np.fill_diagonal(viol, 0.0)
    open_pen = 0.5 * np.sum(viol * viol)

    return -np.sum(radii) + 1000.0 * (bpen + open_pen)


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Force-directed annealing ----------------

def _force_anneal(pos, radii, deadline, rng):
    """
    Force-directed simulated annealing to explore topological arrangements.
    Circles repel each other when overlapping; boundary forces push them inward.
    Temperature controls the magnitude of random perturbations.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    T0 = 0.1
    Tend = 0.001
    n_steps = 3000
    cooling = (Tend / T0) ** (1.0 / n_steps)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Compute initial penalty
    def _penalty(pos, radii):
        pen = 0.0
        for i in range(N):
            if pos[i, 0] < radii[i]:
                pen += (radii[i] - pos[i, 0]) ** 2
            if pos[i, 0] > 1.0 - radii[i]:
                pen += (pos[i, 0] - (1.0 - radii[i])) ** 2
            if pos[i, 1] < radii[i]:
                pen += (radii[i] - pos[i, 1]) ** 2
            if pos[i, 1] > 1.0 - radii[i]:
                pen += (pos[i, 1] - (1.0 - radii[i])) ** 2
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    pen += (min_dist - dist) ** 2
        return pen
    
    current_penalty = _penalty(pos, radii)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        T = T0 * cooling ** step
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces: push circles away from walls
        for i in range(N):
            if pos[i, 0] < radii[i]:
                forces[i, 0] += 10.0 * (radii[i] - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i]:
                forces[i, 0] -= 10.0 * (pos[i, 0] - (1.0 - radii[i]))
            if pos[i, 1] < radii[i]:
                forces[i, 1] += 10.0 * (radii[i] - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i]:
                forces[i, 1] -= 10.0 * (pos[i, 1] - (1.0 - radii[i]))
        
        # Overlap forces: repel overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force_mag = 50.0 * overlap
                    # Direction: away from each other
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist <= 1e-12:
                    # Random kick if exactly overlapping
                    forces[i, 0] += rng.uniform(-1, 1)
                    forces[i, 1] += rng.uniform(-1, 1)
                    forces[j, 0] += rng.uniform(-1, 1)
                    forces[j, 1] += rng.uniform(-1, 1)
        
        # Random thermal noise
        noise = T * rng.randn(N, 2)
        
        # Update positions
        pos += 0.01 * (forces + noise)
        
        # Clamp to unit square (with margin for radii)
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.0, 1.0)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, 1.0)
        
        # Check if this configuration is better
        new_penalty = _penalty(pos, radii)
        if new_penalty < current_penalty or rng.rand() < np.exp(-(new_penalty - current_penalty) / (T + 1e-12)):
            current_penalty = new_penalty
            if np.sum(radii) > best_score and _is_valid(pos, radii):
                best_score = np.sum(radii)
                best_pos = pos.copy()
                best_radii = radii.copy()
    
    return best_pos, best_radii


# ---------------- Initial configurations ----------------

def _make_grid(n, seed=42):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            idx = row * 5 + col
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def _make_corners(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for i, c in enumerate(corners):
        pos[i] = c
        radii[i] = 0.1
    pos[4] = [0.5, 0.5]
    radii[4] = 0.15
    rng = np.random.RandomState(seed)
    for i in range(5, n):
        pos[i] = rng.uniform(0.2, 0.8, 2)
        radii[i] = 0.08
    return pos, radii


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """
    Iteratively enforce validity by resolving overlaps and boundary violations.
    Uses a pairwise separation strategy: if two circles overlap, move them apart
    along the line connecting their centers. If movement pushes a circle out of bounds,
    shrink it instead.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Ensure initial radii are positive and small enough to be handled
    radii = np.maximum(radii, 1e-6)
    
    max_iters = 1000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        changed = False
        
        # 1. Handle boundary violations: shrink or move inward
        for i in range(N):
            # Calculate max possible radius at current pos
            max_r_x = min(pos[i, 0], 1.0 - pos[i, 0])
            max_r_y = min(pos[i, 1], 1.0 - pos[i, 1])
            max_r = min(max_r_x, max_r_y)
            
            if radii[i] > max_r:
                # If radius is too big for the position, shrink it
                radii[i] = max_r * 0.999
                changed = True
                
        # 2. Handle overlaps: pairwise separation
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    changed = True
                    
                    if dist < 1e-12:
                        # If centers are identical, just shrink them to allow separation
                        total_r = radii[i] + radii[j]
                        if total_r > 0:
                            # Shrink to half their current sum, effectively creating space
                            radii[i] *= 0.9
                            radii[j] *= 0.9
                    else:
                        # Move apart along the connecting vector
                        # We want to increase dist by 'overlap'.
                        # Each moves by overlap/2 if possible.
                        # Normalize direction
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Proposed movement
                        move_i = overlap * 0.5
                        move_j = overlap * 0.5
                        
                        new_pos_i = pos[i].copy()
                        new_pos_j = pos[j].copy()
                        
                        new_pos_i[0] += move_i * nx
                        new_pos_i[1] += move_i * ny
                        new_pos_j[0] -= move_j * nx
                        new_pos_j[1] -= move_j * ny
                        
                        # Check if new positions are within bounds
                        # If not, we might need to shrink or reduce movement
                        # Simple strategy: if out of bounds, shrink the circle that is more constrained
                        # Or simply clamp the movement.
                        
                        # Let's check bounds for new_pos_i
                        in_bounds_i = (new_pos_i[0] >= radii[i] and new_pos_i[0] <= 1.0 - radii[i] and
                                      new_pos_i[1] >= radii[i] and new_pos_i[1] <= 1.0 - radii[i])
                        in_bounds_j = (new_pos_j[0] >= radii[j] and new_pos_j[0] <= 1.0 - radii[j] and
                                      new_pos_j[1] >= radii[j] and new_pos_j[1] <= 1.0 - radii[j])
                        
                        if in_bounds_i and in_bounds_j:
                            pos[i] = new_pos_i
                            pos[j] = new_pos_j
                        else:
                            # If moving apart pushes out of bounds, shrink the circles instead
                            # Reduce radii to allow the current distance to be valid
                            # dist >= r_i + r_j  => r_i + r_j <= dist
                            # We have dist < r_i + r_j.
                            # We can scale down radii.
                            # Let r_i' + r_j' = dist * 0.999
                            # Keep ratio of radii? Or shrink equally?
                            # Shrink the smaller one more?
                            # Simple: scale both to fit in current distance
                            total_r = radii[i] + radii[j]
                            if total_r > 0:
                                scale = (dist * 0.999) / total_r
                                radii[i] *= scale
                                radii[j] *= scale
                            
                            # Also, if they are too close to boundary, shrink them further
                            for k in [i, j]:
                                max_r_k = min(pos[k, 0], 1.0 - pos[k, 0], pos[k, 1], 1.0 - pos[k, 1])
                                if radii[k] > max_r_k:
                                    radii[k] = max_r_k * 0.999

        if not changed:
            break

    # Final safety check: ensure radii are positive and within bounds
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
        
    # One last verification pass to shrink any remaining overlaps
    # This is a simple pass to catch any numerical drift
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j]:
                # Shrink both to fit
                overlap = (radii[i] + radii[j]) - dist
                # Reduce radii by half the overlap each, or scale
                # Scale to fit
                total = radii[i] + radii[j]
                if total > 0:
                    scale = dist / total
                    radii[i] *= scale
                    radii[j] *= scale

    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision L-BFGS-B polish with tight bounds and tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
        bounds.append((max(0.0, radii[i] * 0.5), min(1.0, 1.0 - radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass

    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Phase 1: Generate initial configurations
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    configs.append(_make_corners(N))
    
    rng = np.random.RandomState(42)
    for seed in range(5):
        configs.append(_make_random(N, seed=seed + 100))
    
    # Phase 2: Force-directed annealing from each config
    for pos, radii in configs:
        if time.time() > deadline - 60:
            break
        try:
            pos_fa, radii_fa = _force_anneal(pos, radii, deadline - 60, rng)
            update_best(pos_fa, radii_fa)
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B optimization from each config
    for pos, radii in configs:
        if time.time() > deadline - 40:
            break
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.3))
        
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 4: Structural parameterization search
    if time.time() < deadline - 30:
        try:
            p0 = np.array([0.12, 0.15, 0.5, 0.5, 0.3, 0.1, 0.1, 0.42, 0.07, 0.07])
            bounds_struct = [
                (0.05, 0.30), (0.05, 0.35), (0.2, 0.8), (0.2, 0.8),
                (0.15, 0.6), (0.03, 0.20), (0.03, 0.20),
                (0.25, 0.7), (0.02, 0.16), (0.02, 0.16)
            ]
            de_result = differential_evolution(
                lambda p: _struct_objective(p, deadline),
                bounds_struct, maxiter=50, popsize=20, seed=42, tol=1e-8,
                polish=False, mutation=(0.3, 1.0), recombination=0.7
            )
            if de_result.x is not None:
                pos_struct, radii_struct = _expand_struct(de_result.x)
                update_best(pos_struct, radii_struct)
        except Exception:
            pass
    
    # Phase 5: Polish the best result
    if best_pos is not None and time.time() < deadline - 5:
        best_pos, best_radii = _polish(best_pos, best_radii, deadline)
        
        # Extra polish pass if time permits
        if time.time() < deadline - 2:
            best_pos, best_radii = _polish(best_pos, best_radii, deadline)
    
    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
