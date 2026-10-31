"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines multi-phase local optimization with a compact structural
parameterization (corner + center + rings) that reduces the global-search
dimensionality so Differential Evolution can actually explore it.
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
    """Objective: minimize negative sum of radii + smooth penalty.

    Uses a combined approach:
    - Quadratic penalty on the sum of squared positive violations (smooth, well-behaved)
    - Log-sum-exp soft-max on the worst violation (directional, focuses on binding constraints)
    This gives both global smoothness and local directionality.
    """
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Temperature for the soft-max component
    beta = 60.0

    # Boundary violations (vectorized)
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    # Stack all violation magnitudes
    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    # Quadratic penalty on positive violations (smooth, captures total violation magnitude)
    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    # Log-sum-exp soft-max on the worst violation (directional, focuses on binding constraints)
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    # Combined penalty: quadratic for smoothness, soft-max for directionality
    penalty = 50.0 * quad_penalty + 20.0 * sm

    return -np.sum(radii) + penalty


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


# ---------------- Structural seeds ----------------

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


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity, conservatively."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
            break
        valid = True
        # Boundary: clamp radii to fit in square
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.9999
                valid = False
        # Overlap: only shrink the smaller circle, minimally
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Shrink the smaller circle to resolve overlap
                    if radii[i] <= radii[j]:
                        # Need to reduce radii[i] by at least overlap
                        # But be conservative: only shrink by overlap amount
                        new_r = radii[i] - overlap
                        if new_r < 1e-6:
                            new_r = 1e-6
                        radii[i] = new_r
                    else:
                        new_r = radii[j] - overlap
                        if new_r < 1e-6:
                            new_r = 1e-6
                        radii[j] = new_r
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r * 0.9999))

    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """SLSQP polishing with explicit constraints."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii, deadline)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    def objective(x):
        return -np.sum(x[2*N:])

    def make_constraint(i, j):
        def constraint(x):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - x[2*N+i] - x[2*N+j]
        return constraint

    # Boundary constraints
    boundary_cons = []
    for i in range(N):
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i+1]})

    # Overlap constraints
    overlap_cons = []
    for i in range(N):
        for j in range(i + 1, N):
            overlap_cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=boundary_cons + overlap_cons,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() < deadline:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt, deadline)
            if _is_valid(p_opt, r_opt):
                s = np.sum(r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def _gradient_polish(pos, radii, deadline):
    """Gradient-based polishing with smooth penalty."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii, deadline)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    def objective(x):
        return _objective(x, deadline)

    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))

    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if time.time() < deadline:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt, deadline)
            if _is_valid(p_opt, r_opt):
                s = np.sum(r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(p, r):
        nonlocal best_pos, best_radii, best_score
        if _is_valid(p, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p.copy()
                best_radii = r.copy()

    # 1. Structural Differential Evolution
    # This is the key new strategy: exploring a low-dimensional structural space.
    struct_bounds = [
        (0.05, 0.30), # corner_r
        (0.05, 0.35), # center_r
        (0.2, 0.8),   # cx
        (0.2, 0.8),   # cy
        (0.15, 0.6),  # r1_off
        (0.03, 0.20), # r1_r
        (0.03, 0.20), # r1_r2
        (0.25, 0.7),  # r2_off
        (0.02, 0.16), # r2_r
        (0.02, 0.16), # r2_r2
    ]

    if time.time() < deadline - 30:
        try:
            # DE is stochastic, so we run it for a fixed reasonable time or maxiter
            # We'll use a wrapper to check time
            de_start = time.time()
            # maxiter=100, popsize=15 is ~1500 evaluations. Should be fast enough.
            result_de = differential_evolution(
                lambda p: _struct_objective(p, deadline),
                bounds=struct_bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                polish=False, # We do our own polishing
                init='sobol'
            )
            if result_de.success or time.time() < deadline - 30:
                p_struct = result_de.x
                pos_struct, radii_struct = _expand_struct(p_struct)
                pos_struct, radii_struct = _post_process(pos_struct, radii_struct, deadline)
                update_best(pos_struct, radii_struct)
                
                # Polish the structural result
                if time.time() < deadline - 20:
                    pos_p, radii_p, _ = _slsqp_polish(pos_struct, radii_struct, deadline)
                    update_best(pos_p, radii_p)
        except Exception:
            pass

    # 2. Standard seeds + Polish
    seeds = [
        ('grid', 42),
        ('hexagonal', 42),
        ('center_ring', 42),
        ('random', 42),
        ('random', 123),
    ]

    for strat_name, seed in seeds:
        if time.time() > deadline - 25:
            break

        if strat_name == 'grid':
            pos, radii = _make_grid(N, seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(N, seed)
        elif strat_name == 'center_ring':
            pos, radii = _make_center_ring(N, seed)
        elif strat_name == 'random':
            pos, radii = _make_random(N, seed)
        else:
            continue

        pos, radii = _post_process(pos, radii, deadline)
        update_best(pos, radii)

        # SLSQP Polish
        if time.time() < deadline - 15:
            pos_p, radii_p, _ = _slsqp_polish(pos, radii, deadline)
            update_best(pos_p, radii_p)

        # Gradient Polish
        if time.time() < deadline - 10:
            pos_g, radii_g, _ = _gradient_polish(pos_p, radii_p, deadline)
            update_best(pos_g, radii_g)

    # 3. Final polish on the best found so far
    if best_pos is not None:
        if time.time() < deadline - 10:
            pos_f, radii_f, _ = _slsqp_polish(best_pos, best_radii, deadline)
            update_best(pos_f, radii_f)
        
        # Additional gradient polish for high-precision final pass
        if time.time() < deadline - 5:
            pos_g, radii_g, _ = _gradient_polish(best_pos, best_radii, deadline)
            update_best(pos_g, radii_g)
        
        # One more SLSQP pass on the best result
        if time.time() < deadline - 2:
            pos_final, radii_final, _ = _slsqp_polish(best_pos, best_radii, deadline)
            update_best(pos_final, radii_final)

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N, 42)
        pos, radii = _post_process(pos, radii, deadline)
        best_pos, best_radii = pos, radii

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
