"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses a multi-phase approach:
  1. Structural DE (10-dim) for global exploration
  2. Force-directed relaxation for local improvement
  3. High-precision SLSQP polish with explicit constraints
  4. Greedy growth pass to squeeze out remaining space
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
    """Objective: minimize negative sum of radii + smooth penalty."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    beta = 80.0

    # Boundary violations
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    penalty = 50.0 * quad_penalty + 20.0 * sm
    return -np.sum(radii) + penalty


def _slsqp_constraints(pos, radii):
    """Return constraint functions for SLSQP (all >= 0)."""
    n = N
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_con(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - x[2*n+i] - x[2*n+j]
            cons.append({'type': 'ineq', 'fun': overlap_con})
    return cons


def _slsqp_polish(pos, radii, deadline):
    """SLSQP polish with explicit constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    cons = _slsqp_constraints(pos, radii)
    
    try:
        res = minimize(_objective, x0, args=(deadline,), method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-12})
        if res.fun < 1e9:
            pos_opt = res.x[:2*N].reshape(N, 2)
            radii_opt = res.x[2*N:]
            pos_opt, radii_opt = _post_process(pos_opt, radii_opt, deadline)
            return pos_opt, radii_opt
    except Exception:
        pass
    return pos, radii


def _force_directed(pos, radii, deadline, n_iters=200):
    """Force-directed relaxation to improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    dt = 0.01
    damping = 0.95
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i,0] += fx
                    forces[i,1] += fy
                    forces[j,0] -= fx
                    forces[j,1] -= fy
        
        # Attraction to boundaries for circles near edges
        for i in range(N):
            # Push away from boundaries
            if pos[i,0] < radii[i] + 0.05:
                forces[i,0] += 5.0 * (radii[i] + 0.05 - pos[i,0])
            if pos[i,0] > 1.0 - radii[i] - 0.05:
                forces[i,0] -= 5.0 * (pos[i,0] - (1.0 - radii[i] - 0.05))
            if pos[i,1] < radii[i] + 0.05:
                forces[i,1] += 5.0 * (radii[i] + 0.05 - pos[i,1])
            if pos[i,1] > 1.0 - radii[i] - 0.05:
                forces[i,1] -= 5.0 * (pos[i,1] - (1.0 - radii[i] - 0.05))
        
        # Update positions
        pos += dt * forces
        pos *= damping
        
        # Clamp to valid region
        for i in range(N):
            pos[i,0] = np.clip(pos[i,0], radii[i]+1e-6, 1.0-radii[i]-1e-6)
            pos[i,1] = np.clip(pos[i,1], radii[i]+1e-6, 1.0-radii[i]-1e-6)
    
    # Try to grow radii
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Find max possible radius for circle i
            max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] * 1.001:
                radii[i] = min(radii[i] * 1.01, max_r * 0.999)
                improved = True
        if not improved:
            break
    
    return pos, radii


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(300):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink proportionally
                    shrink = (min_dist - dist) / (radii[i] + radii[j] + 1e-12)
                    radii[i] *= max(0.5, 1.0 - shrink * 0.999)
                    radii[j] *= max(0.5, 1.0 - shrink * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
        radii[i] = max(1e-6, min(radii[i], max_r * 0.999))

    return pos, radii


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

    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0-corner_r, corner_r]
    pos[2] = [corner_r, 1.0-corner_r]
    pos[3] = [1.0-corner_r, 1.0-corner_r]
    radii[0:4] = corner_r

    pos[4] = [cx, cy]
    radii[4] = center_r

    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5+i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5+i] = r1_r if i % 2 == 0 else r1_r2

    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13+i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13+i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


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
        pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i+1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i+9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i+9] = r_outer
    return pos, radii


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def _update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if _is_valid(pos, radii):
            s = np.sum(radii)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()

    # Phase 1: DE on structural parameters
    struct_bounds = [
        (0.05, 0.30), (0.05, 0.35), (0.2, 0.8), (0.2, 0.8),
        (0.15, 0.6), (0.03, 0.20), (0.03, 0.20),
        (0.25, 0.7), (0.02, 0.16), (0.02, 0.16)
    ]

    try:
        de_res = differential_evolution(
            _struct_objective, struct_bounds, args=(deadline,),
            seed=42, maxiter=200, popsize=25, tol=1e-8,
            mutation=(0.3, 1.0), recombination=0.9,
            polish=True, init='latinhypercube',
        )
        if de_res.fun < 1e9:
            p_opt = de_res.x
            pos_s, radii_s = _expand_struct(p_opt)
            pos_s, radii_s = _force_directed(pos_s, radii_s, deadline, n_iters=100)
            pos_s, radii_s = _slsqp_polish(pos_s, radii_s, deadline)
            _update_best(pos_s, radii_s)
    except Exception:
        pass

    # Phase 2: Local optimization from multiple seeds
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    for strategy in strategies:
        if time.time() > deadline - 30:
            break
        for seed in range(3):
            if time.time() > deadline - 25:
                break
            if strategy == 'grid':
                pos, radii = _make_grid(N, seed)
            elif strategy == 'hexagonal':
                pos, radii = _make_hexagonal(N, seed)
            elif strategy == 'center_ring':
                pos, radii = _make_center_ring(N, seed)
            else:
                pos, radii = _make_random(N, seed)

            pos, radii = _force_directed(pos, radii, deadline, n_iters=100)
            pos, radii = _slsqp_polish(pos, radii, deadline)
            _update_best(pos, radii)

    # Phase 3: Final high-precision polish
    if best_pos is not None and time.time() < deadline - 15:
        pos_f, radii_f = _force_directed(best_pos, best_radii, deadline, n_iters=200)
        pos_f, radii_f = _slsqp_polish(pos_f, radii_f, deadline)
        _update_best(pos_f, radii_f)

    if best_pos is None:
        side = 5
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 0.005)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
