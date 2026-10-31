"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

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

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(200):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _try_radius_expansion(pos, radii, max_iter=100, step_size=0.005):
    """Try to expand radii of circles not touching any constraint."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            touching_boundary = False
            if abs(pos[i, 0] - radii[i]) < 1e-5 or abs(pos[i, 0] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            if abs(pos[i, 1] - radii[i]) < 1e-5 or abs(pos[i, 1] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            touching_circle = False
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-5:
                    touching_circle = True
                    break
            if not touching_boundary and not touching_circle:
                max_expansion = 0.01
                max_expansion = min(max_expansion, pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, pos[i, 1] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 1] - radii[i])
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    space = dist - radii[i] - radii[j]
                    max_expansion = min(max_expansion, space)
                if max_expansion > 1e-6:
                    expansion = min(step_size, max_expansion * 0.5)
                    radii[i] += expansion
                    improved = True
        if not improved:
            break
    return pos, radii

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
        for _ in range(5000):
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
        for _ in range(100):
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
            for attempt in range(200):
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

def _slsqp_hard(pos, radii, deadline):
    """
    Optimize using SLSQP with explicit hard constraints.
    This is the primary optimizer for strict feasibility.
    """
    n = len(radii)
    x = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.99))
        bounds.append((1e-4, 0.99))
    for i in range(n):
        bounds.append((1e-4, 0.5))
    overlap_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            overlap_pairs.append((i, j))
    def boundary_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = []
        for i in range(n):
            cons.append(pos[i, 0] - radii[i])
            cons.append(1.0 - pos[i, 0] - radii[i])
            cons.append(pos[i, 1] - radii[i])
            cons.append(1.0 - pos[i, 1] - radii[i])
        return np.array(cons)
    def overlap_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = []
        for i, j in overlap_pairs:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            cons.append(dist_sq - min_dist*min_dist)
        return np.array(cons)
    def objective(x):
        radii = x[2*n:]
        return -np.sum(radii)
    def jac_objective(x):
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        return grad
    def jac_boundary(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        J = np.zeros((4*n, 2*n + n))
        for i in range(n):
            row_base = 4*i
            J[row_base, i] = 1.0
            J[row_base, 2*n + i] = -1.0
            J[row_base+1, i] = -1.0
            J[row_base+1, 2*n + i] = -1.0
            J[row_base+2, i+n] = 1.0
            J[row_base+2, 2*n + i] = -1.0
            J[row_base+3, i+n] = -1.0
            J[row_base+3, 2*n + i] = -1.0
        return J
    def jac_overlap(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        m = len(overlap_pairs)
        J = np.zeros((m, 2*n + n))
        for k, (i, j) in enumerate(overlap_pairs):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            J[k, i] = 2*dx
            J[k, j] = -2*dx
            J[k, i+n] = 2*dy
            J[k, j+n] = -2*dy
            sum_r = radii[i] + radii[j]
            J[k, 2*n + i] = -2*sum_r
            J[k, 2*n + j] = -2*sum_r
        return J
    try:
        res = minimize(objective, x, method='SLSQP', bounds=bounds,
                       constraints=[
                           {'type': 'ineq', 'fun': boundary_constraints, 'jac': jac_boundary},
                           {'type': 'ineq', 'fun': overlap_constraints, 'jac': jac_overlap}
                       ],
                       jac=jac_objective,
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or (hasattr(res, 'x') and res.fun < objective(x)):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_radii = np.maximum(res.x[2*n:], 1e-4)
            return new_pos, new_radii
    except Exception:
        pass
    return pos, radii

def _epigraph_optimize(pos, radii, deadline):
    """
    Epigraph formulation: introduce auxiliary variable t for each constraint.
    Maximize sum(radii) subject to t_i >= 0 and constraint_i <= t_i.
    This smooths the nonsmooth min/max in the constraints.
    """
    n = len(radii)
    n_overlap = n * (n - 1) // 2
    n_boundary = 4 * n
    n_t = n_overlap + n_boundary
    # x = [positions, radii, t_overlap, t_boundary]
    x0 = np.concatenate([pos.flatten(), radii, np.zeros(n_t)])
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.99))
        bounds.append((1e-4, 0.99))
    for i in range(n):
        bounds.append((1e-4, 0.5))
    for i in range(n_t):
        bounds.append((0.0, 1.0))
    overlap_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            overlap_pairs.append((i, j))
    def objective(x):
        radii = x[2*n:3*n]
        return -np.sum(radii)
    def jac_objective(x):
        grad = np.zeros_like(x)
        grad[2*n:3*n] = -1.0
        return grad
    def boundary_t_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:3*n]
        t_b = x[3*n:3*n+n_boundary]
        cons = []
        idx = 0
        for i in range(n):
            cons.append(t_b[idx] - (pos[i, 0] - radii[i]))
            idx += 1
            cons.append(t_b[idx] - (1.0 - pos[i, 0] - radii[i]))
            idx += 1
            cons.append(t_b[idx] - (pos[i, 1] - radii[i]))
            idx += 1
            cons.append(t_b[idx] - (1.0 - pos[i, 1] - radii[i]))
            idx += 1
        return np.array(cons)
    def overlap_t_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:3*n]
        t_o = x[3*n+n_boundary:3*n+n_boundary+n_overlap]
        cons = []
        idx = 0
        for i, j in overlap_pairs:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            cons.append(t_o[idx] - (dist_sq - min_dist*min_dist))
            idx += 1
        return np.array(cons)
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=[
                           {'type': 'ineq', 'fun': boundary_t_constraints},
                           {'type': 'ineq', 'fun': overlap_t_constraints}
                       ],
                       jac=jac_objective,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or (hasattr(res, 'x') and res.fun < objective(x0)):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_radii = np.maximum(res.x[2*n:3*n], 1e-4)
            return new_pos, new_radii
    except Exception:
        pass
    return pos, radii

def _force_directed_relax(pos, radii, n_iter=500, dt=0.001, deadline=None):
    """Force-directed relaxation with adaptive damping."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        forces = np.zeros((n, 2))
        radius_forces = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx, dy = 1e-5, 1e-5
                    dist = np.sqrt(dx*dx + dy*dy)
                gap = dist - radii[i] - radii[j]
                if gap < 0.05:
                    strength = 5.0 / (0.01 + abs(gap))
                    fx = strength * dx / dist
                    fy = strength * dy / dist
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
                    radius_forces[i] += 0.5
                    radius_forces[j] += 0.5
        for i in range(n):
            for axis in range(2):
                coord = pos[i, axis]
                if coord - radii[i] < 0.05:
                    strength = 5.0 / (0.01 + max(0.05 - (coord - radii[i]), 0.001))
                    forces[i, axis] += strength
                if (1.0 - coord) - radii[i] < 0.05:
                    strength = 5.0 / (0.01 + max(0.05 - ((1.0 - coord) - radii[i]), 0.001))
                    forces[i, axis] -= strength
        damping = 0.9 - 0.5 * (it / max(n_iter, 1))
        for i in range(n):
            pos[i, 0] += forces[i, 0] * dt
            pos[i, 1] += forces[i, 1] * dt
            pos[i, 0] = np.clip(pos[i, 0], radii[i] + 1e-6, 1.0 - radii[i] - 1e-6)
            pos[i, 1] = np.clip(pos[i, 1], radii[i] + 1e-6, 1.0 - radii[i] - 1e-6)
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] + 0.001:
                radii[i] = min(max_r * 0.999, radii[i] + 0.005)
        if it % 20 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP, epigraph, and local expansion."""
    pos = pos.copy()
    radii = radii.copy()
    # Stage 1: SLSQP with hard constraints
    if time.time() < deadline - 15:
        pos, radii = _slsqp_hard(pos, radii, deadline)
        pos, radii = _post_process(pos, radii)
    # Stage 2: Epigraph optimization
    if time.time() < deadline - 10:
        pos_ep, radii_ep = _epigraph_optimize(pos, radii, deadline)
        pos_pp, radii_pp = _post_process(pos_ep, radii_ep)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii):
            pos, radii = pos_pp, radii_pp
    # Stage 3: Force-directed relaxation
    if time.time() < deadline - 8:
        pos_fd, radii_fd = _force_directed_relax(pos, radii, n_iter=300, dt=0.0008, deadline=deadline)
        if np.sum(radii_fd) > np.sum(radii):
            pos, radii = pos_fd, radii_fd
    # Stage 4: Local radius expansion
    if time.time() < deadline - 5:
        pos, radii = _try_radius_expansion(pos, radii, max_iter=150, step_size=0.004)
        pos, radii = _post_process(pos, radii)
    # Stage 5: Another SLSQP pass
    if time.time() < deadline - 3:
        pos, radii = _slsqp_hard(pos, radii, deadline)
        pos, radii = _try_radius_expansion(pos, radii, max_iter=80, step_size=0.002)
        pos, radii = _post_process(pos, radii)
    # Stage 6: Final micro-relaxation
    if time.time() < deadline - 1:
        pos_fd2, radii_fd2 = _force_directed_relax(pos, radii, n_iter=100, dt=0.0003, deadline=deadline)
        if np.sum(radii_fd2) > np.sum(radii):
            pos, radii = pos_fd2, radii_fd2
        pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    # Primary optimization: SLSQP with hard constraints for each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        # SLSQP hard constraints
        if time.time() < deadline - 18:
            try:
                pos_opt, rad_opt = _slsqp_hard(pos_init, rad_init, deadline)
                pos_pp, radii_pp = _post_process(pos_opt, rad_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        # Epigraph optimization
        if time.time() < deadline - 12:
            try:
                pos_ep, rad_ep = _epigraph_optimize(pos_init, rad_init, deadline)
                pos_pp, radii_pp = _post_process(pos_ep, rad_ep)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        # Force-directed relaxation
        if time.time() < deadline - 8:
            try:
                pos_fd, radii_fd = _force_directed_relax(best_pos, best_radii, n_iter=200, dt=0.001, deadline=deadline)
                pos_pp, radii_pp = _post_process(pos_fd, radii_fd)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    # Final polish
    if time.time() < deadline - 18:
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
        # Secondary polish from perturbed initial
        if time.time() < deadline - 10:
            rng = np.random.RandomState(42)
            pos_pert = best_pos + rng.normal(0, 0.003, (N, 2))
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            radii_pert = best_radii * 0.98
            try:
                pos_pol2, rad_pol2 = _polish(pos_pert, radii_pert, deadline)
                pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
                score2 = np.sum(radii_pp2)
                if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                    best_score = score2
                    best_pos = pos_pp2.copy()
                    best_radii = radii_pp2.copy()
            except Exception:
                pass
        # Tertiary: Force-directed from best
        if time.time() < deadline - 6:
            try:
                pos_fd3, radii_fd3 = _force_directed_relax(best_pos, best_radii, n_iter=300, dt=0.0005, deadline=deadline)
                pos_pp3, radii_pp3 = _post_process(pos_fd3, radii_fd3)
                score3 = np.sum(radii_pp3)
                if _is_valid(pos_pp3, radii_pp3) and score3 > best_score:
                    best_score = score3
                    best_pos = pos_pp3.copy()
                    best_radii = radii_pp3.copy()
            except Exception:
                pass
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    # Final radius expansion
    if time.time() < deadline - 2:
        pos_exp, rad_exp = _try_radius_expansion(best_pos, best_radii, max_iter=150, step_size=0.001)
        pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
            best_pos = pos_pp
            best_radii = radii_pp
            best_score = np.sum(radii_pp)
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
