"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=100.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective_smooth(x, penalty_weight=500.0, beta=100.0):
    """
    Smooth objective: minimize -sum(radii) + penalty * smooth violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # 1. Boundary violations (smooth)
    viol_left = _soft_max(radii - pos[:, 0], beta)
    viol_right = _soft_max(pos[:, 0] - (1.0 - radii), beta)
    viol_bottom = _soft_max(radii - pos[:, 1], beta)
    viol_top = _soft_max(pos[:, 1] - (1.0 - radii), beta)

    boundary_pen = np.sum(viol_left ** 2) + np.sum(viol_right ** 2) + \
                   np.sum(viol_bottom ** 2) + np.sum(viol_top ** 2)

    # 2. Overlap violations (smooth)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_overlap = min_dist_sq - dist_sq
    viol_overlap_clipped = _soft_max(viol_overlap, beta)
    overlap_pen = np.sum(viol_overlap_clipped[mask] ** 2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective(x, penalty_weight=1000.0):
    """
    Objective function: minimize -sum(radii) + penalty * violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)

    # 1. Boundary violations
    viol_left = np.maximum(0.0, radii - pos[:, 0])
    viol_right = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    viol_bottom = np.maximum(0.0, radii - pos[:, 1])
    viol_top = np.maximum(0.0, pos[:, 1] - (1.0 - radii))

    boundary_pen = np.sum(viol_left ** 2) + np.sum(viol_right ** 2) + \
                   np.sum(viol_bottom ** 2) + np.sum(viol_top ** 2)

    # 2. Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_overlap = min_dist_sq - dist_sq
    viol_overlap_clipped = np.maximum(0.0, viol_overlap)
    overlap_pen = np.sum(viol_overlap_clipped[mask] ** 2)

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _slsqp_objective(x):
    """Objective for SLSQP: minimize -sum(radii)."""
    radii = x[2 * N:]
    return -np.sum(radii)

def _slsqp_constraints(x):
    """Constraints for SLSQP: all constraints must be >= 0."""
    pos = x[:2 * N].reshape(N, 2)
    radii = x[2 * N:]
    
    cons = []
    
    # Boundary constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - x[2*N+i] - x[2*N+j]})
    
    return cons

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

    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii and adjust positions to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(50):
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

def _solve_lp_radii(pos):
    """
    Given fixed positions, solve LP to maximize sum of radii.
    """
    n = len(pos)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            dists[i, j] = d
            dists[j, i] = d
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    max_radii = np.array([min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]) for i in range(n)])
    bounds = [(0.0, max_radii[i]) for i in range(n)]
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return max_radii * 0.5
    except Exception:
        return max_radii * 0.5

def _init_hexagonal():
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
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
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

def _init_corner_biased(seed=7):
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

def _init_best_fit(seed=42):
    """Greedy best-fit initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False

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

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _radius_growth_pass(pos, radii, deadline):
    """
    Fix positions and maximize radii via SLSQP with hard constraints.
    This squeezes additional sum-of-radii by growing circles into gaps.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    current_x = np.concatenate([pos.flatten(), radii])
    
    # Only optimize radii, positions fixed
    bounds = [(0.01, 0.5) for _ in range(N)]
    
    # For radius-only optimization, we need a custom objective and constraints
    def obj_radii(x):
        return -np.sum(x)
    
    def cons_radii(x):
        cons = []
        # Boundary constraints
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: pos[i, 0] - x[i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - pos[i, 0] - x[i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: pos[i, 1] - x[i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - pos[i, 1] - x[i]})
        
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j, d=dist: d - x[i] - x[j]})
        
        return cons
    
    try:
        result = minimize(obj_radii, radii.copy(), method='SLSQP', 
                         bounds=bounds, constraints=cons_radii(radii),
                         options={'maxiter': 200, 'ftol': 1e-14})
        if result.success:
            new_radii = np.maximum(result.x, 1e-4)
            pos_pp, radii_pp = _post_process(pos, new_radii)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii

def _refine_with_slsqp(pos, radii, deadline):
    """
    Use SLSQP with hard constraints to optimize positions and radii jointly.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    radii = _solve_lp_radii(pos)
    
    current_x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    try:
        result = minimize(_slsqp_objective, current_x, method='SLSQP', 
                         bounds=bounds, constraints=_slsqp_constraints(current_x),
                         options={'maxiter': 150, 'ftol': 1e-14})
        if result.success:
            pos_new = result.x[:2 * N].reshape(N, 2)
            rad_new = np.maximum(result.x[2 * N:], 1e-4)
            
            pos_pp, radii_pp = _post_process(pos_new, rad_new)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    except Exception:
        pass
    
    if best_score <= np.sum(radii):
        try:
            result = minimize(lambda z: _objective_smooth(z, 500.0, 100.0), current_x, 
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12})
            pos_new = result.x[:2 * N].reshape(N, 2)
            rad_lp = _solve_lp_radii(pos_new)
            pos_pp, radii_pp = _post_process(pos_new, rad_lp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

def _simulated_annealing(pos, radii, n_steps=1000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """
    Simulated Annealing with LP polish.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    x = np.concatenate([pos.flatten(), radii])
    current_energy = _objective_smooth(x, 500.0, 100.0)

    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac

        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)

        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr

        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)

        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r

        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective_smooth(x_new, 500.0, 100.0)

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r

        if step % 100 == 0:
            radii_lp = _solve_lp_radii(pos)
            radii = radii_lp
            x = np.concatenate([pos.flatten(), radii])
            current_energy = _objective_smooth(x, 500.0, 100.0)
            
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()

    radii_lp = _solve_lp_radii(pos)
    radii = radii_lp
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()

    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    inits = []
    pos_bf, rad_bf = _init_hexagonal()
    inits.append((pos_bf, rad_bf))

    pos_cb, rad_cb = _init_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))

    pos_fit, rad_fit = _init_best_fit(seed=42)
    inits.append((pos_fit, rad_fit))

    for i in range(2):
        pos_r, rad_r = _init_random(100 + i)
        inits.append((pos_r, rad_r))

    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    rad_lp = _solve_lp_radii(best_pos)
    best_radii = rad_lp
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break

        if time.time() < deadline - 15:
            pos_ref, rad_ref = _refine_with_slsqp(pos_init, rad_init, deadline - 15)
            pos_pp, radii_pp = _post_process(pos_ref, rad_ref)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

        if time.time() < deadline - 10:
            pos_sa, rad_sa = _simulated_annealing(
                best_pos, best_radii,
                n_steps=800, t0=0.02, t_end=0.0005,
                seed=idx * 100 + 50, deadline=deadline - 10
            )
            pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            pos_final, rad_final = _refine_with_slsqp(best_pos, best_radii, deadline - 3)
            pos_pp, radii_pp = _post_process(pos_final, rad_final)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Radius growth pass
        if time.time() < deadline - 2:
            try:
                pos_rg, rad_rg = _radius_growth_pass(best_pos, best_radii, deadline - 1)
                pos_pp, radii_pp = _post_process(pos_rg, rad_rg)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass

        try:
            rad_final = _solve_lp_radii(best_pos)
            best_radii = rad_final
            best_pos, best_radii = _post_process(best_pos, best_radii)
            if _is_valid(best_pos, best_radii):
                best_score = np.sum(best_radii)
        except Exception:
            pass

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
