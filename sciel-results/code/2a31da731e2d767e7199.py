"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii, shrink_factor=0.99999):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(300):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r * shrink_factor:
                radii[i] = max(max_r * shrink_factor, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * shrink_factor
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _try_radius_expansion(pos, radii, max_iterations=400):
    """Iteratively expand each circle's radius using bisection."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(max_iterations):
        improved = False
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])
            if max_r_circ <= r * 1.000001:
                continue
            lo, hi = r, max_r_circ * 0.9999999
            for _ in range(25):
                mid = (lo + hi) / 2.0
                radii[i] = mid
                if _is_valid(pos, radii):
                    lo = mid
                else:
                    hi = mid
            radii[i] = lo
            if lo > r * 1.000001:
                improved = True
        if not improved:
            break
    return pos, radii

def _objective_constrained(x, penalty=1000.0):
    """Penalty-based objective for unconstrained optimizer."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    penalty_val = 0.0
    # Boundary violations
    bv = (np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) +
          np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii)))
    penalty_val += np.sum(bv * bv)
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, (min_dist * min_dist - dist_sq)[mask])
    penalty_val += np.sum(viol * viol)
    return -np.sum(radii) + penalty * penalty_val

def _slsqp_constraints(x):
    """Build SLSQP inequality constraints: g(x) >= 0 for feasibility."""
    constraints = []
    n = N
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i + 1] - x[2*N + i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i + 1] - x[2*N + i]})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*N+i] + x[2*N+j])**2
            })
    return constraints

def _slsqp_optimize(x0, bounds, deadline, maxiter=3000, ftol=1e-14, gtol=1e-12):
    """SLSQP optimization with hard constraints."""
    try:
        pos_tmp = x0[:2*N].reshape(N, 2)
        radii_tmp = np.maximum(x0[2*N:], 1e-5)
        pos_pp, radii_pp = _post_process(pos_tmp, radii_tmp)
        x_start = np.concatenate([pos_pp.flatten(), radii_pp])
        cons = _slsqp_constraints(x_start)
        result = minimize(lambda v: -np.sum(v[2*N:]), x_start, method='SLSQP',
                         bounds=bounds, constraints=cons,
                         options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        if time.time() > deadline - 2:
            return None
        if result.fun < -np.sum(x_start[2*N:]) - 1e-10:
            return result.x
        return None
    except Exception:
        return None

def _lbfgsb_optimize(x0, bounds, deadline, penalty=5000.0, maxiter=3000):
    """L-BFGS-B optimization with penalty method."""
    try:
        result = minimize(_objective_constrained, x0, method='L-BFGS-B',
                         bounds=bounds, args=(penalty,),
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() > deadline - 2:
            return None
        if result.fun < _objective_constrained(x0, penalty) - 1e-10:
            return result.x
        return None
    except Exception:
        return None

def _force_directed_relax(pos, radii, iterations=2000, deadline=None):
    """Force-directed relaxation to push overlapping circles apart and expand radii."""
    pos = pos.copy()
    radii = radii.copy()
    dt = 0.01
    for it in range(iterations):
        if deadline and time.time() > deadline - 2:
            break
        forces = np.zeros((N, 2))
        # Repulsive forces between overlapping circles
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Overlap: push apart
                    force_mag = (min_dist - dist) * 10.0
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i] += [fx, fy]
                    forces[j] += [-fx, -fy]
        # Attractive forces toward walls to encourage expansion
        for i in range(N):
            x, y = pos[i]
            # Push toward center of available space
            margin = 0.05
            if x < radii[i] + margin:
                forces[i, 0] += (radii[i] + margin - x) * 5.0
            if x > 1.0 - radii[i] - margin:
                forces[i, 0] -= (x - (1.0 - radii[i] - margin)) * 5.0
            if y < radii[i] + margin:
                forces[i, 1] += (radii[i] + margin - y) * 5.0
            if y > 1.0 - radii[i] - margin:
                forces[i, 1] -= (y - (1.0 - radii[i] - margin)) * 5.0
        # Update positions
        pos += dt * forces
        # Clamp to valid range
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        # Gradually increase radii
        growth = 0.0001
        for i in range(N):
            max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r_circ = min(max_r_circ, dist - radii[j])
            if max_r_circ > radii[i] * 1.0001:
                radii[i] = min(radii[i] * (1.0 + growth), max_r_circ * 0.9999)
    # Post-process to ensure validity
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _coordinate_descent(pos, radii, deadline, max_iter=500):
    """Coordinate descent: optimize each circle's position and radius separately."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            x0, y0 = pos[i]
            r0 = radii[i]
            # Optimize position with fixed radius
            best_x, best_y, best_r = x0, y0, r0
            for dx in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                for dy in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99:
                        continue
                    max_r_wall = min(nx, 1.0 - nx, ny, 1.0 - ny)
                    max_r_circ = max_r_wall
                    for j in range(N):
                        if i == j:
                            continue
                        dxj = nx - pos[j, 0]
                        dyj = ny - pos[j, 1]
                        dist = np.sqrt(dxj * dxj + dyj * dyj)
                        max_r_circ = min(max_r_circ, dist - radii[j])
                    if max_r_circ > best_r * 1.001:
                        best_r = max_r_circ
                        best_x, best_y = nx, ny
            # Try radius expansion at best position
            if best_x != x0 or best_y != y0:
                pos[i] = [best_x, best_y]
                improved = True
            # Bisection for max radius at best position
            max_r_wall = min(best_x, 1.0 - best_x, best_y, 1.0 - best_y)
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dxj = best_x - pos[j, 0]
                dyj = best_y - pos[j, 1]
                dist = np.sqrt(dxj * dxj + dyj * dyj)
                max_r_circ = min(max_r_circ, dist - radii[j])
            if max_r_circ > best_r * 1.0001:
                lo, hi = best_r, max_r_circ * 0.9999999
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    radii[i] = mid
                    if _is_valid(pos, radii):
                        lo = mid
                    else:
                        hi = mid
                radii[i] = lo
                if lo > best_r * 1.0001:
                    improved = True
        if not improved:
            break
    return pos, radii

def _simulated_annealing(pos, radii, deadline, max_iter=10000):
    """Simulated annealing to escape local optima."""
    pos = pos.copy()
    radii = radii.copy()
    T = 0.05
    T_min = 0.001
    cooling = 0.9995
    best_pos, best_radii, best_score = pos.copy(), radii.copy(), np.sum(radii)
    current_score = best_score
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        # Random perturbation
        i = np.random.randint(N)
        dx = np.random.normal(0, 0.01)
        dy = np.random.normal(0, 0.01)
        dr = np.random.normal(0, 0.005)
        nx, ny = pos[i, 0] + dx, pos[i, 1] + dy
        nr = radii[i] + dr
        if nx < 0.01 or nx > 0.99 or ny < 0.01 or ny > 0.99 or nr < 0.005:
            continue
        pos[i, 0], pos[i, 1], radii[i] = nx, ny, nr
        # Check validity
        pos_pp, radii_pp = _post_process(pos, radii)
        new_score = np.sum(radii_pp)
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            pos, radii = pos_pp, radii_pp
            if current_score > best_score:
                best_score = current_score
                best_pos, best_radii = pos.copy(), radii.copy()
        else:
            pos[i, 0], pos[i, 1], radii[i] = nx - dx, ny - dy, nr - dr
        T *= cooling
        if T < T_min:
            break
    return best_pos, best_radii

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

def _polish(pos, radii, deadline):
    """Multi-stage polishing."""
    pos = pos.copy()
    radii = radii.copy()
    bounds = [(0.005, 0.995), (0.005, 0.995)] * N + [(0.005, 0.5)] * N
    
    # Stage 1: SLSQP with hard constraints
    if time.time() < deadline - 20:
        x = np.concatenate([pos.flatten(), radii])
        result = _slsqp_optimize(x, bounds, deadline, maxiter=2000, ftol=1e-14, gtol=1e-12)
        if result is not None:
            pos = result[:2*N].reshape(N, 2)
            radii = np.maximum(result[2*N:], 1e-5)
            pos, radii = _post_process(pos, radii)
    
    # Stage 2: Force-directed relaxation
    if time.time() < deadline - 15:
        pos, radii = _force_directed_relax(pos, radii, iterations=1500, deadline=deadline)
    
    # Stage 3: Coordinate descent
    if time.time() < deadline - 10:
        pos, radii = _coordinate_descent(pos, radii, deadline, max_iter=300)
    
    # Stage 4: SLSQP again
    if time.time() < deadline - 8:
        x = np.concatenate([pos.flatten(), radii])
        result = _slsqp_optimize(x, bounds, deadline, maxiter=1500, ftol=1e-15, gtol=1e-13)
        if result is not None:
            pos = result[:2*N].reshape(N, 2)
            radii = np.maximum(result[2*N:], 1e-5)
            pos, radii = _post_process(pos, radii)
    
    # Stage 5: Aggressive radius expansion
    if time.time() < deadline - 5:
        pos, radii = _try_radius_expansion(pos, radii, max_iterations=400)
    
    # Stage 6: Final SLSQP
    if time.time() < deadline - 2:
        x = np.concatenate([pos.flatten(), radii])
        result = _slsqp_optimize(x, bounds, deadline, maxiter=1000, ftol=1e-16, gtol=1e-14)
        if result is not None:
            pos = result[:2*N].reshape(N, 2)
            radii = np.maximum(result[2*N:], 1e-5)
            pos, radii = _post_process(pos, radii)
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 146.0  # 4s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    pos_cb2, rad_cb2 = _initial_corner_biased(seed=13)
    inits.append((pos_cb2, rad_cb2))
    
    pos_cb3, rad_cb3 = _initial_corner_biased(seed=23)
    inits.append((pos_cb3, rad_cb3))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = [(0.01, 0.99), (0.01, 0.99)] * N + [(0.01, 0.5)] * N
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        # Force-directed relaxation
        if time.time() < deadline - 20:
            pos_fd, rad_fd = _force_directed_relax(pos_init, rad_init, iterations=1000, deadline=deadline)
            pos_pp, radii_pp = _post_process(pos_fd, rad_fd)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # SLSQP optimization
        if time.time() < deadline - 15:
            x = np.concatenate([pos_init.flatten(), rad_init])
            result = _slsqp_optimize(x, bounds, deadline, maxiter=1500, ftol=1e-14, gtol=1e-12)
            if result is not None:
                pos_opt = result[:2*N].reshape(N, 2)
                radii_opt = np.maximum(result[2*N:], 1e-5)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Coordinate descent
        if time.time() < deadline - 10:
            pos_cd, rad_cd = _coordinate_descent(pos_init, rad_init, deadline, max_iter=200)
            pos_pp, radii_pp = _post_process(pos_cd, rad_cd)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Simulated annealing on best found
    if time.time() < deadline - 15:
        pos_sa, rad_sa = _simulated_annealing(best_pos, best_radii, deadline, max_iter=8000)
        pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
        pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    # Extra polish
    if time.time() < deadline - 2:
        pos_pol2, rad_pol2 = _polish(best_pos, best_radii, deadline)
        pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
        score2 = np.sum(radii_pp2)
        if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
            best_score = score2
            best_pos = pos_pp2.copy()
            best_radii = radii_pp2.copy()
    
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
