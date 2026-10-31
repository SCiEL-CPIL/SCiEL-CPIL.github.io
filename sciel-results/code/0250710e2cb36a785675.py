"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

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


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        # Check boundaries
        for i in range(N):
            if pos[i, 0] < radii[i]:
                radii[i] = max(0.001, pos[i, 0] * 0.98)
                valid = False
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 0]) * 0.98))
                valid = False
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], max(0.001, pos[i, 1] * 0.98))
                valid = False
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], max(0.001, (1.0 - pos[i, 1]) * 0.98))
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = (dist / (min_dist + 1e-12)) * 0.98
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _growth_pass(pos, radii, deadline=None):
    """Try to grow each circle individually to fill available space."""
    pos = pos.copy()
    radii = radii.copy()
    improved = True
    n_iter = 0
    
    while improved and n_iter < 100:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        n_iter += 1
        for i in range(N):
            # Compute max possible radius for circle i given others fixed
            max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            max_r = max(0.001, max_r * 0.9998)
            if max_r > radii[i] + 1e-8:
                radii[i] = max_r
                improved = True
    
    pos_pp, radii_pp = _post_process(pos, radii)
    return pos_pp, radii_pp


def _slsqp_polish(pos, radii, deadline=None):
    """SLSQP polish with explicit constraints for tighter packing."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    n = N
    n_vars = 3 * n
    
    # Build constraint functions vectorized
    def boundary_fun(x):
        vals = np.zeros(4*n)
        for i in range(n):
            vals[4*i] = x[2*i] - x[2*n+i]
            vals[4*i+1] = 1.0 - x[2*i] - x[2*n+i]
            vals[4*i+2] = x[2*i+1] - x[2*n+i]
            vals[4*i+3] = 1.0 - x[2*i+1] - x[2*n+i]
        return vals
    
    def boundary_jac(x):
        jac = np.zeros((4*n, n_vars))
        for i in range(n):
            jac[4*i, 2*i] = 1.0
            jac[4*i, 2*n+i] = -1.0
            jac[4*i+1, 2*i] = -1.0
            jac[4*i+1, 2*n+i] = -1.0
            jac[4*i+2, 2*i+1] = 1.0
            jac[4*i+2, 2*n+i] = -1.0
            jac[4*i+3, 2*i+1] = -1.0
            jac[4*i+3, 2*n+i] = -1.0
        return jac
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    n_pairs = len(pairs)
    
    def overlap_fun(x):
        vals = np.zeros(n_pairs)
        for k, (i, j) in enumerate(pairs):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            vals[k] = dist - x[2*n+i] - x[2*n+j]
        return vals
    
    def overlap_jac(x):
        jac = np.zeros((n_pairs, n_vars))
        for k, (i, j) in enumerate(pairs):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-14 else 1e-7
            inv = 1.0 / dist
            jac[k, 2*i] = dx * inv
            jac[k, 2*i+1] = dy * inv
            jac[k, 2*j] = -dx * inv
            jac[k, 2*j+1] = -dy * inv
            jac[k, 2*n+i] = -1.0
            jac[k, 2*n+j] = -1.0
        return jac
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_fun, 'jac': boundary_jac},
        {'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac}
    ]
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
        if result.fun < -np.sum(radii) - 1e-10:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _contact_graph_slsqp(pos, radii, deadline=None):
    """SLSQP using only active (near-contact) constraints for efficiency."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos.flatten(), radii])
    n = N
    n_vars = 3 * n
    
    # Find active contacts (dist < r_i + r_j + margin)
    margin = 0.02
    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] + margin:
                active_pairs.append((i, j))
    
    # Also include boundary-active constraints
    boundary_active = []
    for i in range(N):
        if pos[i, 0] - radii[i] < margin:
            boundary_active.append((i, 0))  # left
        if 1.0 - pos[i, 0] - radii[i] < margin:
            boundary_active.append((i, 1))  # right
        if pos[i, 1] - radii[i] < margin:
            boundary_active.append((i, 2))  # bottom
        if 1.0 - pos[i, 1] - radii[i] < margin:
            boundary_active.append((i, 3))  # top
    
    if not active_pairs and not boundary_active:
        return pos, radii
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    constraints = []
    
    # Active boundary constraints
    for i, side in boundary_active:
        if side == 0:
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        elif side == 1:
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        elif side == 2:
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        else:
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    
    # Active overlap constraints
    for i, j in active_pairs:
        constraints.append({'type': 'ineq', 
                           'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - x[2*n+i] - x[2*n+j]})
    
    if not constraints:
        return pos, radii
    
    try:
        result = minimize(lambda x: -np.sum(x[2*N:]), x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-13, 'disp': False})
        if result.fun < -np.sum(radii) - 1e-10:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _smooth_objective(x, beta=100.0):
    """Smooth objective with log-sum-exp based penalty."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Boundary violations (smooth)
    penalty = 0.0
    for i in range(n):
        for coord in range(2):
            d_left = pos[i, coord] - radii[i]
            d_right = (1.0 - pos[i, coord]) - radii[i]
            # Smooth max(0, -d) using softplus
            for d in [d_left, d_right]:
                if d < 0:
                    penalty += (-d) ** 2
            # Additional smooth term for gradient
            for d in [d_left, d_right]:
                if d < 0.05:
                    penalty += 0.1 * (-d) ** 2
    
    # Overlap penalties (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-16)
            min_dist = radii[i] + radii[j]
            violation = min_dist - dist
            if violation > 0:
                penalty += violation ** 2
            elif violation > -0.03:
                # Smooth transition
                t = (violation + 0.03) / 0.03
                penalty += 0.01 * t ** 2 * 0.0009
    
    return -np.sum(radii) + beta * penalty


def _force_directed_anneal(pos, radii, n_iter=2000, deadline=None):
    """Force-directed annealing with phase-based strategy."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    rng = np.random.RandomState(12345)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        phase = it / n_iter
        temp = 0.02 * np.exp(-2.0 * phase)
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Boundary forces
        for i in range(N):
            margin = radii[i] + 0.003
            for coord in range(2):
                if pos[i, coord] < margin:
                    forces[i, coord] += 15.0 * (margin - pos[i, coord])
                if pos[i, coord] > 1.0 - margin:
                    forces[i, coord] -= 15.0 * (pos[i, coord] - (1.0 - margin))
        
        # Pairwise forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < 1e-6:
                    # Push apart randomly
                    forces[i, 0] += rng.normal(0, 0.01)
                    forces[i, 1] += rng.normal(0, 0.01)
                    forces[j, 0] -= forces[i, 0]
                    forces[j, 1] -= forces[i, 1]
                    continue
                
                if dist < min_dist:
                    # Strong repulsion for overlaps
                    overlap = min_dist - dist
                    force_mag = 20.0 * overlap + 5.0 * overlap ** 2
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.15:
                    # Gentle repulsion near contact
                    force_mag = 1.5 * (min_dist * 1.15 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update with velocity damping
        vel = 0.8 * vel + 0.2 * forces
        pos += vel * 0.005 + rng.normal(0, temp, size=pos.shape)
        
        # Clamp
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.002, 0.998)
            pos[i, 1] = np.clip(pos[i, 1], 0.002, 0.998)
        
        # Grow radii where possible
        for i in range(N):
            max_r = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > radii[i]:
                radii[i] = max(0.001, min(max_r, radii[i] * 1.02 + 0.001))
        
        # Check and update best
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score + 1e-10:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _make_grid(seed=42):
    """Grid initial configuration."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side + rng.normal(0, 0.01), (j + 0.5) / side + rng.normal(0, 0.01)]
                radii[idx] = 0.5 / side - 0.008
    pos[25] = [0.5, 0.06]
    radii[25] = 0.045
    pos = np.clip(pos, 0.02, 0.98)
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal initial configuration."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.095
    idx = 0
    for row in range(6):
        y = 0.1 + row * r * np.sqrt(3) * 0.95
        if y > 0.9:
            break
        if row % 2 == 0:
            xs = np.arange(6) * 2 * r * 0.95 + r
        else:
            xs = np.arange(5) * 2 * r * 0.95 + 2 * r
        for x in xs:
            if idx < N and x < 1.0:
                pos[idx] = [x + rng.normal(0, 0.005), y + rng.normal(0, 0.005)]
                radii[idx] = r * (1.0 + rng.normal(0, 0.03))
                idx += 1
    # Fill remaining
    while idx < N:
        pos[idx] = rng.uniform(0.1, 0.9, 2)
        radii[idx] = 0.04
        idx += 1
    pos = np.clip(pos, 0.02, 0.98)
    return pos, radii


def _make_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.09, size=N)
    return pos, radii


def _make_center_ring(seed=42):
    """Center + rings configuration."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.13
    
    # Ring 1: 6 circles
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pos[i+1] = [0.5 + 0.27 * np.cos(angle), 0.5 + 0.27 * np.sin(angle)]
        radii[i+1] = 0.11
    
    # Ring 2: 12 circles
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pos[i+7] = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]
        radii[i+7] = 0.075
    
    # Ring 3: 7 more
    for i in range(7):
        angle = 2 * np.pi * i / 7 + 0.2
        pos[i+19] = [0.5 + 0.44 * np.cos(angle), 0.5 + 0.44 * np.sin(angle)]
        radii[i+19] = 0.05
    
    pos += rng.normal(0, 0.01, size=pos.shape)
    pos = np.clip(pos, 0.02, 0.98)
    return pos, radii


def _polish(pos, radii, deadline=None):
    """Multi-method high-precision polish."""
    from scipy.optimize import minimize
    
    best_pos, best_radii = pos.copy(), radii.copy()
    best_score = np.sum(radii)
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    # L-BFGS-B with smooth objective
    if deadline is None or time.time() < deadline - 5:
        try:
            result = minimize(_smooth_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
                best_score = np.sum(radii_pp)
                best_pos, best_radii = pos_pp, radii_pp
        except Exception:
            pass
    
    # Full SLSQP
    if deadline is None or time.time() < deadline - 8:
        try:
            pos_s, rad_s = _slsqp_polish(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_s, rad_s) and np.sum(rad_s) > best_score:
                best_score = np.sum(rad_s)
                best_pos, best_radii = pos_s, rad_s
        except Exception:
            pass
    
    # Contact-graph SLSQP
    if deadline is None or time.time() < deadline - 5:
        try:
            pos_c, rad_c = _contact_graph_slsqp(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_c, rad_c) and np.sum(rad_c) > best_score:
                best_score = np.sum(rad_c)
                best_pos, best_radii = pos_c, rad_c
        except Exception:
            pass
    
    # Growth pass
    if deadline is None or time.time() < deadline - 2:
        try:
            pos_g, rad_g = _growth_pass(best_pos, best_radii, deadline=deadline)
            if _is_valid(pos_g, rad_g) and np.sum(rad_g) > best_score:
                best_score = np.sum(rad_g)
                best_pos, best_radii = pos_g, rad_g
        except Exception:
            pass
    
    return best_pos, best_radii


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Phase 1: Multi-start optimization
    strategies = [
        ('grid', _make_grid),
        ('hexagonal', _make_hexagonal),
        ('center_ring', _make_center_ring),
        ('random', _make_random),
    ]
    
    for strat_name, strat_func in strategies:
        if time.time() > deadline - 60:
            break
        
        for seed in range(8):
            if time.time() > deadline - 55:
                break
            
            pos, radii = strat_func(seed)
            
            # Initial growth
            pos, radii = _growth_pass(pos, radii, deadline=deadline - 50)
            
            # Force-directed anneal
            if time.time() < deadline - 40:
                pos_fd, radii_fd = _force_directed_anneal(pos, radii, n_iter=800, deadline=deadline - 35)
                if _is_valid(pos_fd, radii_fd) and np.sum(radii_fd) > best_score:
                    best_score = np.sum(radii_fd)
                    best_pos = pos_fd.copy()
                    best_radii = radii_fd.copy()
            
            # Polish
            if time.time() < deadline - 30:
                pos_pol, radii_pol = _polish(pos_fd, radii_fd, deadline=deadline - 25)
                if _is_valid(pos_pol, radii_pol) and np.sum(radii_pol) > best_score:
                    best_score = np.sum(radii_pol)
                    best_pos = pos_pol.copy()
                    best_radii = radii_pol.copy()
    
    # Phase 2: Extended annealing from best
    if best_pos is not None and time.time() < deadline - 30:
        try:
            pos_fd2, radii_fd2 = _force_directed_anneal(best_pos, best_radii, n_iter=3000, deadline=deadline - 25)
            if _is_valid(pos_fd2, radii_fd2) and np.sum(radii_fd2) > best_score:
                best_score = np.sum(radii_fd2)
                best_pos = pos_fd2.copy()
                best_radii = radii_fd2.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_pos is not None and time.time() < deadline - 10:
        try:
            pos_final, radii_final = _polish(best_pos, best_radii, deadline=deadline - 5)
            if _is_valid(pos_final, radii_final) and np.sum(radii_final) > best_score:
                best_score = np.sum(radii_final)
                best_pos = pos_final.copy()
                best_radii = radii_final.copy()
        except Exception:
            pass
    
    # Phase 4: Final growth squeeze
    if best_pos is not None and time.time() < deadline - 3:
        try:
            pos_g, rad_g = _growth_pass(best_pos, best_radii, deadline=deadline - 1)
            if _is_valid(pos_g, rad_g) and np.sum(rad_g) > best_score:
                best_score = np.sum(rad_g)
                best_pos = pos_g.copy()
                best_radii = rad_g.copy()
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
