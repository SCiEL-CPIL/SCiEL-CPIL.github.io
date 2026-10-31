"""Circle packing in unit square - 26 circles maximizing sum of radii."""
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


def _make_initial_config(strategy, seed=42):
    """Create an initial configuration based on strategy."""
    rng = np.random.RandomState(seed)
    
    if strategy == 'grid':
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                    radii[idx] = 0.5 / side - 0.005
        pos[25] = [0.5, 0.05]
        radii[25] = 0.04
        return pos, radii
    
    elif strategy == 'hexagonal':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
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
                if idx < N:
                    pos[idx] = [x, y]
                    radii[idx] = r
        return pos, radii
    
    elif strategy == 'center_ring':
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
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
    
    elif strategy == 'random':
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.1, size=N)
        return pos, radii
    
    else:
        pos, radii = _make_initial_config('grid', seed)
        pos += rng.normal(0, 0.02, size=pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
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
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = (dist / (min_dist + 1e-12)) * 0.98
                    radii[i] = max(0.001, radii[i] * scale)
                    radii[j] = max(0.001, radii[j] * scale)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _smooth_objective(x, beta=100.0, smooth_eps=0.01):
    """Smooth objective with soft penalties for boundary and overlap violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    penalty = 0.0
    
    # Boundary penalty
    for i in range(n):
        for coord in range(2):
            d_left = pos[i, coord] - radii[i]
            if d_left < 0:
                penalty += d_left * d_left
            d_right = (1.0 - pos[i, coord]) - radii[i]
            if d_right < 0:
                penalty += d_right * d_right
    
    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
                violation = min_dist - dist
                if violation > 0:
                    penalty += violation * violation
    
    return -np.sum(radii) + beta * penalty


def _constraint_functions(x):
    """Return list of constraint functions for trust-constr."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    constraints = []
    
    # Boundary constraints: pos[i] - r[i] >= 0 and 1 - pos[i] - r[i] >= 0
    for i in range(n):
        for coord in range(2):
            # Left/bottom: pos[i,coord] - radii[i] >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, c=coord: x[2*i + c] - x[2*n + i]
            })
            # Right/top: 1 - pos[i,coord] - radii[i] >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, c=coord: 1.0 - x[2*i + c] - x[2*n + i]
            })
    
    # Overlap constraints: ||pos[i] - pos[j]|| - (r[i] + r[j]) >= 0
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt(
                    (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2
                ) - (x[2*n + i] + x[2*n + j])
            })
    
    return constraints


def _run_trust_constr(x0, deadline, max_iter=200):
    """Run trust-constr optimization with hard constraints."""
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))
    
    constraints = _constraint_functions(x0)
    
    try:
        result = minimize(
            fun=lambda x: -np.sum(x[2*N:]),  # minimize negative sum of radii
            x0=x0,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if result.success or result.fun < -np.sum(x0[2*N:]):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return None, None


def _run_smooth_optimization(x0, deadline):
    """Run L-BFGS-B with smooth penalty objective."""
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))
    
    try:
        result = minimize(
            _smooth_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < _smooth_objective(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return None, None


def _run_differential_evolution(deadline, max_iter=50):
    """Run differential evolution for global search."""
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.25))
    
    try:
        result = differential_evolution(
            _smooth_objective,
            bounds,
            maxiter=max_iter,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            update_interval=1
        )
        
        if result.success:
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return None, None


def _force_directed_anneal(pos, radii, n_iter=500, deadline=None):
    """Force-directed annealing to explore topology and improve packing."""
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    vel = np.zeros_like(pos)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp = 0.02 * np.exp(-0.002 * it)
        forces = np.zeros_like(pos)
        
        for i in range(N):
            if pos[i, 0] < radii[i] + 0.01:
                forces[i, 0] += 5.0 * (radii[i] + 0.01 - pos[i, 0])
            if pos[i, 0] > 1.0 - radii[i] - 0.01:
                forces[i, 0] -= 5.0 * (pos[i, 0] - (1.0 - radii[i] - 0.01))
            if pos[i, 1] < radii[i] + 0.01:
                forces[i, 1] += 5.0 * (radii[i] + 0.01 - pos[i, 1])
            if pos[i, 1] > 1.0 - radii[i] - 0.01:
                forces[i, 1] -= 5.0 * (pos[i, 1] - (1.0 - radii[i] - 0.01))
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < min_dist * 1.5 and dist > 1e-6:
                    force_mag = 0.5 * (min_dist * 1.5 - dist)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        vel = 0.9 * vel + 0.1 * forces
        pos += vel * 0.01 + np.random.normal(0, temp, size=pos.shape)
        
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        for i in range(N):
            can_grow = True
            max_grow = 0.001
            max_grow = min(max_grow, pos[i, 0] - radii[i], 1.0 - pos[i, 0] - radii[i],
                          pos[i, 1] - radii[i], 1.0 - pos[i, 1] - radii[i])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = dist - radii[j]
                if available < max_grow:
                    max_grow = available
            
            if max_grow > 0.0001:
                radii[i] += max_grow * 0.5
        
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _polish(pos, radii, deadline=None):
    """High-precision polish using trust-constr with hard constraints."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Try trust-constr first
    result_pos, result_radii = _run_trust_constr(x0, deadline, max_iter=100)
    if result_pos is not None:
        score = np.sum(result_radii)
        if score > np.sum(radii):
            return result_pos, result_radii
    
    # Fall back to smooth optimization
    result_pos, result_radii = _run_smooth_optimization(x0, deadline)
    if result_pos is not None:
        score = np.sum(result_radii)
        if score > np.sum(radii):
            return result_pos, result_radii
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = ['grid', 'hexagonal', 'center_ring', 'random']
    
    # Phase 1: Multi-start with smooth optimization
    for strategy in strategies:
        if time.time() > deadline - 80:
            break
        
        for seed in range(3):
            if time.time() > deadline - 75:
                break
            
            pos, radii = _make_initial_config(strategy, seed)
            x0 = np.concatenate([pos.flatten(), radii])
            
            # Try smooth optimization
            result_pos, result_radii = _run_smooth_optimization(x0, deadline)
            if result_pos is not None:
                score = np.sum(result_radii)
                if score > best_score:
                    best_score = score
                    best_pos = result_pos.copy()
                    best_radii = result_radii.copy()
    
    # Phase 2: Trust-constr from best result
    if best_pos is not None and time.time() < deadline - 40:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        result_pos, result_radii = _run_trust_constr(x0, deadline - 35, max_iter=150)
        if result_pos is not None:
            score = np.sum(result_radii)
            if score > best_score:
                best_score = score
                best_pos = result_pos.copy()
                best_radii = result_radii.copy()
    
    # Phase 3: Force-directed annealing
    if best_pos is not None and time.time() < deadline - 30:
        try:
            pos_fd, radii_fd = _force_directed_anneal(best_pos, best_radii, n_iter=600, deadline=deadline - 25)
            score_fd = np.sum(radii_fd)
            if _is_valid(pos_fd, radii_fd) and score_fd > best_score:
                best_score = score_fd
                best_pos = pos_fd.copy()
                best_radii = radii_fd.copy()
        except Exception:
            pass
    
    # Phase 4: Differential evolution for global search
    if time.time() < deadline - 25:
        result_pos, result_radii = _run_differential_evolution(deadline - 20, max_iter=30)
        if result_pos is not None:
            score = np.sum(result_radii)
            if score > best_score:
                best_score = score
                best_pos = result_pos.copy()
                best_radii = result_radii.copy()
    
    # Phase 5: Final polish
    if best_pos is not None and time.time() < deadline - 5:
        pos_pol, radii_pol = _polish(best_pos, best_radii, deadline=deadline - 2)
        score_pol = np.sum(radii_pol)
        if _is_valid(pos_pol, radii_pol) and score_pol > best_score:
            best_score = score_pol
            best_pos = pos_pol.copy()
            best_radii = radii_pol.copy()
    
    # Fallback
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
