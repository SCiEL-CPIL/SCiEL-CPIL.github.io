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


def _penalty(pos, radii):
    """Total constraint violation (sum of squared violations)."""
    n = len(radii)
    pen = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            pen += (radii[i] - pos[i, 0])**2
        if pos[i, 0] > 1.0 - radii[i]:
            pen += (pos[i, 0] - (1.0 - radii[i]))**2
        if pos[i, 1] < radii[i]:
            pen += (radii[i] - pos[i, 1])**2
        if pos[i, 1] > 1.0 - radii[i]:
            pen += (pos[i, 1] - (1.0 - radii[i]))**2
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = radii[i] + radii[j]
            if dist < min_d:
                pen += (min_d - dist)**2
    return pen


def _score(pos, radii):
    """Sum of radii (the objective to maximize)."""
    return float(np.sum(radii))


def _fix_boundaries(pos, radii):
    """Shrink radii to fit within boundaries."""
    radii = radii.copy()
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max(0.001, max_r * 0.999)
    return radii


def _resolve_overlaps(pos, radii, max_iter=200):
    """Iteratively shrink radii to resolve overlaps."""
    radii = radii.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = radii[i] + radii[j]
                if dist < min_d:
                    scale = (dist / min_d) * 0.999 if min_d > 0 else 0.999
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        radii = _fix_boundaries(pos, radii)
        if not changed:
            break
    return radii


def _make_grid(seed=42):
    """5x5 grid with one extra circle."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.003
    # 26th circle in a gap
    pos[25] = [0.5, 0.02]
    radii[25] = 0.015
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal packing pattern."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.095
    idx = 0
    for row in range(6):
        y = 0.1 + row * r * np.sqrt(3)
        if y > 0.9:
            break
        offset = r if row % 2 == 0 else 2 * r
        for col in range(5):
            x = offset + col * 2 * r
            if x > 0.9:
                break
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    # Fill remaining
    while idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.05
        idx += 1
    return pos, radii


def _make_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.08, size=N)
    return pos, radii


def _make_mixed(seed=42):
    """Mixed: some grid, some random."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 16 grid circles
    side = 4
    idx = 0
    for i in range(4):
        for j in range(4):
            pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
            radii[idx] = 0.5 / side - 0.005
            idx += 1
    # 10 random circles
    pos[idx:idx+10] = rng.uniform(0.05, 0.95, size=(10, 2))
    radii[idx:idx+10] = rng.uniform(0.03, 0.07, size=10)
    return pos, radii


def _force_directed_anneal(pos, radii, time_budget=30.0, seed=42):
    """Simulated annealing with force-directed moves."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_iter = 50000
    
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Random move: perturb positions and/or radii
        move_type = rng.random()
        if move_type < 0.5:
            # Move one circle
            i = rng.randint(N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            pos[i, 0] = np.clip(pos[i, 0] + dx, 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        elif move_type < 0.8:
            # Change radius of one circle
            i = rng.randint(N)
            dr = rng.normal(0, 0.005)
            radii[i] = np.clip(radii[i] + dr, 0.005, 0.3)
        else:
            # Move two circles
            i, j = rng.choice(N, size=2, replace=False)
            pos[i, 0] = np.clip(pos[i, 0] + rng.normal(0, 0.02), 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1] + rng.normal(0, 0.02), 0.01, 0.99)
            pos[j, 0] = np.clip(pos[j, 0] + rng.normal(0, 0.02), 0.01, 0.99)
            pos[j, 1] = np.clip(pos[j, 1] + rng.normal(0, 0.02), 0.01, 0.99)
        
        # Fix boundaries
        radii = _fix_boundaries(pos, radii)
        
        # Compute new score with penalty
        pen = _penalty(pos, radii)
        new_score = _score(pos, radii) - 100.0 * pen
        old_score = _score(best_pos, best_radii) - 100.0 * _penalty(best_pos, best_radii)
        
        delta = new_score - old_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            if new_score > best_score - 100.0 * _penalty(pos, radii):
                # Accept
                pos = pos.copy()
                radii = radii.copy()
                if pen < 1e-12:
                    s = _score(pos, radii)
                    if s > best_score:
                        best_score = s
                        best_pos = pos.copy()
                        best_radii = radii.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pos, best_radii, best_score


def _gradient_polish(pos, radii, time_budget=15.0):
    """Gradient-based polishing to fine-tune positions and radii."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Penalty
        pen = 0.0
        for i in range(N):
            if p[i, 0] < r[i]:
                pen += (r[i] - p[i, 0])**2
            if p[i, 0] > 1.0 - r[i]:
                pen += (p[i, 0] - (1.0 - r[i]))**2
            if p[i, 1] < r[i]:
                pen += (r[i] - p[i, 1])**2
            if p[i, 1] > 1.0 - r[i]:
                pen += (p[i, 1] - (1.0 - r[i]))**2
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if dist < min_d:
                    pen += (min_d - dist)**2
        return -np.sum(r) + 50.0 * pen
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.3))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _slsqp_polish(pos, radii, time_budget=10.0):
    """SLSQP polishing."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        pen = 0.0
        for i in range(N):
            if p[i, 0] < r[i]:
                pen += (r[i] - p[i, 0])**2
            if p[i, 0] > 1.0 - r[i]:
                pen += (p[i, 0] - (1.0 - r[i]))**2
            if p[i, 1] < r[i]:
                pen += (r[i] - p[i, 1])**2
            if p[i, 1] > 1.0 - r[i]:
                pen += (p[i, 1] - (1.0 - r[i]))**2
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if dist < min_d:
                    pen += (min_d - dist)**2
        return -np.sum(r) + 100.0 * pen
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.3))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = [
        ('grid', 42), ('grid', 123), ('grid', 7),
        ('hexagonal', 42), ('hexagonal', 99),
        ('random', 42), ('random', 123), ('random', 7), ('random', 55),
        ('mixed', 42), ('mixed', 123), ('mixed', 7),
    ]
    
    for strat_name, seed in strategies:
        if time.time() > deadline - 25:
            break
        
        if strat_name == 'grid':
            pos, radii = _make_grid(seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(seed)
        elif strat_name == 'random':
            pos, radii = _make_random(seed)
        elif strat_name == 'mixed':
            pos, radii = _make_mixed(seed)
        else:
            continue
        
        # Force-directed annealing
        remaining = deadline - time.time()
        if remaining < 10:
            break
        anneal_budget = min(20.0, remaining * 0.5)
        pos_a, radii_a, score_a = _force_directed_anneal(pos, radii, time_budget=anneal_budget, seed=seed)
        
        if score_a > best_score:
            best_score = score_a
            best_pos = pos_a.copy()
            best_radii = radii_a.copy()
        
        # Gradient polish on annealed result
        remaining = deadline - time.time()
        if remaining < 15:
            continue
        polish_budget = min(12.0, remaining * 0.4)
        pos_p, radii_p, score_p = _gradient_polish(pos_a, radii_a, time_budget=polish_budget)
        
        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p.copy()
            best_radii = radii_p.copy()
        
        # SLSQP polish
        remaining = deadline - time.time()
        if remaining < 10:
            continue
        slsqp_budget = min(8.0, remaining * 0.3)
        pos_s, radii_s, score_s = _slsqp_polish(pos_p, radii_p, time_budget=slsqp_budget)
        
        if score_s > best_score:
            best_score = score_s
            best_pos = pos_s.copy()
            best_radii = radii_s.copy()
    
    # Final polish on best result if time allows
    if best_pos is not None:
        remaining = deadline - time.time()
        if remaining > 15:
            pos_f, radii_f, score_f = _gradient_polish(best_pos, best_radii, time_budget=min(12.0, remaining - 5))
            if score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()
    
    # Fallback: simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
