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
    for _ in range(300):
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

def _pbd_solve(pos, radii, max_iter=2000, dt=0.005, deadline=None):
    """
    Position-Based Dynamics solver:
    - Uniform attraction toward center (0.5, 0.5)
    - Iterative constraint projection for non-overlap and boundaries
    - Gradually increase radii to fill space
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Precompute pairs for efficiency
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Gradually increase target radii
        growth_factor = 1.0 + 0.0002 * (it % 100)
        
        # Apply gentle attraction toward center
        for i in range(n):
            dx = 0.5 - pos[i, 0]
            dy = 0.5 - pos[i, 1]
            pos[i, 0] += dx * 0.01 * dt
            pos[i, 1] += dy * 0.01 * dt
        
        # Iterative constraint projection
        for proj_iter in range(50):
            # Boundary constraints
            for i in range(n):
                r = radii[i]
                if pos[i, 0] < r:
                    pos[i, 0] = r
                elif pos[i, 0] > 1.0 - r:
                    pos[i, 0] = 1.0 - r
                if pos[i, 1] < r:
                    pos[i, 1] = r
                elif pos[i, 1] > 1.0 - r:
                    pos[i, 1] = 1.0 - r
            
            # Pairwise non-overlap constraints
            for i, j in pairs:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        # Random perturbation
                        pos[i, 0] += 0.001 * (np.random.random() - 0.5)
                        pos[i, 1] += 0.001 * (np.random.random() - 0.5)
                        pos[j, 0] -= 0.001 * (np.random.random() - 0.5)
                        pos[j, 1] -= 0.001 * (np.random.random() - 0.5)
                    else:
                        overlap = min_dist - dist
                        corr = overlap * 0.5
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] += nx * corr
                        pos[i, 1] += ny * corr
                        pos[j, 0] -= nx * corr
                        pos[j, 1] -= ny * corr
        
        # Try to grow radii slightly
        for i in range(n):
            # Max possible radius given position
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - radii[j])
            if max_r > radii[i]:
                radii[i] = min(radii[i] + 0.0005, max_r * 0.999)
        
        # Check validity and update best
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Early stop if converged
        if it % 200 == 199:
            if best_score <= np.sum(radii) + 1e-8:
                break
    
    return best_pos, best_radii

def _constrained_optimize(pos, radii, deadline):
    """
    Direct constraint-based optimization using SLSQP with explicit constraints.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    def objective(x):
        return -np.sum(x[2*N:])
    
    constraints = []
    # Non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) -
                (x[2*N+i] + x[2*N+j])})
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
    
    bounds = [(0.001, 0.999)] * (2*N) + [(0.001, 0.5)] * N
    
    try:
        result = minimize(objective, x, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success and result.fun < -np.sum(x[2*N:]) - 1e-12:
            return result.x[:2*N].reshape(N, 2), np.maximum(result.x[2*N:], 1e-4)
    except Exception:
        pass
    
    return pos, radii

def _polish(pos, radii, deadline):
    """Multi-method polishing."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Stage 1: PBD relaxation
    if time.time() < deadline - 15:
        pos_pbd, radii_pbd = _pbd_solve(pos, radii, max_iter=1500, deadline=deadline - 15)
        pos_pp, radii_pp = _post_process(pos_pbd, radii_pbd)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii):
            pos, radii = pos_pp, radii_pp
    
    # Stage 2: Constrained optimization
    if time.time() < deadline - 10:
        pos_co, radii_co = _constrained_optimize(pos, radii, deadline)
        pos_pp, radii_pp = _post_process(pos_co, radii_co)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii):
            pos, radii = pos_pp, radii_pp
    
    # Stage 3: Final PBD pass
    if time.time() < deadline - 5:
        pos_pbd2, radii_pbd2 = _pbd_solve(pos, radii, max_iter=1000, deadline=deadline - 5)
        pos_pp, radii_pp = _post_process(pos_pbd2, radii_pbd2)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii):
            pos, radii = pos_pp, radii_pp
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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
    
    # Process each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        # PBD solve
        if time.time() < deadline - 20:
            pos_pbd, radii_pbd = _pbd_solve(pos_init, rad_init, max_iter=2000, deadline=deadline - 20)
            pos_pp, radii_pp = _post_process(pos_pbd, radii_pbd)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Constrained optimization
        if time.time() < deadline - 15:
            pos_co, radii_co = _constrained_optimize(pos_init, rad_init, deadline)
            pos_pp, radii_pp = _post_process(pos_co, radii_co)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Final polish
    if time.time() < deadline - 15:
        pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
        pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
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
