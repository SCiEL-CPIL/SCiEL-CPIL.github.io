"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.sparse import csr_matrix

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

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
    
    for _ in range(100):
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

def _lp_radius_optimize(pos):
    """
    Solve for maximum sum of radii given fixed positions using Linear Programming.
    Maximize sum(r_i) subject to:
      r_i + r_j <= dist(i,j) for all i<j
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
      r_i >= 0
    """
    n = len(pos)
    c = -np.ones(n)  # minimize -sum(r)
    
    A_rows = []
    b_vals = []
    
    # Pairwise distance constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
    
    # Wall constraints: r_i <= wall_dist_i
    for i in range(n):
        w = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(max(w, 0.0))
    
    if not A_rows:
        return np.zeros(n)
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 0.5)] * n, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    
    return np.zeros(n)

def _get_tight_neighbors(pos, radii, tol=1e-4):
    """
    Identify which constraints are tight (active) for each circle.
    Returns a list of lists, where tight_neighbors[i] contains indices of circles
    that are limiting the radius of circle i.
    """
    n = len(radii)
    tight_neighbors = [[] for _ in range(n)]
    
    # Wall constraints
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        # Check walls
        if abs(x - r) < tol: tight_neighbors[i].append(-1) # Left wall
        if abs((1-x) - r) < tol: tight_neighbors[i].append(-2) # Right wall
        if abs(y - r) < tol: tight_neighbors[i].append(-3) # Bottom wall
        if abs((1-y) - r) < tol: tight_neighbors[i].append(-4) # Top wall
        
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if abs(dist - (radii[i] + radii[j])) < tol:
                tight_neighbors[i].append(j)
                tight_neighbors[j].append(i)
                
    return tight_neighbors

def _jitter_and_expand(pos, radii, deadline, seed=0):
    """
    Jitter-and-expand heuristic with active constraint awareness.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Initial LP solve
    lp_radii = _lp_radius_optimize(pos)
    if np.sum(lp_radii) > best_score:
        best_score = np.sum(lp_radii)
        best_radii = lp_radii.copy()
        radii = lp_radii.copy()
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 150
    jitter_scale = 0.01
    temperature = 0.01
    
    while time.time() < deadline:
        iteration += 1
        
        # Solve LP for current positions
        lp_radii = _lp_radius_optimize(pos)
        current_score = np.sum(lp_radii)
        
        if current_score > best_score + 1e-10:
            best_score = current_score
            best_pos = pos.copy()
            best_radii = lp_radii.copy()
            no_improve_count = 0
            temperature = max(temperature * 0.99, 0.001) # Cool down on success
        else:
            no_improve_count += 1
            temperature = max(temperature * 1.01, 0.01) # Heat up on stagnation
        
        if no_improve_count > max_no_improve:
            # Restart with larger jitter and reset position slightly
            jitter_scale *= 1.5
            if jitter_scale > 0.05:
                jitter_scale = 0.01
                pos = best_pos + rng.normal(0, 0.005, (n, 2))
                pos = np.clip(pos, 0.01, 0.99)
            no_improve_count = 0
            continue
        
        # Get tight neighbors to guide jitter
        tight_neighbors = _get_tight_neighbors(pos, lp_radii, tol=1e-4)
        
        # Select a circle to jitter. Prefer circles with many tight constraints (stuck)
        stuck_counts = np.array([len(t) for t in tight_neighbors])
        if np.any(stuck_counts > 0):
            # Weighted choice based on stuckness
            weights = stuck_counts + 1
            i = rng.choice(n, p=weights / weights.sum())
        else:
            i = rng.randint(n)
        
        # Determine jitter direction
        # If there are tight neighbors, try to move away from them
        direction = np.zeros(2)
        if tight_neighbors[i]:
            for neighbor in tight_neighbors[i]:
                if neighbor < 0:
                    # Wall constraint
                    if neighbor == -1: direction[0] += 1.0 # Move right
                    if neighbor == -2: direction[0] -= 1.0 # Move left
                    if neighbor == -3: direction[1] += 1.0 # Move up
                    if neighbor == -4: direction[1] -= 1.0 # Move down
                else:
                    # Circle constraint
                    dx = pos[i, 0] - pos[neighbor, 0]
                    dy = pos[i, 1] - pos[neighbor, 1]
                    norm = np.sqrt(dx*dx + dy*dy)
                    if norm > 1e-9:
                        direction[0] += dx / norm
                        direction[1] += dy / norm
            
            norm_dir = np.sqrt(direction[0]**2 + direction[1]**2)
            if norm_dir > 1e-9:
                direction /= norm_dir
            else:
                angle = rng.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Add some randomness
        noise_angle = rng.uniform(-np.pi/4, np.pi/4)
        cos_n, sin_n = np.cos(noise_angle), np.sin(noise_angle)
        new_dir = np.array([direction[0]*cos_n - direction[1]*sin_n,
                            direction[0]*sin_n + direction[1]*cos_n])
        
        magnitude = jitter_scale * rng.uniform(0.5, 1.5)
        
        old_pos_i = pos[i].copy()
        pos[i, 0] += magnitude * new_dir[0]
        pos[i, 1] += magnitude * new_dir[1]
        pos[i] = np.clip(pos[i], 0.01, 0.99)
        
        # Solve LP for new position
        new_lp_radii = _lp_radius_optimize(pos)
        new_score = np.sum(new_lp_radii)
        
        # Acceptance criterion: Simulated Annealing
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(-delta / max(temperature, 1e-6)):
            radii = new_lp_radii.copy()
        else:
            # Reject: revert position
            pos[i] = old_pos_i
    
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # First: LP-based radius optimization
    if time.time() < deadline - 5:
        try:
            lp_radii = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Jitter-and-expand for local refinement
    if time.time() < deadline - 5:
        try:
            pos_je, rad_je = _jitter_and_expand(best_pos, best_radii, deadline - 5, seed=999)
            pos_pp, radii_pp = _post_process(pos_je, rad_je)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final LP pass
    if time.time() < deadline - 2:
        try:
            lp_radii_final = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_radii_final)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    return best_pos, best_radii

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
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    # For each initialization, run a quick LP + jitter-and-expand cycle
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        # Quick LP to get maximal radii for this initial configuration
        try:
            lp_rad = _lp_radius_optimize(pos_init)
            pos_pp, radii_pp = _post_process(pos_init, lp_rad)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Short jitter-and-expand from this initialization
        if time.time() < deadline - 35:
            try:
                pos_je, rad_je = _jitter_and_expand(pos_init, rad_init, deadline - 35, seed=idx * 100 + 1)
                pos_pp, radii_pp = _post_process(pos_je, rad_je)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Main optimization: longer jitter-and-expand from the best configuration
    if time.time() < deadline - 20:
        try:
            pos_je, rad_je = _jitter_and_expand(best_pos, best_radii, deadline - 20, seed=42)
            pos_pp, radii_pp = _post_process(pos_je, rad_je)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Polish: Jitter-and-expand + LP
    if time.time() < deadline - 8:
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
    
    # Final LP radius optimization on best positions
    if time.time() < deadline - 4:
        try:
            lp_rad = _lp_radius_optimize(best_pos)
            pos_pp, radii_pp = _post_process(best_pos, lp_rad)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to a known valid configuration
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
