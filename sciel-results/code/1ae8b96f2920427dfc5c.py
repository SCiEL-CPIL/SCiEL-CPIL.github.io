"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
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

def _post_process(pos, radii, max_iter=300):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(max_iter):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    k = dist / (sum_r + 1e-12) * 0.99999
                    if radii[i] <= radii[j]:
                        radii[i] = max(radii[i] * k, 1e-6)
                    else:
                        radii[j] = max(radii[j] * k, 1e-6)
                    valid = False
        if valid:
            break
    return pos, radii

def _objective(x, penalty_weight=1000.0):
    """Penalty-based objective: minimize -sum(radii) + penalty * violations."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    bv = (np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) +
          np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii)))
    boundary_pen = np.sum(bv * bv)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=1000.0, beta=50.0):
    """Smooth objective with softplus penalties."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    def softplus(v):
        return np.where(v > 20.0/beta, v, np.log1p(np.exp(beta * v)) / beta)
    boundary_pen = (np.sum(softplus(radii - pos[:, 0])) + np.sum(softplus(pos[:, 0] - (1.0 - radii))) +
                    np.sum(softplus(radii - pos[:, 1])) + np.sum(softplus(pos[:, 1] - (1.0 - radii))))
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(softplus(viol_raw[mask]))
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(10000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r:
                    max_r = possible_r
            if max_r > best_r and max_r > 0.003:
                best_r = max_r
                best_x = x
                best_y = y
        if best_r < 0.003:
            best_x, best_y, best_r = 0.5, 0.5, 0.003
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.96
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
        for _ in range(300):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.94
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
            for attempt in range(400):
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
        radii[i] = max(max_r, 0.01) * 0.94
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _contact_graph(pos, radii, tol=1e-3):
    """Build contact graph."""
    n = len(radii)
    contacts = set()
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] + tol:
                contacts.add((i, j))
    return contacts

def _qp_radius_optimization(pos, contacts):
    """Solve LP to maximize sum of radii given fixed positions and contact graph."""
    n = len(pos)
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        A_ub.append(np.eye(n)[i])
        b_ub.append(max_r)
    for (i, j) in contacts:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        row = np.zeros(n)
        row[i] = 1
        row[j] = 1
        A_ub.append(row)
        b_ub.append(dist)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.001, 0.5)] * n
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return np.maximum(result.x, 0.001)
    except Exception:
        pass
    return None

def _force_directed_relax(pos, radii, n_iter=2000, deadline=None):
    """Position-Based Dynamics (PBD) solver with constraint projection.
    
    Uses iterative constraint projection to ensure strict non-overlap and
    boundary adherence at every iteration. Applies a gentle centripetal force
    to pack circles densely, then projects onto the feasible set.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # PBD parameters
    gravity_strength = 0.0002  # gentle pull toward center
    damping = 0.95             # velocity damping
    velocity = np.zeros((n, 2))
    proj_iters = 30            # constraint projection iterations per step
    dt = 1.0                   # fixed timestep
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Apply centripetal force toward center
        for i in range(n):
            cx = 0.5 - pos[i, 0]
            cy = 0.5 - pos[i, 1]
            dist_c = np.sqrt(cx*cx + cy*cy)
            if dist_c > 1e-8:
                # Force proportional to distance from center, but capped
                f = min(gravity_strength, dist_c * 0.01)
                velocity[i, 0] += f * cx / dist_c * dt
                velocity[i, 1] += f * cy / dist_c * dt
        
        # Damp velocity
        velocity *= damping
        
        # Predict positions
        pos_pred = pos + velocity * dt
        
        # Constraint projection: iteratively resolve overlaps and boundaries
        for _ in range(proj_iters):
            # Resolve circle-circle overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos_pred[i, 0] - pos_pred[j, 0]
                    dy = pos_pred[i, 1] - pos_pred[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist - 1e-12:
                        if dist < 1e-10:
                            dx, dy = 1e-5, 1e-5
                            dist = np.sqrt(dx*dx + dy*dy)
                        # Equal weight correction
                        corr = (min_dist - dist) / (2.0 * dist)
                        pos_pred[i, 0] += corr * dx
                        pos_pred[i, 1] += corr * dy
                        pos_pred[j, 0] -= corr * dx
                        pos_pred[j, 1] -= corr * dy
            
            # Resolve boundary violations
            for i in range(n):
                r = radii[i]
                # Left boundary
                if pos_pred[i, 0] < r:
                    pos_pred[i, 0] = r
                # Right boundary
                if pos_pred[i, 0] > 1.0 - r:
                    pos_pred[i, 0] = 1.0 - r
                # Bottom boundary
                if pos_pred[i, 1] < r:
                    pos_pred[i, 1] = r
                # Top boundary
                if pos_pred[i, 1] > 1.0 - r:
                    pos_pred[i, 1] = 1.0 - r
        
        # Update velocity from position change
        velocity = (pos_pred - pos) / dt
        pos = pos_pred
        
        # Try to expand radii every few iterations
        if it % 20 == 0:
            for i in range(n):
                max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                for j in range(n):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - radii[j])
                if max_r > radii[i] + 1e-6:
                    radii[i] = min(max_r * 0.9999, radii[i] + 0.001)
        
        # Check validity and update best
        if it % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _simulated_annealing(pos, radii, n_steps=4000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """Simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 1000.0)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.015)
        dy = rng.normal(0, temp * 0.015)
        dr = rng.normal(0, temp * 0.008)
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.4)
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 1000.0)
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    return best_pos, best_radii

def _try_radius_expansion(pos, radii, max_iter=100, step_size=0.005):
    """Greedy radius expansion."""
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

def _iterative_qp_refinement(pos, radii, deadline, max_rounds=40):
    """Iteratively update contact graph and solve LP for radii, then adjust positions."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for round_idx in range(max_rounds):
        if time.time() > deadline - 5:
            break
        contacts = _contact_graph(pos, radii, tol=1e-3)
        new_radii = _qp_radius_optimization(pos, contacts)
        if new_radii is not None:
            pos, radii = _post_process(pos, new_radii)
            if _is_valid(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        # Adjust positions to accommodate larger radii
        if time.time() > deadline - 5:
            break
        target_radii = radii.copy()
        def _pos_objective(x):
            p = x.reshape(N, 2)
            total_viol = 0.0
            for i in range(N):
                max_r = min(p[i,0], 1-p[i,0], p[i,1], 1-p[i,1])
                if target_radii[i] > max_r:
                    total_viol += (target_radii[i] - max_r)**2
            for i in range(N):
                for j in range(i+1, N):
                    dx = p[i,0] - p[j,0]
                    dy = p[i,1] - p[j,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    need = target_radii[i] + target_radii[j]
                    if dist < need:
                        total_viol += (need - dist)**2
            return total_viol
        x0 = pos.flatten()
        bounds = [(0.01, 0.99)] * (2*N)
        try:
            result = minimize(_pos_objective, x0, method='SLSQP', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-14})
            if result.fun < 1e-10:
                new_pos = result.x.reshape(N, 2)
                contacts2 = _contact_graph(new_pos, target_radii, tol=1e-3)
                new_radii2 = _qp_radius_optimization(new_pos, contacts2)
                if new_radii2 is not None:
                    pos, radii = _post_process(new_pos, new_radii2)
                    if _is_valid(pos, radii):
                        score = np.sum(radii)
                        if score > best_score:
                            best_score = score
                            best_pos = pos.copy()
                            best_radii = radii.copy()
        except Exception:
            pass
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """Multi-stage polishing."""
    x = np.concatenate([pos.flatten(), radii])
    bounds = [(0.01, 0.99)] * (2*N) + [(0.01, 0.5)] * N
    
    # Stage 1: L-BFGS-B with penalty
    if time.time() < deadline - 10:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 5000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    # Stage 2: Smooth objective
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective_smooth, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0, 50.0),
                              options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective_smooth(x, 5000.0, 50.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    # Stage 3: SLSQP final
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                              args=(20000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
            if result.fun < _objective(x, 20000.0) - 1e-14:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    if time.time() < deadline - 2:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iter=100, step_size=0.002)
        pos_out, radii_out = _post_process(pos_out, radii_out)
    return pos_out, radii_out

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
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = [(0.01, 0.99)] * (2*N) + [(0.01, 0.5)] * N
    penalty_stages = [50.0, 500.0, 2000.0, 5000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # L-BFGS-B with increasing penalty
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Smooth objective
        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 50.0),
                                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 50.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # PBD relaxation (primary method)
        if time.time() < deadline - 12:
            try:
                pos_fd, rad_fd = _force_directed_relax(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_iter=3000, deadline=deadline
                )
                pos_exp, rad_exp = _try_radius_expansion(pos_fd, rad_fd, max_iter=80, step_size=0.003)
                pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Simulated annealing
        if time.time() < deadline - 10:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=3000, t0=0.03, t_end=0.0005,
                    seed=idx * 100, deadline=deadline
                )
                pos_exp, rad_exp = _try_radius_expansion(pos_sa, rad_sa, max_iter=50, step_size=0.005)
                pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # QP-based refinement on best solution
    if time.time() < deadline - 20:
        try:
            pos_qp, rad_qp = _iterative_qp_refinement(best_pos, best_radii, deadline, max_rounds=30)
            if _is_valid(pos_qp, rad_qp):
                score = np.sum(rad_qp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_qp.copy()
                    best_radii = rad_qp.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 15:
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
        
        # PBD-based final refinement: use PBD to settle, then polish
        if time.time() < deadline - 8:
            try:
                pos_pbd_final, rad_pbd_final = _force_directed_relax(
                    best_pos.copy(), best_radii.copy(),
                    n_iter=2000, deadline=deadline
                )
                pos_exp_f, rad_exp_f = _try_radius_expansion(pos_pbd_final, rad_pbd_final, max_iter=100, step_size=0.002)
                pos_pol_f, rad_pol_f = _polish(pos_exp_f, rad_exp_f, deadline)
                pos_pp_f, radii_pp_f = _post_process(pos_pol_f, rad_pol_f)
                score_f = np.sum(radii_pp_f)
                if _is_valid(pos_pp_f, radii_pp_f) and score_f > best_score:
                    best_score = score_f
                    best_pos = pos_pp_f.copy()
                    best_radii = radii_pp_f.copy()
            except Exception:
                pass
        
        # Secondary polish from perturbed initial
        if time.time() < deadline - 6:
            rng = np.random.RandomState(42)
            pos_pert = best_pos + rng.normal(0, 0.005, (N, 2))
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            radii_pert = best_radii.copy()
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
        
        # Tertiary: PBD from slightly perturbed state
        if time.time() < deadline - 4:
            rng = np.random.RandomState(123)
            pos_pert3 = best_pos + rng.normal(0, 0.008, (N, 2))
            pos_pert3 = np.clip(pos_pert3, 0.01, 0.99)
            radii_pert3 = best_radii * 0.97
            try:
                pos_pbd3, rad_pbd3 = _force_directed_relax(pos_pert3, radii_pert3, n_iter=1500, deadline=deadline)
                pos_exp3, rad_exp3 = _try_radius_expansion(pos_pbd3, rad_pbd3, max_iter=80, step_size=0.002)
                pos_pol3, rad_pol3 = _polish(pos_exp3, rad_exp3, deadline)
                pos_pp3, radii_pp3 = _post_process(pos_pol3, rad_pol3)
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
    if time.time() < deadline - 1:
        pos_exp, rad_exp = _try_radius_expansion(best_pos, best_radii, max_iter=100, step_size=0.001)
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
