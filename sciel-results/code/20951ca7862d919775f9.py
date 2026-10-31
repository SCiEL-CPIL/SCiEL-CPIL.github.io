"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay, KDTree

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
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    for _ in range(500):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if radii[i] <= radii[j]:
                        max_i = dist - radii[j]
                        if max_i < radii[i]:
                            radii[i] = max(max_i * 0.99999, 1e-6)
                            valid = False
                    else:
                        max_j = dist - radii[i]
                        if max_j < radii[j]:
                            radii[j] = max(max_j * 0.99999, 1e-6)
                            valid = False
        if valid:
            break
    return pos, radii

def _contact_graph(pos, radii, eps=1e-4):
    """Find pairs of circles that are in contact (distance ≈ r_i + r_j)."""
    n = len(radii)
    contacts = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            adaptive_eps = max(eps, 0.02 * (radii[i] + radii[j]))
            if abs(dist - (radii[i] + radii[j])) < adaptive_eps:
                contacts.add((i, j))
    return contacts

def _build_delaunay(pos):
    """Build Delaunay triangulation of the circle centers."""
    try:
        if len(pos) < 3:
            return np.empty((0, 3), dtype=int)
        tri = Delaunay(pos)
        return tri.simplices
    except Exception:
        return np.empty((0, 3), dtype=int)

def _solve_qp_radii(pos, contacts, boundary_tol=1e-8):
    """
    Solve for maximum radii given fixed positions and contact constraints.
    Uses linear programming: maximize sum(r_i) subject to:
      r_i + r_j <= d_ij for contact pairs (i,j)
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for boundary
      r_i >= 0
    """
    n = len(pos)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        A_ub.append(np.eye(n)[i])
        b_ub.append(max(max_r, 0.0))
    
    for (i, j) in contacts:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        A_ub.append(row)
        b_ub.append(d)
    
    if not A_ub:
        return np.zeros(n)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    
    radii = np.zeros(n)
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for (a, b) in contacts:
            if a == i:
                j = b
            elif b == i:
                j = a
            else:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, d - radii[j])
        radii[i] = max(max_r, 0.0)
    return radii

def _expand_positions(pos, radii, n_iter=200, deadline=None):
    """
    Iteratively expand circles and adjust positions to maintain packing.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        contacts = _contact_graph(pos, radii, eps=1e-3)
        new_radii = _solve_qp_radii(pos, contacts)
        score = np.sum(new_radii)
        
        moved = False
        for (i, j) in contacts:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            push = 0.001
            pos[i, 0] += push * dx / dist
            pos[i, 1] += push * dy / dist
            pos[j, 0] -= push * dx / dist
            pos[j, 1] -= push * dy / dist
            moved = True
        
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        if not moved:
            break
        
        contacts = _contact_graph(pos, radii, eps=1e-3)
        new_radii = _solve_qp_radii(pos, contacts)
        score = np.sum(new_radii)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = new_radii.copy()
    
    return best_pos, best_radii

def _initial_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.095
                k += 1
    if k < N:
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        k += 1
    
    pos, radii = _post_process(pos, radii)
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

def _initial_corner(seed=7):
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
            for _ in range(200):
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

def _softplus(z, beta=1.0):
    """Numerically stable softplus."""
    z = np.asarray(z, dtype=float)
    return beta * np.logaddexp(0.0, z / beta)

def _objective(x, penalty_weight=100.0):
    """Penalty-based objective with smooth softplus penalties."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = min_dist_sq - dist_sq
    overlap_pen = np.sum(_softplus(viol[mask], beta=0.01))
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0):
    """Fully smooth objective using softplus for all constraints."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    margin_x = np.minimum(pos[:, 0] - radii, 1.0 - pos[:, 0] - radii)
    margin_y = np.minimum(pos[:, 1] - radii, 1.0 - pos[:, 1] - radii)
    boundary_pen = np.sum(_softplus(-margin_x, beta=0.01)) + np.sum(_softplus(-margin_y, beta=0.01))
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = min_dist_sq - dist_sq
    overlap_pen = np.sum(_softplus(viol[mask], beta=0.01))
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _slsqp_hard(x0, deadline, maxiter=400, ftol=1e-12):
    """SLSQP with hard constraints."""
    x = x0.copy()
    
    def constraint_boundary(x):
        pos = x[:2 * N].reshape(N, 2)
        radii = x[2 * N:]
        return np.concatenate([
            pos[:, 0] - radii,
            1.0 - pos[:, 0] - radii,
            pos[:, 1] - radii,
            1.0 - pos[:, 1] - radii
        ])
    
    def constraint_overlap(x):
        pos = x[:2 * N].reshape(N, 2)
        radii = x[2 * N:]
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        min_dist_sq = min_dist * min_dist
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        return (dist_sq - min_dist_sq)[mask]
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    def neg_sum_radii(x):
        return -np.sum(x[2 * N:])
    
    try:
        result = minimize(neg_sum_radii, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8})
        if result.success or (not result.success and result.fun < neg_sum_radii(x)):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-4)
    return pos_out, radii_out

def _graph_based_optimize(pos, radii, deadline):
    """
    Graph-based optimization:
    1. Build contact graph
    2. Solve LP for optimal radii given positions
    3. Expand positions to create room
    4. Repeat
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for outer in range(30):
        if time.time() > deadline:
            break
        
        contacts = _contact_graph(pos, radii, eps=1e-3)
        new_radii = _solve_qp_radii(pos, contacts)
        new_score = np.sum(new_radii)
        
        if new_score > best_score:
            best_score = new_score
            best_pos = pos.copy()
            best_radii = new_radii.copy()
        
        if len(contacts) > 0:
            for (i, j) in contacts:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    continue
                push = 0.002
                pos[i, 0] += push * dx / dist
                pos[i, 1] += push * dy / dist
                pos[j, 0] -= push * dx / dist
                pos[j, 1] -= push * dy / dist
            
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], 0.01, 0.99)
                pos[i, 1] = np.clip(pos[i, 1], 0.01, 0.99)
        
        radii = new_radii
    
    return best_pos, best_radii

def _simulated_annealing(pos, radii, n_steps=2000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """Simulated annealing with adaptive step size."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 50.0)
    accept_count = 0
    total_count = 0
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        if total_count > 100:
            accept_rate = accept_count / total_count
            if accept_rate > 0.8:
                step_scale = 1.5
            elif accept_rate < 0.2:
                step_scale = 0.7
            else:
                step_scale = 1.0
        else:
            step_scale = 1.0
        
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02 * step_scale)
        dy = rng.normal(0, temp * 0.02 * step_scale)
        dr = rng.normal(0, temp * 0.01 * step_scale)
        
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.4)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 50.0)
        
        delta_e = new_energy - current_energy
        total_count += 1
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
            accept_count += 1
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

def _high_precision_polish(pos, radii, deadline):
    """High-precision local polish: alternating LP radius solve and tiny position adjustments."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for it in range(200):
        if time.time() > deadline:
            break
        
        contacts = _contact_graph(pos, radii, eps=1e-5)
        new_radii = _solve_qp_radii(pos, contacts)
        new_score = np.sum(new_radii)
        
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_pos = pos.copy()
            best_radii = new_radii.copy()
        
        moved_any = False
        for (i, j) in contacts:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            push = 0.0005
            pos[i, 0] += push * dx / dist
            pos[i, 1] += push * dy / dist
            pos[j, 0] -= push * dx / dist
            pos[j, 1] -= push * dy / dist
            moved_any = True
        
        if not moved_any:
            break
        
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
        
        radii = new_radii
    
    return best_pos, best_radii

def _distance_field_init(n_grid=100):
    """
    Distance-field based initialization.
    Compute Euclidean distance to boundary on a grid, then greedily place
    circles at local maxima of the distance field, subtracting their influence.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Create grid
    xs = np.linspace(0.0, 1.0, n_grid)
    ys = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(xs, ys)
    
    # Distance to boundary
    dist = np.minimum(np.minimum(X, 1.0 - X), np.minimum(Y, 1.0 - Y))
    
    # Greedy placement
    placed = []
    for i in range(N):
        # Subtract influence of already-placed circles
        for j, (px, py, pr) in enumerate(placed):
            d_to_c = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
            dist = np.maximum(dist, d_to_c - pr)
        
        # Find local maximum
        # Use a simple approach: find the max in the remaining field
        # But we need to be careful about the grid resolution
        max_idx = np.unravel_index(np.argmax(dist), dist.shape)
        max_x = xs[max_idx[1]]
        max_y = ys[max_idx[0]]
        max_r = dist[max_idx[0], max_idx[1]]
        
        # Refine around the max with finer search
        best_x, best_y, best_r = max_x, max_y, max_r
        for _ in range(500):
            # Random perturbation
            rng = np.random.RandomState(i * 1000 + _ + 42)
            rx = max_x + rng.uniform(-0.05, 0.05)
            ry = max_y + rng.uniform(-0.05, 0.05)
            rx = np.clip(rx, 0.0, 1.0)
            ry = np.clip(ry, 0.0, 1.0)
            
            r_candidate = min(rx, 1.0 - rx, ry, 1.0 - ry)
            for j, (px, py, pr) in enumerate(placed):
                d = np.sqrt((rx - px) ** 2 + (ry - py) ** 2)
                r_candidate = min(r_candidate, d - pr)
            
            if r_candidate > best_r:
                best_r = r_candidate
                best_x = rx
                best_y = ry
        
        if best_r < 0.005:
            # Fallback: try random placement
            rng = np.random.RandomState(i * 100 + 7)
            for _ in range(1000):
                rx = rng.uniform(0.05, 0.95)
                ry = rng.uniform(0.05, 0.95)
                r_candidate = min(rx, 1.0 - rx, ry, 1.0 - ry)
                for j, (px, py, pr) in enumerate(placed):
                    d = np.sqrt((rx - px) ** 2 + (ry - py) ** 2)
                    r_candidate = min(r_candidate, d - pr)
                if r_candidate > best_r:
                    best_r = r_candidate
                    best_x = rx
                    best_y = ry
        
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.95
        placed.append((best_x, best_y, best_r))
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    pos_cb, rad_cb = _initial_corner(seed=7)
    inits.append((pos_cb, rad_cb))
    pos_grid, rad_grid = _initial_grid()
    inits.append((pos_grid, rad_grid))
    pos_df, rad_df = _distance_field_init(n_grid=80)
    inits.append((pos_df, rad_df))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        # Stage 1: Smooth objective L-BFGS-B with annealing penalty
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.5))
        
        for p_w in [100.0, 500.0, 2000.0]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
                if result.fun < _objective_smooth(current_x, p_w) - 1e-12:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2 * N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2 * N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 2: SLSQP hard constraints
        if time.time() < deadline - 15:
            try:
                pos_hc, rad_hc = _slsqp_hard(current_x, deadline, maxiter=400, ftol=1e-14)
                pos_pp, radii_pp = _post_process(pos_hc, rad_hc)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 3: Graph-based optimization
        if time.time() < deadline - 20:
            try:
                pos_gb, rad_gb = _graph_based_optimize(
                    current_x[:2 * N].reshape(N, 2),
                    np.maximum(current_x[2 * N:], 1e-4),
                    deadline - 20
                )
                pos_pp, radii_pp = _post_process(pos_gb, rad_gb)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 4: Simulated annealing
        if time.time() < deadline - 15:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2 * N].reshape(N, 2),
                    np.maximum(current_x[2 * N:], 1e-4),
                    n_steps=2000, t0=0.03, t_end=0.0005,
                    seed=idx * 100, deadline=deadline - 15
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final refinement: high-precision polish
    if time.time() < deadline - 20:
        try:
            pos_hp, rad_hp = _high_precision_polish(best_pos, best_radii, deadline - 20)
            pos_pp, radii_pp = _post_process(pos_hp, rad_hp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final SLSQP polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            pos_hc, rad_hc = _slsqp_hard(
                np.concatenate([best_pos.flatten(), best_radii]), deadline, maxiter=600, ftol=1e-16)
            pos_pp, radii_pp = _post_process(pos_hc, rad_hc)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Another high-precision polish after SLSQP
    if time.time() < deadline - 5:
        try:
            pos_hp, rad_hp = _high_precision_polish(best_pos, best_radii, deadline - 5)
            pos_pp, radii_pp = _post_process(pos_hp, rad_hp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
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
