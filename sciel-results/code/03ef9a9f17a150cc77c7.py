"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _elapsed(deadline):
    return time.time() - deadline

def _remaining(deadline):
    return deadline - time.time()

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    n = len(pos)
    dx = pos[:, 0:1] - pos[0:1, 0]
    dy = pos[:, 1:2] - pos[0:1, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)

def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))

def _is_valid(pos, eps=1e-9):
    """Check pairwise non-overlap and boundary containment with computed radii."""
    radii = _max_radius_for_positions(pos)
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
            if dist < radii[i] + radii[j] - eps:
                return False
        if (pos[i, 0] < radii[i] - eps or pos[i, 0] > 1.0 - radii[i] + eps or
                pos[i, 1] < radii[i] - eps or pos[i, 1] > 1.0 - radii[i] + eps):
            return False
    return True

def _soft_objective(v, n, beta=50.0):
    """
    Smooth objective: minimize -sum(r) + penalty*(wall + pairwise violations)
    v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    # Wall violations
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta
    
    # Pairwise violations
    total_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = r[i] + r[j] - dist
            total_pen += np.logaddexp(0.0, beta * viol) / beta
    
    return -np.sum(r) + 50.0 * (wall_pen + total_pen)

def _soft_gradient(v, n, beta=50.0):
    """Analytic gradient of _soft_objective."""
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()
    grad = np.zeros(3 * n)
    
    # Wall terms
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    
    for i in range(n):
        s = sig_wall[i]
        if s < 1e-12:
            continue
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        factor = 50.0 * s
        grad[2*n + i] += factor
        if min_idx == 0:
            grad[i] -= factor
        elif min_idx == 1:
            grad[i] += factor
        elif min_idx == 2:
            grad[n + i] -= factor
        else:
            grad[n + i] += factor
    
    # Pairwise terms
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-12:
                continue
            factor = 50.0 * s
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor
    
    # Objective: -sum(r)
    grad[2*n:3*n] -= 1.0
    
    return grad

def _hex_packing_init():
    """Hexagonal close packing initialization."""
    pos = []
    # Try hexagonal lattice with various spacings
    for spacing in [0.18, 0.19, 0.20, 0.21, 0.22, 0.23]:
        rows = int(np.floor(1.0 / spacing)) + 1
        cols = int(np.floor(1.0 / (spacing * np.sqrt(3)/2))) + 1
        for i in range(rows):
            for j in range(cols):
                if len(pos) >= N:
                    break
                x = (j + 0.5 * (i % 2)) * spacing * np.sqrt(3) / 2
                y = i * spacing * 0.75
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pos.append([x, y])
            if len(pos) >= N:
                break
        if len(pos) >= N:
            break
    while len(pos) < N:
        pos.append([0.5, 0.5])
    return np.array(pos[:N])

def _grid_init():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            pos.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pos[:N])

def _random_init(seed=0):
    """Random initialization with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = []
    attempts = 0
    while len(pos) < N and attempts < 10000:
        attempts += 1
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - px, y - py) > 0.07 for px, py in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos)

def _grow_radii(pos, deadline, n_iter=5000, dt=0.0005):
    """
    Iteratively try to grow radii by adjusting positions to create more space.
    Uses a gradient-like approach where each circle moves to increase its radius.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_current = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_obj = float(np.sum(r_current))
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # For each circle, try small moves that increase its radius
        for i in range(n):
            if time.time() > deadline:
                break
            
            x, y = current_pos[i]
            r = r_current[i]
            
            # Determine which constraint is limiting
            boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
            
            # Find nearest neighbor
            dx = current_pos[:, 0] - x
            dy = current_pos[:, 1] - y
            dists = np.sqrt(dx*dx + dy*dy)
            dists[i] = np.inf
            j = int(np.argmin(dists))
            min_dist = dists[j]
            
            # Try moving away from nearest neighbor
            if min_dist < 1e-10:
                continue
            move_x = -dx[j] / min_dist * dt
            move_y = -dy[j] / min_dist * dt
            
            # Also try moving toward center if near boundary
            if boundary < min_dist / 2:
                move_x += (0.5 - x) * dt * 0.5
                move_y += (0.5 - y) * dt * 0.5
            
            new_pos = current_pos.copy()
            new_pos[i, 0] = np.clip(x + move_x, 0.0, 1.0)
            new_pos[i, 1] = np.clip(y + move_y, 0.0, 1.0)
            
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            
            if new_obj > best_obj + 1e-14:
                current_pos = new_pos
                r_current = new_r
                best_pos = current_pos.copy()
                best_obj = new_obj
    
    return best_pos, best_obj

def _simulated_annealing(pos, deadline, t_start=0.3, t_end=0.0002, n_iter=25000, seed=42):
    """Simulated annealing to maximize sum of radii."""
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    best_pos = current_pos.copy()
    best_obj = current_obj
    n = len(current_pos)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (it / n_iter)
        i = rng.randint(n)
        step_size = 0.03 * t + 0.0005
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
        
        new_obj = _objective(new_pos)
        delta = new_obj - current_obj
        
        if delta > 0 or rng.random() < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_pos = current_pos.copy()
                best_obj = current_obj
    
    return best_pos, best_obj

def _local_search(pos, deadline, n_iter=3000, step_size=0.005, seed=42):
    """Local search: iteratively nudge each circle to improve the objective."""
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)
    
    it = 0
    while it < n_iter:
        if time.time() > deadline:
            break
        it += 1
        order = rng.permutation(n)
        improved = False
        for i in order:
            if time.time() > deadline:
                break
            for _ in range(12):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                new_pos = current_pos.copy()
                new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
                new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
                new_obj = _objective(new_pos)
                if new_obj > current_obj + 1e-13:
                    current_pos = new_pos
                    current_obj = new_obj
                    improved = True
                    break
        if not improved:
            step_size *= 0.7
            if step_size < 0.0002:
                break
    
    return current_pos, current_obj

def _gradient_polish(pos, deadline, betas=None):
    """
    Progressive beta gradient polish: run L-BFGS-B with increasing beta values.
    """
    n = len(pos)
    if betas is None:
        betas = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0]
    
    current_pos = pos.copy()
    r_current = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_r = r_current.copy()
    best_obj = float(np.sum(r_current))
    
    for beta in betas:
        if time.time() > deadline - 1.0:
            break
        
        r_init = _max_radius_for_positions(current_pos)
        v0 = np.concatenate([current_pos[:, 0], current_pos[:, 1], r_init])
        
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0))
        for _ in range(n):
            bounds.append((0.0, 1.0))
        for _ in range(n):
            bounds.append((0.0, 0.5))
        
        try:
            maxiter = 1200 if beta >= 100 else 800
            res = minimize(
                _soft_objective, v0, args=(n, beta),
                jac=_soft_gradient, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 100}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 0.5:
            break
        
        v_opt = res.x
        x_opt = v_opt[0:n]
        y_opt = v_opt[n:2*n]
        pos_opt = np.column_stack([x_opt, y_opt])
        pos_opt = np.clip(pos_opt, 0.0, 1.0)
        
        r_exact = _max_radius_for_positions(pos_opt)
        obj_opt = float(np.sum(r_exact))
        
        if _is_valid(pos_opt, eps=1e-9) and obj_opt > best_obj:
            best_pos = pos_opt
            best_r = r_exact
            best_obj = obj_opt
            current_pos = pos_opt.copy()
        elif _is_valid(pos_opt, eps=1e-9):
            current_pos = pos_opt.copy()
    
    return best_pos, best_r, best_obj

def _electrostatic_relaxation(pos, deadline, n_iter=3000, dt=0.0008):
    """
    Electrostatic equilibrium: circles are charged particles with repulsive
    Coulomb forces and attractive harmonic potential toward center.
    Uses symplectic-like integrator with adaptive damping.
    """
    n = len(pos)
    current_pos = pos.copy()
    r_current = _max_radius_for_positions(current_pos)
    best_pos = current_pos.copy()
    best_obj = float(np.sum(r_current))
    
    vx = np.zeros(n)
    vy = np.zeros(n)
    damping = 0.92
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Boundary repulsion
        for i in range(n):
            x, y = current_pos[i]
            r = r_current[i]
            if x < r:
                fx[i] += (r - x) * 80.0
            if x > 1.0 - r:
                fx[i] -= (x - (1.0 - r)) * 80.0
            if y < r:
                fy[i] += (r - y) * 80.0
            if y > 1.0 - r:
                fy[i] -= (y - (1.0 - r)) * 80.0
        
        # Pairwise repulsion (Coulomb-like)
        for i in range(n):
            for j in range(i + 1, n):
                dx = current_pos[j, 0] - current_pos[i, 0]
                dy = current_pos[j, 1] - current_pos[i, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-15:
                    dx = np.random.uniform(-0.001, 0.001)
                    dy = np.random.uniform(-0.001, 0.001)
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 1e-15:
                        continue
                dist = np.sqrt(dist_sq)
                
                overlap = r_current[i] + r_current[j] - dist
                if overlap > 0:
                    force_mag = overlap * 40.0 + 0.01 / max(dist_sq, 1e-6)
                    fx[i] -= force_mag * dx / dist
                    fy[i] -= force_mag * dy / dist
                    fx[j] += force_mag * dx / dist
                    fy[j] += force_mag * dy / dist
        
        # Harmonic attraction toward center
        for i in range(n):
            fx[i] += (0.5 - current_pos[i, 0]) * 2.0
            fy[i] += (0.5 - current_pos[i, 1]) * 2.0
        
        # Update velocities
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        
        # Update positions
        current_pos[:, 0] += vx * dt
        current_pos[:, 1] += vy * dt
        current_pos[:, 0] = np.clip(current_pos[:, 0], 0.0, 1.0)
        current_pos[:, 1] = np.clip(current_pos[:, 1], 0.0, 1.0)
        
        r_new = _max_radius_for_positions(current_pos)
        obj_new = float(np.sum(r_new))
        
        if obj_new > best_obj + 1e-12:
            best_pos = current_pos.copy()
            best_obj = obj_new
            r_current = r_new
    
    return best_pos, best_obj

def _best_fit_local_search(pos, deadline, n_iter=8000, seed=42):
    """Dense grid-based best-fit local search."""
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)
    
    base_step = 0.012
    grid_res = 4
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        improved_any = False
        order = rng.permutation(n)
        step_scale = max(0.2, 1.0 - it / (n_iter * 0.7))
        step = base_step * step_scale
        
        for i in order:
            if time.time() > deadline:
                break
            
            x_i, y_i = current_pos[i]
            best_new_pos = None
            best_new_obj = current_obj
            
            for di in range(-grid_res, grid_res + 1):
                for dj in range(-grid_res, grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = x_i + di * step
                    cy = y_i + dj * step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i] = [cx, cy]
                    new_obj = _objective(new_pos)
                    if new_obj > best_new_obj + 1e-14:
                        best_new_obj = new_obj
                        best_new_pos = new_pos
            
            for _ in range(6):
                angle = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(x_i + mag * np.cos(angle), 0.0, 1.0)
                cy = np.clip(y_i + mag * np.sin(angle), 0.0, 1.0)
                new_pos = current_pos.copy()
                new_pos[i] = [cx, cy]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-14:
                    best_new_obj = new_obj
                    best_new_pos = new_pos
            
            if best_new_pos is not None:
                current_pos = best_new_pos
                current_obj = best_new_obj
                improved_any = True
        
        if not improved_any:
            base_step *= 0.6
            if base_step < 0.0003:
                break
    
    return current_pos, current_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Diverse initializations
    initializations = [_hex_packing_init(), _grid_init()]
    for s in range(6):
        initializations.append(_random_init(seed=s * 1009 + 7))
    
    best_pos = None
    best_obj = -np.inf
    
    # Phase 1: SA + local search from each init
    for init_idx, init_pos in enumerate(initializations):
        if time.time() > deadline - 40:
            break
        
        sa_pos, sa_obj = _simulated_annealing(
            init_pos, deadline=deadline - 35,
            t_start=0.3, t_end=0.0002,
            n_iter=20000, seed=init_idx * 7 + 1
        )
        
        if sa_obj > best_obj:
            best_pos = sa_pos
            best_obj = sa_obj
        
        if time.time() > deadline - 30:
            break
        
        bf_pos, bf_obj = _best_fit_local_search(
            sa_pos, deadline=deadline - 25,
            n_iter=5000, seed=init_idx * 13 + 2
        )
        
        if bf_obj > best_obj:
            best_pos = bf_pos
            best_obj = bf_obj
    
    if best_pos is None:
        pos = _grid_init()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])
    
    # Phase 2: Additional local search passes
    if time.time() < deadline - 30:
        ls_pos, ls_obj = _local_search(
            best_pos, deadline=deadline - 25,
            n_iter=3000, step_size=0.004, seed=555
        )
        if ls_obj > best_obj:
            best_pos = ls_pos
            best_obj = ls_obj
    
    # Phase 3: Electrostatic relaxation
    if time.time() < deadline - 25:
        elec_pos, elec_obj = _electrostatic_relaxation(
            best_pos, deadline=deadline - 20,
            n_iter=2500, dt=0.0008
        )
        if elec_obj > best_obj:
            best_pos = elec_pos
            best_obj = elec_obj
    
    # Phase 4: Grow radii
    if time.time() < deadline - 20:
        grow_pos, grow_obj = _grow_radii(
            best_pos, deadline=deadline - 15,
            n_iter=4000, dt=0.0004
        )
        if grow_obj > best_obj:
            best_pos = grow_pos
            best_obj = grow_obj
    
    # Phase 5: Gradient polish with progressive beta
    if time.time() < deadline - 18:
        grad_pos, grad_r, grad_obj = _gradient_polish(
            best_pos, deadline=deadline - 5,
            betas=[10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
        )
        if grad_obj > best_obj:
            best_pos = grad_pos
            best_obj = grad_obj
    
    # Phase 6: Final best-fit pass
    if time.time() < deadline - 10:
        bf_final, bf_final_obj = _best_fit_local_search(
            best_pos, deadline=deadline - 5,
            n_iter=6000, seed=777
        )
        if bf_final_obj > best_obj:
            best_pos = bf_final
            best_obj = bf_final_obj
    
    # Phase 7: Final fine local search
    if time.time() < deadline - 5:
        final_pos, final_obj = _local_search(
            best_pos, deadline=deadline - 2,
            n_iter=2000, step_size=0.002, seed=999
        )
        if final_obj > best_obj:
            best_pos = final_pos
            best_obj = final_obj
    
    # Phase 8: Ultra-fine polish
    if time.time() < deadline - 1:
        rng = np.random.RandomState(42424)
        for _ in range(300):
            if time.time() > deadline - 0.1:
                break
            i = rng.randint(len(best_pos))
            dx = rng.uniform(-0.0003, 0.0003)
            dy = rng.uniform(-0.0003, 0.0003)
            new_pos = best_pos.copy()
            new_pos[i, 0] = np.clip(best_pos[i, 0] + dx, 0.0, 1.0)
            new_pos[i, 1] = np.clip(best_pos[i, 1] + dy, 0.0, 1.0)
            new_r = _max_radius_for_positions(new_pos)
            new_obj = float(np.sum(new_r))
            if new_obj > best_obj + 1e-13:
                best_pos = new_pos
                best_obj = new_obj
    
    # Build result
    r_final = _max_radius_for_positions(best_pos)
    result = np.column_stack([best_pos, r_final])
    return result

# EVOLVE-BLOCK-END
