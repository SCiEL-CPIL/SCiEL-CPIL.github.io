"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _compute_energy(pos, radii, penalty_weight=1000.0):
    """Compute energy: -sum(radii) + penalty * violations."""
    # Boundary violations
    bv = np.maximum(0, radii - pos[:, 0]) + np.maximum(0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0, radii - pos[:, 1]) + np.maximum(0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _compute_gradient(pos, radii, penalty_weight=1000.0):
    """Compute gradient of energy with respect to positions and radii."""
    grad_pos = np.zeros_like(pos)
    grad_r = np.zeros_like(radii)
    
    # Boundary gradients
    for i in range(N):
        r = radii[i]
        x, y = pos[i]
        # Left boundary
        if r > x:
            grad_pos[i, 0] += 2 * penalty_weight * (r - x)
            grad_r[i] += 2 * penalty_weight * (r - x)
        # Right boundary
        if x > 1.0 - r:
            grad_pos[i, 0] += 2 * penalty_weight * (x - (1.0 - r))
            grad_r[i] += -2 * penalty_weight * (x - (1.0 - r))
        # Bottom boundary
        if r > y:
            grad_pos[i, 1] += 2 * penalty_weight * (r - y)
            grad_r[i] += 2 * penalty_weight * (r - y)
        # Top boundary
        if y > 1.0 - r:
            grad_pos[i, 1] += 2 * penalty_weight * (y - (1.0 - r))
            grad_r[i] += -2 * penalty_weight * (y - (1.0 - r))
    
    # Overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            min_dist_sq = min_dist * min_dist
            
            if min_dist_sq > dist_sq:
                viol = min_dist_sq - dist_sq
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                # Gradient of (min_dist_sq - dist_sq) w.r.t. positions
                factor = 2 * penalty_weight * viol / dist
                grad_pos[i, 0] += factor * dx
                grad_pos[i, 1] += factor * dy
                grad_pos[j, 0] -= factor * dx
                grad_pos[j, 1] -= factor * dy
                # Gradient w.r.t. radii
                grad_r[i] += 2 * penalty_weight * viol * 2 * radii[i]
                grad_r[j] += 2 * penalty_weight * viol * 2 * radii[j]
    
    # Objective gradient: -sum(radii)
    grad_r -= 1.0
    
    return grad_pos, grad_r

def _project_feasible(pos, radii):
    """Project positions and radii to be feasible (inside square, no overlaps)."""
    pos = pos.copy()
    radii = radii.copy()
    
    # Ensure inside square
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max_r * 0.999
    
    # Resolve overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r and dist > 1e-10:
                    # Shrink radii to resolve
                    scale = dist / sum_r * 0.999
                    radii[i] = max(radii[i] * scale, 0.001)
                    radii[j] = max(radii[j] * scale, 0.001)
                    changed = True
        if not changed:
            break
    
    return pos, radii

def _gradient_descent(pos, radii, max_iter=2000, lr=0.01, deadline=None):
    """Gradient descent with adaptive learning rate."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    penalty_weight = 500.0
    current_energy = _compute_energy(pos, radii, penalty_weight)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad_pos, grad_r = _compute_gradient(pos, radii, penalty_weight)
        
        # Adaptive learning rate
        norm = np.sqrt(np.sum(grad_pos**2) + np.sum(grad_r**2)) + 1e-10
        step = lr / norm
        
        new_pos = pos - step * grad_pos
        new_radii = radii - step * grad_r
        
        # Clamp
        new_pos = np.clip(new_pos, 0.01, 0.99)
        new_radii = np.clip(new_radii, 0.005, 0.4)
        
        new_energy = _compute_energy(new_pos, new_radii, penalty_weight)
        
        if new_energy < current_energy:
            pos = new_pos
            radii = new_radii
            current_energy = new_energy
            
            # Update best
            pos_pp, radii_pp = _project_feasible(pos, radii)
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_pos, best_radii

def _local_refine(pos, radii, deadline):
    """Local refinement: try to increase each radius by moving it to a better position."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for _ in range(100):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try to increase radius of circle i
            old_r = radii[i]
            new_r = old_r * 1.02
            
            # Check if we can place circle i with new_r
            found = False
            for attempt in range(20):
                # Try random nearby positions
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, 0.05)
                new_x = pos[i, 0] + dist * np.cos(angle)
                new_y = pos[i, 1] + dist * np.sin(angle)
                
                # Check boundaries
                if new_x < new_r or new_x > 1 - new_r or new_y < new_r or new_y > 1 - new_r:
                    continue
                
                # Check overlaps
                valid = True
                for j in range(N):
                    if j == i:
                        continue
                    dx = new_x - pos[j, 0]
                    dy = new_y - pos[j, 1]
                    if np.sqrt(dx*dx + dy*dy) < new_r + radii[j]:
                        valid = False
                        break
                
                if valid:
                    pos[i] = [new_x, new_y]
                    radii[i] = new_r
                    improved = True
                    found = True
                    break
            
            if not found:
                # Try to move circle i to a position where it can be larger
                for attempt in range(10):
                    new_x = np.random.uniform(0.05, 0.95)
                    new_y = np.random.uniform(0.05, 0.95)
                    
                    max_r = min(new_x, 1-new_x, new_y, 1-new_y)
                    for j in range(N):
                        if j == i:
                            continue
                        dx = new_x - pos[j, 0]
                        dy = new_y - pos[j, 1]
                        max_r = min(max_r, np.sqrt(dx*dx + dy*dy) - radii[j])
                    
                    if max_r > old_r * 1.01:
                        pos[i] = [new_x, new_y]
                        radii[i] = max_r * 0.98
                        improved = True
                        break
        
        if not improved:
            break
        
        # Update best
        pos_pp, radii_pp = _project_feasible(pos, radii)
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _initial_structured(seed=0):
    """Structured initialization: corners, edges, center."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corners
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.15
    
    # 4 edge midpoints
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for i, (cx, cy) in enumerate(edges):
        pos[4+i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[4+i] = 0.12
    
    # Center
    pos[8] = [0.5 + rng.uniform(-0.02, 0.02), 0.5 + rng.uniform(-0.02, 0.02)]
    radii[8] = 0.14
    
    # Remaining circles: place in gaps
    for i in range(9, N):
        # Try to find a good position
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for attempt in range(100):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            max_r = min(x, 1-x, y, 1-y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > best_r:
                best_r = max_r
                best_x = x
                best_y = y
        
        if best_r < 0.01:
            best_r = 0.01
            # Find a position with at least this radius
            for attempt in range(50):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                max_r = min(x, 1-x, y, 1-y)
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - radii[j])
                if max_r >= 0.01:
                    best_x, best_y = x, y
                    best_r = max_r
                    break
        
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.95
    
    return pos, radii

def _initial_grid(seed=0):
    """Grid-based initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Create a 5x5 grid and pick 26 positions
    grid_positions = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            grid_positions.append((x, y))
    
    # Shuffle and pick 26
    rng.shuffle(grid_positions)
    
    for i in range(N):
        x, y = grid_positions[i]
        # Add some randomness
        x += rng.uniform(-0.02, 0.02)
        y += rng.uniform(-0.02, 0.02)
        pos[i] = [x, y]
        radii[i] = 0.1
    
    # Optimize radii
    for i in range(N):
        max_r = min(pos[i, 0], 1-pos[i, 0], pos[i, 1], 1-pos[i, 1])
        for j in range(N):
            if j != i:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
        radii[i] = max(max_r, 0.01) * 0.95
    
    return pos, radii

def _initial_random(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.12, N)
    pos, radii = _project_feasible(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """Simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    penalty_weight = 500.0
    current_energy = _compute_energy(pos, radii, penalty_weight)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Perturb: move one circle
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.03)
        dy = rng.normal(0, temp * 0.03)
        dr = rng.normal(0, temp * 0.02)
        
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.4)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_energy = _compute_energy(pos, radii, penalty_weight)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        # Periodically check validity and update best
        if step % 200 == 0:
            pos_pp, radii_pp = _project_feasible(pos, radii)
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _polish_slsqp(pos, radii, deadline):
    """Polish with SLSQP."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = np.maximum(x[2*N:], 1e-6)
        return _compute_energy(p, r, 2000.0)
    
    try:
        result = minimize(objective, x, method='SLSQP', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < objective(x) - 1e-10:
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    pos_out, radii_out = _project_feasible(pos_out, radii_out)
    
    return pos_out, radii_out

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
    
    for _ in range(200):
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_s, rad_s = _initial_structured(seed=0)
    inits.append((pos_s, rad_s))
    
    pos_g, rad_g = _initial_grid(seed=0)
    inits.append((pos_g, rad_g))
    
    for i in range(3):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_s.copy()
    best_radii = rad_s.copy()
    best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        current_pos = pos_init.copy()
        current_radii = rad_init.copy()
        
        # Stage 1: Gradient descent
        if time.time() < deadline - 15:
            pos_gd, rad_gd = _gradient_descent(current_pos, current_radii, max_iter=1500, lr=0.02, deadline=deadline-15)
            if np.sum(rad_gd) > best_score:
                best_score = np.sum(rad_gd)
                best_pos = pos_gd.copy()
                best_radii = rad_gd.copy()
            current_pos = pos_gd
            current_radii = rad_gd
        
        # Stage 2: Local refinement
        if time.time() < deadline - 10:
            pos_lr, rad_lr = _local_refine(current_pos, current_radii, deadline=deadline-10)
            if np.sum(rad_lr) > best_score:
                best_score = np.sum(rad_lr)
                best_pos = pos_lr.copy()
                best_radii = rad_lr.copy()
            current_pos = pos_lr
            current_radii = rad_lr
        
        # Stage 3: Simulated annealing
        if time.time() < deadline - 8:
            pos_sa, rad_sa = _simulated_annealing(
                current_pos, current_radii,
                n_steps=2000, t0=0.03, t_end=0.0005,
                seed=idx * 100, deadline=deadline-8
            )
            if np.sum(rad_sa) > best_score:
                best_score = np.sum(rad_sa)
                best_pos = pos_sa.copy()
                best_radii = rad_sa.copy()
        
        # Stage 4: SLSQP polish
        if time.time() < deadline - 5:
            pos_pol, rad_pol = _polish_slsqp(best_pos, best_radii, deadline)
            if np.sum(rad_pol) > best_score:
                best_score = np.sum(rad_pol)
                best_pos = pos_pol.copy()
                best_radii = rad_pol.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: simple grid
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
