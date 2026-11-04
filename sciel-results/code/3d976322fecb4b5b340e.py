import numpy as np
import time

N = 26

def _max_radius_for_circle(i, centers, radii):
    """Compute the maximum radius circle i can have given other circles' positions."""
    x, y = centers[i]
    max_r = min(x, 1 - x, y, 1 - y)
    for j in range(N):
        if j == i:
            continue
        dx = x - centers[j][0]
        dy = y - centers[j][1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0:
            max_r = min(max_r, dist - radii[j])
    return max(0.0, max_r)

def _compute_radii(centers, radii):
    """Vectorized: compute max feasible radius for each circle given others fixed."""
    n = len(centers)
    max_radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(n):
            if j == i:
                continue
            dx = x - centers[j][0]
            dy = y - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-12:
                max_r = min(max_r, dist - radii[j])
        max_radii[i] = max(0.0, max_r)
    return max_radii

def _eval_packing(centers, radii):
    """Evaluate a packing: return (sum_of_radii, is_valid)."""
    sum_r = float(np.sum(radii))
    
    for i in range(N):
        if centers[i][0] < -1e-9 or centers[i][0] > 1 + 1e-9:
            return sum_r, False
        if centers[i][1] < -1e-9 or centers[i][1] > 1 + 1e-9:
            return sum_r, False
        if radii[i] < -1e-9:
            return sum_r, False
        if radii[i] > centers[i][0] + 1e-9:
            return sum_r, False
        if radii[i] > 1 - centers[i][0] + 1e-9:
            return sum_r, False
        if radii[i] > centers[i][1] + 1e-9:
            return sum_r, False
        if radii[i] > 1 - centers[i][1] + 1e-9:
            return sum_r, False
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return sum_r, False
    
    return sum_r, True

def _fix_feasibility(centers, radii, tol=1e-9):
    """Shrink radii iteratively until packing is feasible. Returns (radii, is_feasible)."""
    radii = radii.copy()
    for _ in range(200):
        feasible = True
        # Boundary check
        for i in range(N):
            max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            if radii[i] > max_r + tol:
                radii[i] = max(0.0, max_r)
                feasible = False
        # Pairwise check
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    # Split the violation
                    excess = radii[i] + radii[j] - dist
                    if excess > 0:
                        radii[i] = max(0.0, radii[i] - excess / 2.0)
                        radii[j] = max(0.0, radii[j] - excess / 2.0)
                        feasible = False
        if feasible:
            break
    # Final validation
    _, valid = _eval_packing(centers, radii)
    return radii, valid

def _best_fit_local_search(centers, radii, max_iters=300, deadline=None, rng=None):
    """Iteratively move each circle to maximize its radius given others fixed."""
    if rng is None:
        rng = np.random.default_rng()
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = centers[i].copy()
            best_r = radii[i]
            
            # Current position
            max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            for j in range(N):
                if j == i:
                    continue
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-12:
                    max_r = min(max_r, dist - radii[j])
            max_r = max(0.0, max_r)
            if max_r > best_r + 1e-12:
                best_r = max_r
            
            # Random perturbations with adaptive step size
            step_sizes = [0.02, 0.05, 0.1, 0.15]
            for step in step_sizes:
                for _ in range(15):
                    angle = rng.uniform(0, 2 * np.pi)
                    dist = rng.uniform(0, step)
                    nx = centers[i][0] + dist * np.cos(angle)
                    ny = centers[i][1] + dist * np.sin(angle)
                    if -0.01 <= nx <= 1.01 and -0.01 <= ny <= 1.01:
                        nx = np.clip(nx, 0, 1)
                        ny = np.clip(ny, 0, 1)
                        max_r = min(nx, 1 - nx, ny, 1 - ny)
                        for j in range(N):
                            if j == i:
                                continue
                            dx = nx - centers[j][0]
                            dy = ny - centers[j][1]
                            d = np.sqrt(dx * dx + dy * dy)
                            if d > 1e-12:
                                max_r = min(max_r, d - radii[j])
                        max_r = max(0.0, max_r)
                        
                        if max_r > best_r + 1e-12:
                            best_r = max_r
                            best_pos = np.array([nx, ny])
                            improved = True
        
        centers[i] = best_pos
        radii[i] = best_r
    
    return centers, radii

def _gradient_polish(centers, radii, deadline=None, max_steps=500):
    """
    Gradient-based polish using a soft surrogate.
    Objective: maximize sum(radii) - lambda * penalty(violations)
    Uses soft-min for smoothness.
    """
    if deadline is not None and time.time() > deadline:
        return centers, radii
    
    centers = centers.copy().astype(float)
    radii = radii.copy().astype(float)
    
    # We optimize positions and radii jointly
    # Soft penalty for overlap: for pair (i,j), penalty = max(0, r_i + r_j - dist)^2
    # Soft boundary: max(0, r_i - min(x_i, 1-x_i, y_i, 1-y_i))^2
    
    beta = 50.0  # softness parameter
    
    def _objective_and_grad(pos, rad):
        """
        pos: shape (N, 2)
        rad: shape (N,)
        Returns: (value, grad_pos, grad_rad)
        value = sum(rad) - penalty
        """
        n = len(pos)
        value = np.sum(rad)
        grad_pos = np.zeros((n, 2))
        grad_rad = np.ones(n)
        
        # Boundary constraints
        for i in range(n):
            x, y = pos[i]
            # max_r from boundary
            bound_r = min(x, 1 - x, y, 1 - y)
            if rad[i] > bound_r:
                excess = rad[i] - bound_r
                # penalty = excess^2
                value -= excess * excess
                grad_rad[i] -= 2 * excess
                # gradient w.r.t. position: d(bound_r)/d(x,y)
                # bound_r = min(x, 1-x, y, 1-y)
                # For soft version, use the active constraint
                # d(min)/dx: if x is min, grad = 1; if 1-x is min, grad = -1
                # We approximate: the gradient of min is the gradient of the argmin
                if x == bound_r:
                    grad_pos[i, 0] += 2 * excess
                elif 1 - x == bound_r:
                    grad_pos[i, 0] -= 2 * excess
                elif y == bound_r:
                    grad_pos[i, 1] += 2 * excess
                elif 1 - y == bound_r:
                    grad_pos[i, 1] -= 2 * excess
        
        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    continue
                overlap = rad[i] + rad[j] - dist
                if overlap > 0:
                    # penalty = overlap^2
                    value -= overlap * overlap
                    grad_rad[i] -= 2 * overlap
                    grad_rad[j] -= 2 * overlap
                    # d(overlap)/d(pos_i) = -d(dist)/d(pos_i) = (dx, dy)/dist
                    # d(overlap)/d(pos_j) = d(dist)/d(pos_j) = (-dx, -dy)/dist
                    grad_pos[i, 0] += 2 * overlap * dx / dist
                    grad_pos[i, 1] += 2 * overlap * dy / dist
                    grad_pos[j, 0] += 2 * overlap * (-dx) / dist
                    grad_pos[j, 1] += 2 * overlap * (-dy) / dist
        
        return value, grad_pos, grad_rad
    
    # Simple gradient ascent with adaptive step
    lr = 0.01
    best_value = -np.inf
    best_pos = centers.copy()
    best_rad = radii.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        val, gp, gr = _objective_and_grad(centers, radii)
        
        # Clip radii to be non-negative
        radii = np.maximum(radii, 0)
        
        if val > best_value:
            best_value = val
            best_pos = centers.copy()
            best_rad = radii.copy()
        
        # Gradient ascent
        centers = centers + lr * gp
        radii = radii + lr * gr
        radii = np.maximum(radii, 0)
        centers = np.clip(centers, 0, 1)
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            val_new, _, _ = _objective_and_grad(centers, radii)
            if val_new < best_value - 0.01:
                lr *= 0.5
        
        # Occasionally recompute exact radii
        if step % 100 == 0:
            exact_r = _compute_radii(centers, radii)
            radii = exact_r
    
    # Final: compute exact radii
    radii = _compute_radii(best_pos, best_rad)
    best_pos, best_rad = _fix_feasibility(best_pos, radii)
    
    return best_pos, best_rad

def _high_precision_polish(centers, radii, deadline=None):
    """Final polish: iterate best-fit with very small steps for precision."""
    if deadline is not None and time.time() > deadline:
        return centers, radii
    
    centers = centers.copy()
    radii = radii.copy()
    rng = np.random.default_rng(12345)
    
    for iteration in range(200):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = centers[i].copy()
            best_r = radii[i]
            
            # Very small perturbations
            for _ in range(30):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, 0.01)
                nx = centers[i][0] + dist * np.cos(angle)
                ny = centers[i][1] + dist * np.sin(angle)
                nx = np.clip(nx, 0, 1)
                ny = np.clip(ny, 0, 1)
                
                max_r = min(nx, 1 - nx, ny, 1 - ny)
                for j in range(N):
                    if j == i:
                        continue
                    dx = nx - centers[j][0]
                    dy = ny - centers[j][1]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-12:
                        max_r = min(max_r, d - radii[j])
                max_r = max(0.0, max_r)
                
                if max_r > best_r + 1e-14:
                    best_r = max_r
                    best_pos = np.array([nx, ny])
                    improved = True
            
            centers[i] = best_pos
            radii[i] = best_r
        
        if not improved:
            break
    
    return centers, radii

def _hex_init(seed=42):
    """Create hexagonal-like initial packing with proper radii."""
    rng = np.random.default_rng(seed)
    centers = []
    radii = []
    
    r0 = 0.09
    dy = r0 * 2 * 0.866
    for row in range(7):
        x_offset = 0 if row % 2 == 0 else r0
        for col in range(6):
            x = r0 + col * 2 * r0 + x_offset
            y = r0 + row * dy
            if x < 1 and y < 1:
                centers.append((x, y))
                radii.append(r0)
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    while len(centers) < N:
        centers.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
        radii.append(0.05)
    
    centers = np.array(centers[:N])
    radii = np.array(radii[:N])
    
    # Compute proper radii
    for i in range(N):
        radii[i] = _max_radius_for_circle(i, centers, radii)
    
    return centers, radii

def _grid_init():
    """Create initial grid packing with proper radii."""
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append(((i + 0.5) / 5.0, (j + 0.5) / 5.0))
            radii.append(0.09)
    centers.append((0.5, 0.5))
    radii.append(0.09)
    
    centers = np.array(centers[:N])
    radii = np.array(radii[:N])
    
    for i in range(N):
        radii[i] = _max_radius_for_circle(i, centers, radii)
    
    return centers, radii

def _random_init(seed=42):
    """Create random initial packing with proper radii."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = np.full(N, 0.04)
    
    for i in range(N):
        radii[i] = _max_radius_for_circle(i, centers, radii)
    
    return centers, radii

def _structured_init(seed=42):
    """Structured initialization: mix of hexagonal positions with jitter."""
    rng = np.random.default_rng(seed)
    centers = []
    radii = []
    
    # Create a 5x5 grid with some perturbation
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0 + rng.uniform(-0.03, 0.03)
            y = (j + 0.5) / 5.0 + rng.uniform(-0.03, 0.03)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append((x, y))
            radii.append(0.09)
    
    # Add 6 more in gaps
    gap_positions = [
        (0.5/5, 0.5/5), (4.5/5, 0.5/5), (0.5/5, 4.5/5), (4.5/5, 4.5/5),
        (2.5/5, 0.5/5), (2.5/5, 4.5/5)
    ]
    for x, y in gap_positions:
        if len(centers) >= N:
            break
        x += rng.uniform(-0.02, 0.02)
        y += rng.uniform(-0.02, 0.02)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        centers.append((x, y))
        radii.append(0.06)
    
    centers = np.array(centers[:N])
    radii = np.array(radii[:N])
    
    for i in range(N):
        radii[i] = _max_radius_for_circle(i, centers, radii)
    
    return centers, radii

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Generate multiple initial configurations
    inits = []
    
    c, r = _grid_init()
    inits.append((c, r, "grid"))
    
    c, r = _hex_init(42)
    inits.append((c, r, "hex42"))
    
    c, r = _hex_init(123)
    inits.append((c, r, "hex123"))
    
    c, r = _random_init(42)
    inits.append((c, r, "rand42"))
    
    c, r = _random_init(789)
    inits.append((c, r, "rand789"))
    
    c, r = _structured_init(42)
    inits.append((c, r, "struct42"))
    
    c, r = _structured_init(2020)
    inits.append((c, r, "struct2020"))
    
    # Run optimization on each initialization
    for idx, (centers, radii, name) in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            # Phase 1: Best-fit local search
            c_opt, r_opt = _best_fit_local_search(centers, radii, 
                                                   max_iters=200, 
                                                   deadline=deadline)
            
            # Fix feasibility
            r_opt, valid = _fix_feasibility(c_opt, r_opt)
            if not valid:
                # Try harder fix
                r_opt = np.minimum(r_opt, 0.05)
                r_opt, valid = _fix_feasibility(c_opt, r_opt)
            
            score = float(np.sum(r_opt))
            if valid and score > best_score:
                best_score = score
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
            
            # Phase 2: Gradient polish (if time remains)
            if time.time() < deadline - 20:
                c_grad, r_grad = _gradient_polish(c_opt, r_opt, deadline=deadline, max_steps=300)
                r_grad, valid2 = _fix_feasibility(c_grad, r_grad)
                if valid2:
                    score2 = float(np.sum(r_grad))
                    if score2 > best_score:
                        best_score = score2
                        best_centers = c_grad.copy()
                        best_radii = r_grad.copy()
            
            # Phase 3: High precision polish (if time remains)
            if time.time() < deadline - 10:
                c_hp, r_hp = _high_precision_polish(best_centers, best_radii, deadline=deadline)
                r_hp, valid3 = _fix_feasibility(c_hp, r_hp)
                if valid3:
                    score3 = float(np.sum(r_hp))
                    if score3 > best_score:
                        best_score = score3
                        best_centers = c_hp.copy()
                        best_radii = r_hp.copy()
                        
        except Exception:
            pass
    
    # If no valid solution found, use a simple grid fallback
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    # Final validation
    final_score, final_valid = _eval_packing(best_centers, best_radii)
    if not final_valid:
        # Try to fix
        best_radii, final_valid = _fix_feasibility(best_centers, best_radii)
        if not final_valid:
            # Fallback
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) >= N:
                        break
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append([x, y, r])
            return np.array(cells)
    
    return np.column_stack((best_centers, best_radii))
