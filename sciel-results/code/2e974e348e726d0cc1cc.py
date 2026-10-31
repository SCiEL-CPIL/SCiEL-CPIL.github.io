"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    # Check inside square
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x):
    """Objective: minimize negative sum of radii + smooth penalty for violations."""
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = np.maximum(x[2*n:], 1e-6)
    
    # Smooth penalty for boundary violations using soft-plus
    boundary_pen = 0.0
    for i in range(n):
        # Distance to each boundary (negative means violation)
        d_left = pos[i, 0] - radii[i]
        d_right = (1.0 - pos[i, 0]) - radii[i]
        d_bottom = pos[i, 1] - radii[i]
        d_top = (1.0 - pos[i, 1]) - radii[i]
        
        # Soft penalty: smooth approximation of max(0, -d)^2
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < 0:
                boundary_pen += d * d
    
    # Smooth penalty for overlaps using soft constraint
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            min_dist = radii[i] + radii[j]
            violation = min_dist - dist
            if violation > 0:
                overlap_pen += violation * violation
    
    # Adaptive penalty weight
    penalty = boundary_pen + overlap_pen
    weight = 500.0 + 500.0 * min(penalty * 10, 10.0)
    
    return -np.sum(radii) + weight * penalty


def _initial_grid():
    """Create a 5x5 grid of circles with the 26th placed in a corner gap."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid: 25 circles
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            pos[idx, 0] = (i + 0.5) / 5.0
            pos[idx, 1] = (j + 0.5) / 5.0
            radii[idx] = 0.09
    
    # 26th circle: place in a corner
    pos[25, 0] = 0.1
    pos[25, 1] = 0.1
    radii[25] = 0.08
    
    return pos, radii


def _initial_hexagonal():
    """Create a hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.09
    # Row 0: 5 circles
    for i in range(5):
        idx = i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1
        radii[idx] = r
    
    # Row 1: 4 circles (offset)
    dy = r * np.sqrt(3)
    for i in range(4):
        idx = 5 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + dy
        radii[idx] = r
    
    # Row 2: 5 circles
    for i in range(5):
        idx = 9 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 2 * dy
        radii[idx] = r
    
    # Row 3: 4 circles (offset)
    for i in range(4):
        idx = 14 + i
        pos[idx, 0] = 0.1 + r + i * 2 * r
        pos[idx, 1] = 0.1 + 3 * dy
        radii[idx] = r
    
    # Row 4: 5 circles
    for i in range(5):
        idx = 18 + i
        pos[idx, 0] = 0.1 + i * 2 * r
        pos[idx, 1] = 0.1 + 4 * dy
        radii[idx] = r
    
    # 26th circle: place in center gap
    pos[25, 0] = 0.5
    pos[25, 1] = 0.5
    radii[25] = 0.06
    
    return pos, radii


def _initial_center_ring():
    """Create a center circle with rings around it."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Center circle
    pos[0, 0] = 0.5
    pos[0, 1] = 0.5
    radii[0] = 0.12
    
    # Inner ring: 6 circles
    r_inner = 0.10
    dist_inner = radii[0] + r_inner
    for i in range(6):
        idx = 1 + i
        angle = i * np.pi / 3.0
        pos[idx, 0] = 0.5 + dist_inner * np.cos(angle)
        pos[idx, 1] = 0.5 + dist_inner * np.sin(angle)
        radii[idx] = r_inner
    
    # Outer ring: 12 circles
    r_outer = 0.08
    dist_outer = radii[0] + 2 * r_inner
    for i in range(12):
        idx = 7 + i
        angle = i * np.pi / 6.0
        pos[idx, 0] = 0.5 + dist_outer * np.cos(angle)
        pos[idx, 1] = 0.5 + dist_outer * np.sin(angle)
        radii[idx] = r_outer
    
    # 7 more circles to reach 26
    for i in range(7):
        idx = 19 + i
        # Place in corners and edges
        if i == 0:
            pos[idx, 0] = 0.08
            pos[idx, 1] = 0.08
        elif i == 1:
            pos[idx, 0] = 0.92
            pos[idx, 1] = 0.08
        elif i == 2:
            pos[idx, 0] = 0.08
            pos[idx, 1] = 0.92
        elif i == 3:
            pos[idx, 0] = 0.92
            pos[idx, 1] = 0.92
        elif i == 4:
            pos[idx, 0] = 0.5
            pos[idx, 1] = 0.06
        elif i == 5:
            pos[idx, 0] = 0.5
            pos[idx, 1] = 0.94
        else:
            pos[idx, 0] = 0.06
            pos[idx, 1] = 0.5
        radii[idx] = 0.06
    
    return pos, radii


def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, size=(N, 2))
    radii = rng.uniform(0.05, 0.12, size=N)
    return pos, radii


def _post_process(pos, radii):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        valid = True
        
        # Check and fix boundaries
        for i in range(N):
            # Left
            if pos[i, 0] < radii[i]:
                radii[i] = max(pos[i, 0] * 0.99, 1e-4)
                valid = False
            # Right
            if pos[i, 0] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 0]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            # Bottom
            if pos[i, 1] < radii[i]:
                radii[i] = min(radii[i], pos[i, 1] * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
            # Top
            if pos[i, 1] > 1.0 - radii[i]:
                radii[i] = min(radii[i], (1.0 - pos[i, 1]) * 0.99)
                radii[i] = max(radii[i], 1e-4)
                valid = False
        
        # Check and fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(radii[i] * scale, 1e-4)
                    radii[j] = max(radii[j] * scale, 1e-4)
                    valid = False
        
        if valid:
            break
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision local search to squeeze out more sum of radii."""
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Multiple polish passes with different methods
    methods = [
        ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}),
        ('Powell', {'maxiter': 500, 'ftol': 1e-12}),
    ]
    
    for method, opts in methods:
        if time.time() > deadline - 3:
            break
        try:
            result = minimize(_objective, x0, method=method, bounds=bounds, options=opts)
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = np.maximum(x_opt[2*N:], 1e-5)
            
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def _optimize_from(pos, radii, deadline):
    """Run optimization from given initial configuration."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
    for i in range(N):
        bounds.append((0.01, 0.25))  # radius
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Try multiple optimization methods
    methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
    
    for method in methods:
        if time.time() > deadline - 8:
            break
        
        try:
            if method == 'L-BFGS-B':
                result = minimize(_objective, x0, method=method, bounds=bounds,
                                  options={'maxiter': 800, 'ftol': 1e-12})
            elif method == 'Powell':
                result = minimize(_objective, x0, method=method, bounds=bounds,
                                  options={'maxiter': 400, 'ftol': 1e-12})
            else:  # Nelder-Mead
                result = minimize(_objective, x0, method=method,
                                  options={'maxiter': 800, 'fatol': 1e-12})
            
            if result.fun < _objective(x0):
                x_opt = result.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                radii_opt = np.maximum(x_opt[2*N:], 1e-4)
                
                # Post-process to ensure validity
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x0 = np.concatenate([best_pos.flatten(), best_radii])
        except Exception:
            pass
    
    # Polish pass
    if time.time() > deadline - 5:
        pass
    else:
        try:
            pos_pol, radii_pol, score_pol = _polish(best_pos, best_radii, deadline)
            if score_pol > best_score:
                best_score = score_pol
                best_pos = pos_pol.copy()
                best_radii = radii_pol.copy()
        except Exception:
            pass
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Try different initial configurations
    init_configs = []
    
    # Grid
    pos, radii = _initial_grid()
    init_configs.append((pos, radii, 'grid'))
    
    # Hexagonal
    pos, radii = _initial_hexagonal()
    init_configs.append((pos, radii, 'hex'))
    
    # Center ring
    pos, radii = _initial_center_ring()
    init_configs.append((pos, radii, 'ring'))
    
    # Random with different seeds
    for seed in [42, 123, 456, 789]:
        pos, radii = _initial_random(seed)
        init_configs.append((pos, radii, f'rand_{seed}'))
    
    # Perturbed grid
    rng = np.random.RandomState(99)
    pos, radii = _initial_grid()
    pos_pert = pos + rng.normal(0, 0.02, size=pos.shape)
    pos_pert = np.clip(pos_pert, 0.05, 0.95)
    init_configs.append((pos_pert, radii, 'grid_pert'))
    
    # Perturbed hex
    rng2 = np.random.RandomState(77)
    pos, radii = _initial_hexagonal()
    pos_pert2 = pos + rng2.normal(0, 0.015, size=pos.shape)
    pos_pert2 = np.clip(pos_pert2, 0.05, 0.95)
    init_configs.append((pos_pert2, radii, 'hex_pert'))
    
    # Run optimization from each initial config
    for pos, radii, name in init_configs:
        if time.time() > deadline - 15:
            break
        
        pos_opt, radii_opt, score = _optimize_from(pos, radii, deadline)
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
            best_radii = radii_opt.copy()
    
    # If we didn't find a valid solution, fall back to simple grid
    if best_pos is None or best_score < 0:
        side = 5
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                if idx < N:
                    pos[idx, 0] = (i + 0.5) / side
                    pos[idx, 1] = (j + 0.5) / side
                    radii[idx] = 0.09
        # 26th circle
        pos[25, 0] = 0.1
        pos[25, 1] = 0.9
        radii[25] = 0.07
        best_pos = pos
        best_radii = radii
    
    # Final polish and post-processing to ensure validity
    if time.time() > deadline - 3:
        best_pos, best_radii = _post_process(best_pos, best_radii)
    else:
        try:
            best_pos, best_radii, _ = _polish(best_pos, best_radii, deadline)
        except Exception:
            best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
