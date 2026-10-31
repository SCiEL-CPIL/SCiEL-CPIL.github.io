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


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty for violations. Vectorized."""
    if time.time() > deadline:
        return 1e10
    
    n = N
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    # Penalty for going outside square (smooth)
    # Vectorized: compute violations for all circles
    # Left: r - x, Right: x - (1-r), Bottom: r - y, Top: y - (1-r)
    # We want max(0, violation)^2
    
    # Left
    viol_l = np.maximum(0, radii - pos[:, 0])
    # Right
    viol_r = np.maximum(0, pos[:, 0] - (1.0 - radii))
    # Bottom
    viol_b = np.maximum(0, radii - pos[:, 1])
    # Top
    viol_t = np.maximum(0, pos[:, 1] - (1.0 - radii))
    
    boundary_penalty = np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2)
    
    # Penalty for overlapping (smooth)
    # Compute pairwise distances
    # pos is (n, 2)
    # dist_matrix[i, j] = ||pos[i] - pos[j]||
    
    # Using broadcasting
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (n, n, 2)
    dists = np.sqrt(np.sum(diff**2, axis=2)) # (n, n)
    
    # min_dist_matrix[i, j] = radii[i] + radii[j]
    min_dists = radii[:, np.newaxis] + radii[np.newaxis, :] # (n, n)
    
    # Overlap violation: max(0, min_dist - dist)^2
    # We only care about i < j, but calculating full matrix is fast enough for N=26
    # To avoid double counting and self-interaction, we can just sum upper triangle
    # or divide by 2. Self terms are 0 violation because dist=0, min_dist=2r>0, but we set dists[i,i]=0.
    # Actually, for i=j, dist=0, min_dist=2r. Violation = (2r)^2. This is wrong.
    # We should mask the diagonal.
    
    np.fill_diagonal(dists, np.inf) # Ensure no self-penalty
    
    # Calculate violations
    violations = np.maximum(0, min_dists - dists)
    # Zero out diagonal just in case
    np.fill_diagonal(violations, 0)
    
    # Sum upper triangle to avoid double counting
    # Or just sum all and divide by 2, since violation matrix is symmetric
    overlap_penalty = 0.5 * np.sum(violations**2)
    
    total_penalty = boundary_penalty + overlap_penalty
    
    return -np.sum(radii) + 100.0 * total_penalty


def _make_grid(n, seed=42):
    """Create a grid of n circles inside the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    """Create a hexagonal packing pattern."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
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
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    """One center, rings around. Ensures valid initial packing."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.14
    
    # Inner ring: 6 circles
    r_inner = 0.11
    dist_inner = radii[0] + r_inner + 0.01 # Small gap
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i+1] = r_inner
    
    # Outer ring: remaining circles (26 - 1 - 6 = 19)
    r_outer = 0.09
    dist_outer = radii[0] + 2*r_inner + 0.02 # Approximate
    # Adjust dist_outer to fit in square? 
    # Max radius is 0.5. dist_outer + r_outer <= 0.5?
    # 0.14 + 0.22 + 0.09 = 0.45. OK.
    
    # Place 19 circles in outer ring
    # To avoid overlaps in the ring itself, they need to be spaced.
    # Circumference at dist_outer: 2 * pi * dist_outer
    # Available space per circle: 2 * pi * dist_outer / 19
    # Chord length approx 2 * dist_outer * sin(pi/19)
    # Required chord for non-overlap: 2 * r_outer
    # Check if 2 * r_outer <= 2 * dist_outer * sin(pi/19)
    # 0.18 <= 2 * 0.37 * 0.165 = 0.12? No. 0.18 > 0.12.
    # So they will overlap. We need to reduce r_outer or increase dist_outer.
    # Max dist_outer is limited by square boundaries.
    # Let's just place them and let post-process/optimizer fix it, 
    # but start with smaller radii to be safer.
    r_outer = 0.07
    for i in range(n - 7):
        angle = 2 * np.pi * i / (n - 7)
        pos[i+7] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i+7] = r_outer
        
    return pos, radii

def _make_corners(n, seed=42):
    """Initialize circles near corners and center."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 4 corners
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    for i, c in enumerate(corners):
        pos[i] = c
        radii[i] = 0.1
    
    # Center
    pos[4] = [0.5, 0.5]
    radii[4] = 0.15
    
    # Remaining 21 circles: distribute randomly in remaining space
    rng = np.random.RandomState(seed)
    for i in range(5, n):
        # Random position in [0.2, 0.8]
        pos[i] = rng.uniform(0.2, 0.8, 2)
        radii[i] = 0.08
        
    return pos, radii


def _make_random(n, seed=42):
    """Random initial positions."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        valid = True
        
        # Check boundaries
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.99
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        
        if valid:
            break
    
    # Final clamp
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision polish pass with tighter tolerances."""
    pos = pos.copy()
    radii = radii.copy()
    
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((radii[i], 1.0 - radii[i]))  # x
        bounds.append((radii[i], 1.0 - radii[i]))  # y
    for i in range(N):
        bounds.append((1e-4, 0.3))  # radius
    
    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success:
            pos_opt = result.x[:2*N].reshape(N, 2)
            radii_opt = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Generate initial configurations
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    configs.append(_make_corners(N))
    
    rng = np.random.RandomState(42)
    for seed in range(5):
        configs.append(_make_random(N, seed=seed + 100))
    
    # Phase 1: L-BFGS-B optimization from each config
    # Use vectorized objective for speed
    for pos, radii in configs:
        if time.time() > deadline - 30:
            break
        
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))  # x
            bounds.append((0.01, 0.99))  # y
        for i in range(N):
            bounds.append((0.01, 0.3))  # radius
        
        try:
            # Increase maxiter since objective is faster
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement on best found
    if best_pos is not None and time.time() < deadline - 20:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        try:
            result_nm = minimize(lambda x: _objective(x, deadline), x0, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'maxfev': 15000, 'xatol': 1e-8, 'fatol': 1e-8})
            pos_opt = result_nm.x[:2*N].reshape(N, 2)
            radii_opt = result_nm.x[2*N:]
            update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global exploration
    if time.time() < deadline - 40:
        try:
            def de_objective(x):
                return _objective(x, deadline)
            
            bounds_de = []
            for i in range(N):
                bounds_de.append((0.05, 0.95))  # x
                bounds_de.append((0.05, 0.95))  # y
            for i in range(N):
                bounds_de.append((0.02, 0.25))  # radius
            
            de_result = differential_evolution(de_objective, bounds_de, maxiter=80,
                                               popsize=15, seed=42, tol=1e-8,
                                               polish=False, mutation=(0.3, 1.0),
                                               recombination=0.7)
            
            if de_result.x is not None:
                pos_opt = de_result.x[:2*N].reshape(N, 2)
                radii_opt = de_result.x[2*N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 4: Polish the best result
    if best_pos is not None and time.time() < deadline - 5:
        best_pos, best_radii = _polish(best_pos, best_radii, deadline)
        
        # Extra polish pass if time permits
        if time.time() < deadline - 2:
            best_pos, best_radii = _polish(best_pos, best_radii, deadline)
    
    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
